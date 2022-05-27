import momepy
import fiona
import copy
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
import torch
from shapely.geometry import Point, LineString
from itertools import chain
from numbers import Number

from utils.from_networkx import from_networkx
from utils.gdf_to_nx import gdf_to_nx
from utils.remove_false_nodes import remove_false_nodes
from utils.utils import apply_agg, convert_categorical_features_to_one_hot
from utils.constants import dataset_root, osmnx_buffer, fields_to_ignore
from utils.constants import included_places, full_dataset_label, saved_data_files
from utils.constants import NUM_GEOM
# Projection
from pyproj import CRS, Transformer

crs_proj = CRS.from_epsg(27700)
crs_4326 = CRS("WGS84")
transformer = Transformer.from_crs(crs_4326, crs_proj)

def proj_and_reorder_bounds(bbox):
    S, W, N, E = bbox
    [S, N], [W, E] = transformer.transform([S, N], [W, E])
    return (W, E, S, N)

loaded_gdfs = {} # For OSMnx data
loaded_graphs = {} # To cache graphs
full_gdf = None
full_gdf_ignore_fields = None
accident_gdf = None
    

def load_gdf(place,
             bbox=None, 
             ignore_fields=fields_to_ignore, 
             clean=True,
             dist=50, # For accident gdf only
             agg='mean', # For cleaned gdf only
             verbose=False): #(W, S, E, N)
    """
    Load the geodataframe (gdf) corresponding to a local authority (SSx) or any place (OSMnx) with caching.
    """
    global full_gdf, full_gdf_ignore_fields
    
    if full_gdf is None or full_gdf_ignore_fields != ignore_fields:
        if verbose:
            print('Reading SSx OpenMapping dataset, this may take a while...')
        full_gdf = gpd.read_file(f'{dataset_root}/OpenMapping-gb-v1_gpkg/gpkg/ssx_openmapping_gb_v1.gpkg', ignore_fields=ignore_fields)
        full_gdf_ignore_fields = ignore_fields.copy()

    if 'accident_count' not in ignore_fields:
        if verbose:
            print('Constructing SSx-Accident dataset, this may take a while...')
            # Combine SSx dataset with accidents data
        full_gdf = construct_accident_dataset(ignore_fields=ignore_fields + ['accident_count'],
                                              year_from=2011, year_to=2020, maxdist=dist)
    
    if type(place) == list and len(place) == 4 and type(place[0]) == float:
        # Read bounding box
        ymin, ymax, xmin, xmax = proj_and_reorder_bounds(place)
        gdf = full_gdf.cx[xmin:xmax, ymin:ymax].copy()
    elif place in included_places:
        # Retrieve matching rows corresponding to the Local Authority
        if verbose:
            print(f'Loading {place} from SSx')
        gdf = full_gdf.query(f'lad11nm == "{place}"').copy()
    # elif place in 
    #     print(f'Loading {place} from SSx')
    #     gdf = full_gdf.query(f'lad11nm == "{place}"').copy()
    elif place == full_dataset_label:
        # Read full UK dataset without boundaries
        clean = False # Override this flag, takes very long
        gdf = full_gdf.copy()
    else:
        if place in loaded_gdfs:
            if verbose:
                return(f'Returning loaded OSMnx graph')
            return loaded_gdfs[place]
        if verbose:
            print(f'Loading {place} from OSMnx')
        # Load gdf from osmnx (to be used for testing only, graph will lack SSx target attr)
        # Actually uses the Nominatim API:
        # https://nominatim.org/release-docs/latest/api/Overview/
        g = ox.graph.graph_from_place(place, buffer_dist=osmnx_buffer)
        g = ox.projection.project_graph(g)
        gdf = ox.utils_graph.graph_to_gdfs(g, nodes=False)
        gdf = gdf.rename(columns={'length': 'metres'})
        loaded_gdfs[place] = gdf
        return gdf
    
    if clean:
        gdf = remove_false_nodes(gdf, agg=agg)
    if verbose:
        print(f'{gdf.size} geometries retrieved from {place}')
    return gdf


def construct_accident_dataset(gdf=None, ignore_fields=fields_to_ignore,
                               place='No Bounds', year_from=2011, year_to=2020, maxdist=10):
    """
    Combines a gdf loaded from SSx from accident counts from 2011 to 2020
    Uses spatial join to compute the road segment corresponding to an accident's location. 
    Better accuracy achieved when done before geometry cleaning.
    """
    if gdf is None:
        gdf = load_gdf(place, ignore_fields=ignore_fields, clean=(place != 'No Bounds'))
    
    # Load in accident csv in chunks: File is quite large
    iter_csv = pd.read_csv(f'{dataset_root}/dft-road-casualty-statistics-accidents.csv',
                              iterator=True, chunksize=1000)
    # Filter by year
    accident_df = pd.concat([chunk[(chunk['accident_year']  >= year_from) & (chunk['accident_year'] <= year_to)] for chunk in iter_csv])
    
    # Convert to GeoDataFrame
    accident_gdf = gpd.GeoDataFrame(
        accident_df, 
        geometry=gpd.points_from_xy(accident_df.location_easting_osgr,
                                    accident_df.location_northing_osgr),
        crs=gdf.crs
    )
    
    # Spatial join with OpenMapping Dataset, obtaining the closest road segment
    # Filters out accidents on non-included roads by limiting distance
    join_gdf = accident_gdf.sjoin_nearest(
        gdf.reset_index(level=0), 
        how='left', 
        max_distance=maxdist, distance_col='dist'
    )
    
    # Count the number of accidents assigned to each road segment
    count_series = join_gdf.groupby(['index']).size().reset_index()
    for i in range(len(count_series)):
        idx = count_series.iloc[i]['index']
        gdf.at[idx, 'accident_count'] = count_series.iloc[i][0]
    gdf['accident_count'] = gdf['accident_count'].fillna(0)
    return gdf


"""
Utilities to generate centroids for roads (for a dual graph dataset)
Adapted from https://github.com/zahrag/GAIN/blob/main/codes/roadnetwork_graphs.py
"""
def centroid_generation(g):
    pos = {}
    for e, d in g.nodes(data=True):
        mid = d['geometry'].centroid
        pos[e] = {'midpoint': np.array([mid.x, mid.y])}
    nx.set_node_attributes(g, pos)
    pass

def centroid_subtraction(g):
    for _, d in g.nodes(data=True):
        d['geom'] = d['geom'] - d['midpoint']
    pass

def generate_geometries(g, steps, attr_name='geom', verbose=True):
    if verbose:
        print(f'Generating fixed length {steps} geometry vectors...')
    geoms = nx.get_node_attributes(g, 'geometry')
    np_same_length_geoms = {}
    
    count_no = 0
    count_yes = 0
    for e, d in g.nodes(data=True):
        points = []
        if e not in geoms:  # Assumes conversion via nx.line_graph
            line = LineString([e[0], e[1]])
            d['geometry'] = line
            for step in np.linspace(0, 1, steps):
                point = line.interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_no += 1

        else:  # all other edges
            for step in np.linspace(0, 1, steps):
                point = geoms[e].interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_yes += 1
        np_same_length_geoms[e] = np.array([np.array((p[0], p[1])) for p in points])

    if verbose > 0:
        print('- Geometry inserted from intersection coordinates for', count_no, 'nodes.')
        print('- Standardized geometry created for', count_no + count_yes, 'nodes.')

    nx.set_node_attributes(g, np_same_length_geoms, attr_name)
    if verbose > 0:
        print('Done.')
    pass


"""
Graph and feature preprocessing
"""
def process_primal_graph(g, feature_fields=[], return_nx=False, agg='mean'):
    """
    Takes SSx metrics from adjacent edges and averages them to form the node attribute
    """
    for node, d in g.nodes(data=True):
        # get attributes from adjacent edges
        new_data = {field:[] for field in feature_fields}
        for _, _, edge_data in g.edges(node, data=True):
            for field in feature_fields:
                new_data[field].append(edge_data[field])
        
        if not return_nx:
            d.clear()
        # take the sum/mean of the collected attributes
        for field in feature_fields:
            d[field] = apply_agg(new_data[field], agg)
        
        # Encode node coordinate as feature
        d['x'], d['y'] = node
        d['lng'], d['lat'] = node
    
    for u, v, d in g.edges(data=True):
        if not return_nx:
            d.clear()
        # Encode edge information for later indexing
        d['u'] = u
        d['v'] = v
    return g

def process_dual_graph(g, feature_fields=[], target_field=None, geoms=True, return_nx=False):
    if geoms:
        # Get line centroid coordinate by taking the centroid of interpolated points
        generate_geometries(g, steps=NUM_GEOM)
        centroid_generation(g)
        centroid_subtraction(g)
    
    for node, d in g.nodes(data=True):
        # Break down the geometry features to single floats
        if geoms:
            new_data = {
                'metres': d['metres'],
                'mid_x': d['midpoint'][0],
                'mid_y': d['midpoint'][1]
            }
            for i, point in enumerate(d['geom']):
                new_data[f'geom{i}_x'] = point[0]
                new_data[f'geom{i}_y'] = point[1]
        else:
            new_data = {}
        for attr in feature_fields:
            if attr not in new_data:
                # get feature from node data
                new_data[attr] = d[attr]
                
        if target_field:
            new_data[target_field] = d[target_field]
                
        # Reset the data dictionary to only included desired features
        if not return_nx:
            d.clear()

        for k in new_data:
            d[k] = new_data[k]
    
    for u, v, d in g.edges(data=True):
        # Remove non numeric edge data
        if not return_nx:
            d.clear()

    return g, [key for key in new_data.keys() if key != target_field]

def add_latent_feats(G):
    if G.approach == 'primal':
        link_pred_metrics = torch.load(f'{dataset_root}/link_pred/clean_link_pred.pt')[1]
    elif G.approach == 'dual':
        link_pred_metrics = torch.load(f'{dataset_root}/link_pred/dual_link_pred.pt')[1]
    if place not in link_pred_metrics:
        raise ValueError
    
    enc = link_pred_metrics[G.place]['z']
    enc_dict = {}
    # Sanity check
    assert len(enc) == G.number_of_nodes()
    for i, node in enumerate(G.nodes()):
        z = enc[i]
        enc_dict[node] = {'z_{i}': z_i for idx, z_i in enumerate(z)}
    nx.set_node_attributes(G, enc_dict)
    
def clean_gdf(gdf, approach='primal', agg='sum'):
    G = gdf_to_nx(gdf, approach='primal', directed=True)
    for u, d in G.nodes(data=True):
        d['x'], d['y'] = u
    for _, _, d in G.edges(data=True):
        # Simplification cannot handle object attributes
        del d['geometry']
    G = ox.simplify_graph(G)
    
    # Aggregate edge attributes that have been converted to lists
    for _, _, data in G.edges(data=True):
        # Regenerate geometry
        for key in data:
            attr = data[key]
            if type(attr) == list:
                data[key] = apply_agg(attr, agg)
                    
    G = G.to_undirected()
    
    if approach == 'primal':
        return G
    line_G = nx.line_graph(G)
    gsf = True
    for u, data in line_G.nodes(data=True):
        data.update(G.edges[u])
        if gsf:
            print(data)
            gsf = False
    for u, v, data in line_G.edges(data=True):
        data.update(G.nodes[u[1]])
    return line_G

def load_graph(
    place,
    from_gdf=None,
    feature_fields=[],
    cat_fields=[],
    target_field=None,
    approach='primal',
    agg='sum',
    clean=True,
    clean_agg='sum',
    dist=50, # applies only to accident count feature
    geoms=True, # applies only to dual graph
    force_connected=True,
    reset=False,
    return_nx=False,
    verbose=False
):
    key = (str(place), str(feature_fields + cat_fields), target_field, approach,
           clean, force_connected, dist, return_nx)
    if verbose:
        print(f'Loading graph of {place} with key {key}...')
    if reset:
        pass
    elif key in loaded_graphs:
        if verbose:
            print('Loaded graph from cache.')
        if key[-1]: # NetworkX
            return loaded_graphs[key].copy()
        else: # PyG
            return loaded_graphs[key].clone()
    elif key in saved_data_files:
        if verbose:
            print('Loaded graph from saved files.')
        return torch.load(f'{dataset_root}/{saved_data_files[key]}')
    
    ignore_fields = [field for field in fields_to_ignore \
                        if field not in [*feature_fields, *cat_fields, target_field]]
    if from_gdf is None:
        gdf = load_gdf(place, verbose=verbose, ignore_fields=ignore_fields, clean=clean, agg=clean_agg, dist=dist)
    else:
        gdf = from_gdf

    if len(cat_fields) > 0:
        gdf, new_cols = convert_categorical_features_to_one_hot(gdf, cat_fields)
        feature_fields += new_cols
        
    if place == full_dataset_label and clean:
        G = clean_gdf(gdf, approach=approach, agg=agg)
    else:
        G = momepy.gdf_to_nx(gdf, approach=approach, multigraph=False)

    if force_connected:
        G = G.subgraph(max(nx.connected_components(G), key=len))
        
    if 'linkpred' in feature_fields:
        G, added_feats = add_latent_feats(G)
        feature_fields.extend(added_feats)
        
    if approach == 'primal':
        G = process_primal_graph(G, feature_fields, return_nx=return_nx, agg=agg)
        expected_node_attrs = set(feature_fields + ['x', 'y', 'lng', 'lat'])
        features = feature_fields + ['x', 'y']
    elif approach == 'dual':
        G, features = process_dual_graph(G, feature_fields,
                                         target_field=target_field,
                                         return_nx=return_nx,
                                         geoms=geoms)
        expected_node_attrs = features

    if verbose:
        print(f'Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
        edge_attrs = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
        # List node and edge attributes
        print(f'Node features: {features}')
        print(f'Node target: {target_field}')
        print(f'Edge attributes: {edge_attrs}')                            

    if return_nx:
        # Return before conversion to PyG
        loaded_graphs[key] = G
        return G.copy()

    # If no node attributes, node degree will be added later
    # Edge attribute (angle) are not added
    if len(features) > 0:
        g = from_networkx(G, group_node_attrs=features)
    else:
        g = from_networkx(G)

    # For subsequent reference
    g.place = place
    g.node_attrs = list(features)
    
    # Caching
    loaded_graphs[key] = g
    
    return g.clone()
