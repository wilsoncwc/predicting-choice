import momepy
import fiona
import geopandas as gpd
import networkx as nx
import osmnx as ox

from torch_geometric.utils.convert import from_networkx
from itertools import chain
from utils.constants import dataset_root, osmnx_buffer, ignore_fields
from utils.constants import included_places, full_dataset_label

loaded_gdfs = {}
loaded_graphs={}

full_gdf = None

def load_gdf(place, ignore_fields=ignore_fields, reset_gdf=False, verbose=False): #(W, S, E, N)
    """
    Load the geodataframe (gdf) corresponding to a local authority (SSx) or any place (OSMnx) with caching.
    """
    global full_gdf
    if full_gdf is None or reset_gdf:
        full_gdf = gpd.read_file(f'{dataset_root}/OpenMapping-gb-v1_gpkg/gpkg/ssx_openmapping_gb_v1.gpkg',
                                 ignore_fields=ignore_fields)
    if place in included_places:
        # Retrieve matching rows corresponding to the Local Authority
        print(f'Loading {place} from SSx')
        gdf = full_gdf.query(f'lad11nm == "{place}"').copy()
    elif place == full_dataset_label:
        # Read full UK dataset without boundaries
        gdf = full_gdf.copy()
    else:
        print(f'Loading {place} from OSMnx')
        # Load gdf from osmnx (to be used for testing only, graph will lack SSx target attr)
        # Actually uses the Nominatim API:
        # https://nominatim.org/release-docs/latest/api/Overview/
        g = ox.graph.graph_from_place(place, buffer_dist=osmnx_buffer)
        g = ox.projection.project_graph(g)
        gdf = ox.utils_graph.graph_to_gdfs(g, nodes=False)
        gdf = gdf.rename(columns={'length': 'metres'})
        return gdf

    if verbose:
        print(f'{gdf.size} geometries retrieved from {place}')

    loaded_gdfs[place] = gdf
    return gdf

def process_graph(g, feature_fields=[]):
    """
    Takes SSx metrics from adjacent edges and averages them to form the node attribute
    """
    for node, d in g.nodes(data=True):
        # get attributes from adjacent edges
        new_data = {field:[] for field in feature_fields}
        for _, _, edge_data in g.edges(node, data=True):
            for field in feature_fields:
                new_data[field].append(edge_data[field])
        
        d.clear()
        # take the average of the collected attributes
        for field in feature_fields:
            d[field] = sum(new_data[field]) / len(new_data[field])
        
        # Encode node coordinate as feature
        d['x'], d['y'] = node
        d['lng'], d['lat'] = node
    
    for u, v, d in g.edges(data=True):
        d.clear()
        # Encode edge information for later indexing
        d['u'] = u
        d['v'] = v
    return g

def load_graph(place, feature_fields=[], force_connected=True, reload=True, verbose=False):
    if verbose:
        print(f'Loading graph of {place}...')
    key = (place)
    if key in loaded_graphs and reload:
        g = loaded_graphs[key]
        if verbose:
            print('Loaded existing graph.')
        print(g)
    else:
        gdf = load_gdf(place, verbose=verbose)
        G = momepy.gdf_to_nx(gdf, approach='primal', multigraph=False)
        if force_connected:
            G = G.subgraph(max(nx.connected_components(G), key=len))
        G = process_graph(G, feature_fields)

        if verbose:
            print(f'Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
        node_attrs = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))
        assert node_attrs == set(feature_fields + ['x', 'y', 'lng', 'lat'])
        node_attrs = feature_fields + ['x', 'y']
        edge_attrs = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))

        if verbose:
            # List node attributes
            print(f'Node attributes: {node_attrs}')
            print(f'Edge attributes: {edge_attrs}')                            

        # If no node attributes, node degree will be added later
        # Edge attribute (angle) are not added
        if len(node_attrs) > 0:
            g = from_networkx(G, group_node_attrs=node_attrs)
        else:
            g = from_networkx(G)
        loaded_graphs[key] = g
        g.place = place
    return g
