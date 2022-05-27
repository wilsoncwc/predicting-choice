import geopandas as gpd
import pygeos
import numpy as np
import collections

def remove_false_nodes(gdf, agg='mean'):
    """
    Modified version of momepy's remove_false_nodes utility function
    that also generates attributes for the merged LineString geometry
    using the mean or sum over the attributes of the components
    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries, array of pygeos geometries
        (Multi)LineString data of street network
    Returns
    -------
    gdf : GeoDataFrame, GeoSeries
    """
    if isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # explode to avoid MultiLineStrings
        # reset index due to the bug in GeoPandas explode
        df = gdf.reset_index(drop=True).explode(ignore_index=True)

        # get underlying pygeos geometry
        geom = df.geometry.values.data
    else:
        geom = gdf
        df = gpd.GeoSeries(gdf)

    # extract array of coordinates and number per geometry
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = pygeos.STRtree(geom)
    inp, res = tree.query_bulk(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    merge = res[np.isin(inp, unique[counts == 2])]

    if len(merge) > 0:
        # filter duplications and create a dictionary with indication of components to
        # be merged together
        dups = [item for item, count in collections.Counter(merge).items() if count > 1]
        split = np.split(merge, len(merge) / 2)
        components = {}
        for i, a in enumerate(split):
            if a[0] in dups or a[1] in dups:
                if a[0] in components.keys():
                    i = components[a[0]]
                elif a[1] in components.keys():
                    i = components[a[1]]
            components[a[0]] = i
            components[a[1]] = i

        # iterate through components and create new geometries
        new = []
        data = []
        for c in set(components.values()):
            keys = []
            for item in components.items():
                if item[1] == c:
                    keys.append(item[0])
                    
            # MODIFIED
            # compute mean/sum/max of component attributes
            # for non-numeric attributes, take the mode
            comp_df = gdf.iloc[keys]
            if agg == 'mean':
                agg_data = comp_df.mean(numeric_only=True)
            elif agg == 'sum':
                agg_data = comp_df.sum(numeric_only=True)
            else: # assume max
                agg_data = comp_df.max(numeric_only=True)
            non_numeric = comp_df.select_dtypes(include=['object'])
            for col in non_numeric.columns:
                agg_data[col] = gdf[col].mode()[0]

            data.append(agg_data)
            new.append(pygeos.line_merge(pygeos.union_all(geom[keys])))

        # remove incorrect geometries and append fixed versions
        df = df.drop(merge)
        final = gpd.GeoSeries(new).explode(ignore_index=True)
        data_df = gpd.GeoDataFrame(data)
        if isinstance(gdf, gpd.GeoDataFrame):
            return df.append(
                gpd.GeoDataFrame({df.geometry.name: final, **data_df.to_dict()},
                                 geometry=df.geometry.name),
                ignore_index=True,
            )
        return df.append(final, ignore_index=True)