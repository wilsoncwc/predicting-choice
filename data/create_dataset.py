import torch
import pandas as pd
import geopandas as gpd

from utils.load_geodata import load_graph, load_gdf
from utils.constants import project_root, dataset_root

def construct_ssx_dataset():
    loaded_graphs ={}
    dataset = [ load_graph(place, all_feature_fields, verbose=True) \
                for place in included_places ]
    torch.save(dataset, 'datasets/ssx_dataset.pt')
    return dataset
    