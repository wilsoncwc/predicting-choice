#!/usr/bin/env python
# coding: utf-8

# Set path to include project root so that modules can be directly imported
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.constants import dataset_root
from utils.remove_false_nodes import remove_false_nodes
from utils.load_geodata import construct_accident_dataset

def main():
    accident_gdf = construct_accident_dataset(place=place)
    accident_gdf = remove_false_nodes(accident_gdf, agg='sum')
    accident_gdf.to_file(f'{dataset_root}/clean_accident_sample.gpkg', driver='GPKG', layer='road')

if __name__ == '__main__':
    main()