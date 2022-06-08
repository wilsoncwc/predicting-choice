#!/usr/bin/env python
# coding: utf-8

# Set path to include project root so that modules can be directly imported
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import pandas as pd
import networkx as nx
from utils.utils import remove_item
from utils.load_geodata import load_graph, load_gdf
from utils.constants import *


def build_network_df():
    records = []
    for place in included_places:
        primal_G = load_graph(place, return_nx=True)
        gdf = load_gdf(place)
        
        graph_stats = {
            'place': place,
            'average_degree_connectivity': nx.average_degree_connectivity(primal_G),
            'average_degree': sum(val for (node, val) in primal_G.degree()) / primal_G.number_of_nodes(),
            'num_nodes': primal_G.number_of_nodes(),
            'num_edges': primal_G.number_of_edges(),
            'density': nx.density(primal_G),
            'transitivity': nx.transitivity(primal_G),
            'average_shortest_path_length': nx.average_shortest_path_length(primal_G),
            'diameter': nx.algorithms.distance_measures.diameter(primal_G)
        }
        ssx_measures = all_feature_fields
        average_ssx_metrics  = {
            measure: gdf[measure].mean() 
            for measure in ssx_measures
        }
        records.append({ **graph_stats, **average_ssx_metrics })
    df = pd.DataFrame.from_records(records)
    df.to_pickle(f'{dataset_root}/network_df.pt')

def add_graph_stats(net_df):
    for place in net_df['place']:
        G = load_graph(place, approach='primal', clean=True, return_nx=True, verbose=False)
        
        # Degree fractions
        deg_counts = { 1: 0., 2: 0., 3:0., 4:0.}
        num_nodes = G.number_of_nodes()
        for i in (d for _, d in G.degree()):
            if i in deg_counts:
                deg_counts[i] += 1
        for deg in deg_counts:
            frac = deg_counts[deg] / num_nodes
            net_df.loc[net_df['place'] == place, f'{deg}_frac'] = frac
        
        # Dendricity
        cycle_edges = [list(zip(nodes,(nodes[1:]+nodes[:1]))) for nodes in nx.cycle_basis(G)]
        edge_class = {
            'bridge': 0,
            'dead-end': 0,
            'cycle': 0
        }
        for (u, v) in G.edges():
            found = False
            for cycle in cycle_edges:
                if (u, v) in cycle or (v, u) in cycle:
                    found = True
            if found:
                # Cycle
                edge_class['cycle'] += 1
            else:
                if G.degree[u] == 1 or G.degree[v] == 1:
                    # Dead end
                    edge_class['dead-end'] += 1
                else:
                    # Bridge
                    edge_class['bridge'] += 1 
        num_edges = G.number_of_edges()

        for c in edge_class:
            frac = edge_class[c] / num_edges
            net_df.loc[net_df['place'] == place, c] = frac

def construct_ssx_dataset(places, **kwargs):
    dataset = [ load_graph(place, reset=True, **kwargs) for place in places ]
    return dataset


def main():
    all_feat_fields = [field for field in all_feature_fields]
    for agg in [
        'min',
        'max',
        'sum',
        'mean',
        'median',
        'std'
    ]:
        data = load_graph(remove_item(included_places, inductive_places),
                          feature_fields=all_feature_fields,
                          cat_fields=['meridian_class'],
                          force_connected=True, 
                          approach='dual',
                          clean=True,
                          clean_agg=agg,
                          target_field='accident_count',
                          verbose=True,
                          save_file=f'accident_transductive_cut_{agg}_pyg.pt')
        data = load_graph(inductive_places,
                  feature_fields=all_feature_fields,
                  cat_fields=['meridian_class'],
                  force_connected=True, 
                  approach='dual',
                  clean=True,
                  clean_agg=agg,
                  target_field='accident_count',
                  verbose=True,
                  save_file=f'accident_inductive_cut_{agg}_pyg.pt')

if __name__ == '__main__':
    main()

    
"""
net_df = pd.read_pickle(f'{dataset_root}/network_df.pt')
add_graph_stats(net_df)
net_df.to_pickle(f'{dataset_root}/network_df_w_stats.pt')


    dataset = construct_ssx_dataset(included_places, 
                                    feature_fields=all_feature_fields, 
                                    approach='primal', 
                                    clean=True, 
                                    agg='min', 
                                    verbose=True)
    for data in dataset:
        data.node_attrs = list(data.node_attrs)
    torch.save(dataset, f'{dataset_root}/ssx_dataset_clean_min.pt')
    
    all_feat_fields = [field for field in all_feature_fields]
    for agg in [
        'max',
        'min',
        'sum',
        'mean',
        'median',
        'std'
    ]:
        data = load_graph(remove_item(included_places, inductive_places),
                          feature_fields=all_feature_fields,
                          cat_fields=['meridian_class'],
                          force_connected=True, 
                          approach='dual',
                          clean=True,
                          clean_agg=agg,
                          target_field='meridian_class',
                          verbose=True,
                          save_file=f'accident_transductive_cut_{agg}_pyg.pt')
        data = load_graph(inductive_places,
                  feature_fields=all_feature_fields,
                  cat_fields=['meridian_class'],
                  force_connected=True, 
                  approach='dual',
                  clean=True,
                  clean_agg=agg,
                  target_field='meridian_class',
                  verbose=True,
                  save_file=f'accident_inductive_cut_{agg}_pyg.pt')
"""
