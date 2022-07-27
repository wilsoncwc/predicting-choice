from utils.load_geodata import load_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import momepy

def visualize_graph(place, approach='primal', feature=None, title=None, save_path=None, ax=None):
    feature_fields = [feature] if feature else []
    G = load_graph(place, approach=approach, feature_fields=feature_fields, return_nx=True)
    gdf = momepy.nx_to_gdf(G)[1] if approach == 'primal' else momepy.nx_to_gdf(G)
    # colors = [node[1][feature] for node in G.nodes(data=True)]
    if title is None:
        title = place
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,10))
    # nx.draw(G, {n:[n[0], n[1]] for n in list(G.nodes)}, node_size=10, node_color=colors, cmap='viridis')
    ax.set_title(title)
    
    if feature:
        gdf.plot(ax=ax, 
                 column=feature, 
                 legend=True, 
                 legend_kwds={
                     'label': feature,
                     'orientation': "horizontal"
                 })
    else:
        gdf.plot(ax=ax)
    plt.tight_layout()
    
def visualize_nx(graph, ax=None, remove_self_loops=True, **kwargs):
    if remove_self_loops:
        graph.remove_edges_from(nx.selfloop_edges(graph))
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,10))
    nx.draw_networkx(graph,
                     ax=ax,
                     with_labels=False,
                     pos={n:[n[0], n[1]] for n in list(graph.nodes)},
                     **kwargs)
    

def plot_loss_curve(losses, label, ax=None, color=None, show_std=True, annotate_last=True, **plot_kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('Validation Loss over training')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Validation Loss')
    
    epochs = np.arange(1, len(losses[0]) + 1)
    losses_mean = np.mean(losses, axis=0)
    losses_std = np.std(losses, axis=0)
    
    if color is None:
        color = 'g'
    # Plot learning curve and error margin 
    if show_std:
        ax.fill_between(
            epochs,
            losses_mean - losses_std,
            losses_mean + losses_std,
            alpha=0.3,
            color=color,
        )
    ax.plot(
        epochs, losses_mean, color=color, label=label, **plot_kwargs
    )
    
    if annotate_last:
        last_value = losses_mean[-1]
        ax.annotate('%0.3f' % last_value, xy=(1, last_value), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
        
    ax.legend(loc="best")

    return ax
    
def plot_multiple_roc_pr(place, rocs, prs):
    f, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot ROC curves
    ax = axs[0]
    for fpr, tpr, label, auc in rocs:
        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{label} ROC curve (AUROC = {auc:.3f})',
        )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f'Receiver Operating Characteristic curve for {place}')
    ax.legend(loc="lower right")
    
    # Plot PR
    # Obtain curve corresponding to the best AP
    ax = axs[1]
    for prec, rec, label, ap in prs:
        ax.plot(
            prec,
            rec,
            lw=2,
            label=f'{label} PR curve (AP = {ap:.3f})',
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f'Precision-Recall curve for {place}')
    ax.legend(loc="lower right")
    plt.show()
    

def plot_roc_pr(place, link_pred_metrics):
    transductive_data = link_pred_metrics[place]
    f, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot ROC
    # Obtain curve corresponding to the best AUC
    auc_list = transductive_data['train_auc']
    max_auc = max(auc_list)
    max_idx = auc_list.index(max_auc)
    fpr, tpr, _ = transductive_data['roc'][max_idx]
    
    ax = axs[0]
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f'ROC curve (AUROC = {max_auc:.3f})',
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f'Receiver Operating Characteristic curve for {place}')
    ax.legend(loc="lower right")
    
    # Plot PR
    # Obtain curve corresponding to the best AP
    ax = axs[1]
    prec, rec, _ = transductive_data['pr'][max_idx]
    max_ap = transductive_data['train_ap'][max_idx]
    ax.plot(
        prec,
        rec,
        lw=2,
        label=f'PR curve (AP = {max_ap:.3f})',
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f'Precision-Recall curve for {place}')
    ax.legend(loc="lower right")
    plt.show()
