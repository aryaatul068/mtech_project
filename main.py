from pathlib import Path
import argparse
import yaml
import torch
import os

from model import FBNETGEN, GNNPredictor, SeqenceModel, BrainNetCNN, FBNETGEN_MultiView
from train import BasicTrain, BiLevelTrain, SeqTrain, GNNTrain, BrainCNNTrain
from datetime import datetime
from dataloader import init_dataloader


def showGraph(dataloaders):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Unpack the dataloaders
    train_loader, _, _ = dataloaders

    # Get one batch
    for fc_batch, pearson_batch, label_batch in train_loader:
        # Just visualize the first sample in the batch
        fc_sample = fc_batch[0].numpy()           # shape: (nodes, time)
        pearson_sample = pearson_batch[0].numpy() # shape: (nodes, nodes)
        label = label_batch[0].item()             # class label (0 or 1)
        break

    node_idx = 0

    plt.figure(figsize=(10, 4))
    plt.plot(fc_sample[node_idx], label=f"Node {node_idx}")
    plt.title(f"Time Series for Node {node_idx} (Label: {label})")
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Threshold to show only strong connections
    threshold = 0.5
    adj_matrix = pearson_sample

    # Create graph
    G = nx.Graph()

    # Add edges for values above threshold
    for i in range(adj_matrix.shape[0]):
        for j in range(i+1, adj_matrix.shape[1]):
            weight = adj_matrix[i, j]
            if abs(weight) > threshold:
                G.add_edge(i, j, weight=weight)

    # Plot the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges(data=True)
    weights = [abs(d['weight']) for (u, v, d) in edges]

    nx.draw(G, pos, with_labels=False, node_size=50,
            edge_color=weights, edge_cmap=plt.cm.Blues, width=2.0)
    plt.title(f"Functional Connectivity Graph (Label: {label})")
    plt.show()


def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

        dataloaders, node_size, node_feature_size, timeseries_size = \
            init_dataloader(config['data'])

        # here save dataloaders, node_size, node_features, timeseries_size

        # load same data

        # train_dataloader, test_dataloader, val_dataloader = dataloaders
        # for batch in train_dataloader:
        #     if isinstance(batch, (list, tuple)):
        #         for item in batch:
        #             print("test 1")
        #             print(type(item), item.shape if hasattr(item, 'shape') else item)
        #     else:
        #         print("test 2")
        #         print(type(batch), batch.shape if hasattr(batch, 'shape') else batch)
        #     break
        # print(train_data)

        # showGraph(dataloaders)

        # return

        config['train']["seq_len"] = timeseries_size
        config['train']["node_size"] = node_size

        if config['model']['type'] == 'seq':
            model = SeqenceModel(config['model'], node_size, timeseries_size)
            use_train = SeqTrain

        elif config['model']['type'] == 'gnn':
            model = GNNPredictor(node_feature_size, node_size)
            use_train = GNNTrain

        elif config['model']['type'] == 'fbnetgen':
            model = FBNETGEN(config['model'], node_size,
                             node_feature_size, timeseries_size)
            use_train = BasicTrain
        
        elif config['model']['type'] == 'fbnetgen_multiview':
            model = FBNETGEN_MultiView(node_input_dim=node_feature_size, roi_num=node_size)
            use_train = BasicTrain

        elif config['model']['type'] == 'brainnetcnn':

            model = BrainNetCNN(node_size)

            use_train = BrainCNNTrain


        if config['train']['method'] == 'bilevel' and \
                config['model']['type'] == 'fbnetgen':
            parameters = {
                'lr': config['train']['lr'],
                'weight_decay': config['train']['weight_decay'],
                'params': [
                    {'params': model.extract.parameters()},
                    {'params': model.emb2graph.parameters()}
                ]
            }

            optimizer1 = torch.optim.Adam(**parameters)

            optimizer2 = torch.optim.Adam(model.predictor.parameters(),
                                          lr=config['train']['lr'],
                                          weight_decay=config['train']['weight_decay'])
            opts = (optimizer1, optimizer2)
            use_train = BiLevelTrain

        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config['train']['lr'],
                weight_decay=config['train']['weight_decay'])
            opts = (optimizer,)

        loss_name = 'loss'
        if config['train']["group_loss"]:
            loss_name = f"{loss_name}_group_loss"
        if config['train']["sparsity_loss"]:
            loss_name = f"{loss_name}_sparsity_loss"

        now = datetime.now()

        date_time = now.strftime("%m-%d-%H-%M-%S")

        extractor_type = config['model']['extractor_type'] if 'extractor_type' in config['model'] else "none"
        embedding_size = config['model']['embedding_size'] if 'embedding_size' in config['model'] else "none"
        window_size = config['model']['window_size'] if 'window_size' in config['model'] else "none"

        save_folder_name = Path(config['train']['log_folder'])/Path(
            date_time +
            f"_{config['data']['dataset']}_{config['model']['type']}_{config['train']['method']}" 
            + f"_{extractor_type}_{loss_name}_{embedding_size}_{window_size}")

        train_process = use_train(
            config['train'], model, opts, dataloaders, save_folder_name)

        train_process.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/pnc.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=1, type=int)
    args = parser.parse_args()
    for i in range(args.repeat_time):
        main(args)
