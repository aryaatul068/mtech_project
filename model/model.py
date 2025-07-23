from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU

import math

from torch.nn import Linear # Explicitly import Linear

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- PyTorch Geometric Imports (Crucial for preventing NameError) ---
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
# ==============================================================================
# 1. SELF-CONTAINED CONTINUOUS WAVELET TRANSFORM (CWT) MODULE
# This replaces the need for an external library.
# ==============================================================================
class CWT(nn.Module):
    def __init__(self, num_freqs, window_size, trainable=False):
        super().__init__()
        self.num_freqs = num_freqs
        self.window_size = window_size
        
        # Create the wavelet filters (Morlet wavelets)
        # These filters are applied via convolution to perform the transform
        t = torch.linspace(-1, 1, window_size)
        w = 6.0 # Omega0, central frequency of the mother wavelet
        freqs = torch.linspace(1, window_size / 2, num_freqs)
        
        self.wavelets_real = torch.zeros(num_freqs, 1, window_size)
        self.wavelets_imag = torch.zeros(num_freqs, 1, window_size)

        for i, scale in enumerate(freqs):
            # Normalization factor
            norm = (scale**-0.5) * (np.pi**-0.25)
            # Scaled and shifted time
            ts = t / scale
            # Morlet wavelet formula
            psi = norm * torch.exp(1j * w * ts) * torch.exp(-0.5 * (ts**2))
            
            self.wavelets_real[i, 0] = psi.real
            self.wavelets_imag[i, 0] = psi.imag

        if trainable:
            self.wavelets_real = nn.Parameter(self.wavelets_real)
            self.wavelets_imag = nn.Parameter(self.wavelets_imag)

    def forward(self, x):
        # Input x shape: [batch_size, 1, window_size]
        # Move wavelets to the correct device
        self.wavelets_real = self.wavelets_real.to(x.device)
        self.wavelets_imag = self.wavelets_imag.to(x.device)
        
        # Apply convolution with real and imaginary parts of wavelets
        # Padding is set to 'same' to keep the output length equal to the input length
        output_real = F.conv1d(x, self.wavelets_real, padding='same')
        output_imag = F.conv1d(x, self.wavelets_imag, padding='same')
        
        # Compute magnitude to get the scalogram
        # Shape: [batch_size, num_freqs, window_size]
        scalogram = torch.sqrt(output_real**2 + output_imag**2)
        
        return scalogram

# ==============================================================================
# 2. FEATURE EXTRACTOR (extract): Wavelet + 2D CNN
# Now uses our self-contained CWT module.
# ==============================================================================
class WaveletCnnExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # A simple 2D CNN to process the 2D scalogram
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, scalograms):
        # Input scalograms shape: [batch_size (num_rois), 1, num_freqs, window_size]
        x = self.conv_block(scalograms)
        x = self.pool(x) # [batch_size, 32, 1, 1]
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x) # [batch_size, embedding_dim]
        return x

# ==============================================================================
# 3. GRAPH GENERATOR (emb2graph): Sparse and Learnable
# ==============================================================================
# In model/model.py

class SparseAttentionGraphGenerator(nn.Module):
    def __init__(self, input_dim, topk=20):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.topk = topk
        # --- ADD THIS LINE ---
        self.leaky_relu = nn.LeakyReLU(0.2) # Add a LeakyReLU layer
    def forward(self, x):
        # --- ADD THESE DEBUGGING PRINTS ---
       # print(f"\n--- Inside SparseAttentionGraphGenerator ---")
       # print(f"Input x stats:          min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        Q = self.query_proj(x)
        K = self.key_proj(x)
       # print(f"Projected Q stats:      min={Q.min():.4f}, max={Q.max():.4f}, mean={Q.mean():.4f}")

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        # Apply LeakyReLU to introduce non-linearity and positive values
        attn = self.leaky_relu(attn)
       # print(f"Raw attention scores stats: min={attn.min():.4f}, max={attn.max():.4f}, mean={attn.mean():.4f}")

        # Top-k sparsification
        topk_vals, topk_idx = torch.topk(attn, self.topk, dim=-1)
        mask = torch.zeros_like(attn)
        mask.scatter_(-1, topk_idx, 1.0)
        
        sparse_attn_before_softmax = attn * mask
        sparse_attn_before_softmax[sparse_attn_before_softmax == 0] = -1e9
       # print(f"Attn after masking stats: min={sparse_attn_before_softmax.min():.4f}, max={sparse_attn_before_softmax.max():.4f}, mean={sparse_attn_before_softmax.mean():.4f}")
        
        final_adj_matrix = F.softmax(sparse_attn_before_softmax, dim=-1)
        #print(f"Final Adj Matrix stats:   min={final_adj_matrix.min():.4f}, max={final_adj_matrix.max():.4f}, mean={final_adj_matrix.mean():.4f}")
        #print(f"--- Exiting SparseAttentionGraphGenerator ---\n")
        # --------------------------------

        return final_adj_matrix

# ==============================================================================
# 4. PREDICTOR: Temporal GNN (GCN-GRU)
# ==============================================================================
class TemporalGNNPredictor(nn.Module):
    def __init__(self, gnn_input_dim, gnn_hidden_dim, gru_hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(gnn_input_dim, gnn_hidden_dim)
        self.gcn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.gru = nn.GRU(gnn_hidden_dim, gru_hidden_dim, batch_first=True)
        self.classifier = nn.Linear(gru_hidden_dim, num_classes)

    def forward(self, graph_sequence):
        graph_embeddings = []
        for snapshot in graph_sequence:
            x, edge_index, edge_attr = snapshot.x, snapshot.edge_index, snapshot.edge_attr
            x = F.relu(self.gcn1(x, edge_index, edge_attr))
            x = F.relu(self.gcn2(x, edge_index, edge_attr))
            graph_embedding = torch.mean(x, dim=0)
            graph_embeddings.append(graph_embedding)

        sequence_tensor = torch.stack(graph_embeddings).unsqueeze(0)
        gru_out, _ = self.gru(sequence_tensor)
        
        # Shape of last_hidden_state is [1, gru_hidden_dim]
        last_hidden_state = gru_out[:, -1, :]
        
        # --- CORRECTED LINE ---
        # Do NOT squeeze the batch dimension. Pass the [1, dim] tensor directly.
        output = self.classifier(last_hidden_state)
        # The output shape will now be [1, 2], which is correct.
        
        return output
    
class AdvancedDynamicFBNETGEN(nn.Module):   #AdvancedDynamicFBNETGEN
    def __init__(self, roi_num, time_window_size, feature_embedding_dim, gnn_hidden_dim, gru_hidden_dim, num_classes, topk, num_wavelet_freqs):
        super().__init__()
        self.cwt = CWT(num_freqs=num_wavelet_freqs, window_size=time_window_size)
        self.feature_extractor = WaveletCnnExtractor(embedding_dim=feature_embedding_dim)
        self.graph_generator = SparseAttentionGraphGenerator(input_dim=feature_embedding_dim, topk=topk)
        self.predictor = TemporalGNNPredictor(
            gnn_input_dim=feature_embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            num_classes=num_classes
        )

    def forward(self, time_windows_sequence):
        # This forward pass now processes a SINGLE subject's sequence of windows
        graph_sequence = []
        generated_adj_matrices = []

        for window_data in time_windows_sequence:
            # 1. Apply Wavelet Transform
            window_data_reshaped = window_data.unsqueeze(1)
            scalograms = self.cwt(window_data_reshaped)
            
            # 2. Extract features
            scalograms_for_cnn = scalograms.unsqueeze(1)
            node_embeddings = self.feature_extractor(scalograms_for_cnn)
            
            # 3. Generate graph for this window
            adj_matrix = self.graph_generator(node_embeddings)
            generated_adj_matrices.append(adj_matrix)

            # 4. Create PyG Data object
            edge_index, edge_attr = dense_to_sparse(adj_matrix)
            snapshot_data = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)
            graph_sequence.append(snapshot_data)
            
        # 5. Pass the sequence of graphs to the temporal predictor
        final_prediction = self.predictor(graph_sequence)
        
        # --- MODIFICATION ---
        # Stack the generated matrices and calculate variance for logging/loss
        all_adj_matrices = torch.stack(generated_adj_matrices) # [seq_len, roi_num, roi_num]
        edge_variance = torch.mean(torch.var(all_adj_matrices, dim=0)) # Average variance across sequence
        
        # We return the average matrix for saving, but you could save all of them
        avg_learnable_matrix = torch.mean(all_adj_matrices, dim=0)

        return final_prediction, avg_learnable_matrix.unsqueeze(0), edge_variance
    
# --- PyTorch Geometric Imports ---
# Ensure you have torch_geometric installed: pip install torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

class AdvancedGNNPredictor(nn.Module):
    def __init__(self, node_input_dim, dropout_rate=0.5):
        """
        Initializes the Advanced GNN Predictor.

        Args:
            node_input_dim (int): The dimensionality of the input node features.
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.5.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        # GAT Layers
        self.gat1 = GATConv(node_input_dim, 64, heads=4, dropout=dropout_rate)
        self.gat2 = GATConv(64 * 4, 32, heads=4, dropout=dropout_rate)
        self.gat3 = GATConv(32 * 4, 64, heads=1, concat=False, dropout=dropout_rate)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64 * 4)
        self.bn2 = nn.BatchNorm1d(32 * 4)
        self.bn3 = nn.BatchNorm1d(64)

        # Fully Connected Network for classification
        self.fcn = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(32, 2)
        )

    def forward(self, data):
        """
        Forward pass of the AdvancedGNNPredictor.

        Args:
            data: A PyTorch Geometric Batch object.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        x_pooled = global_mean_pool(x, batch)

        return self.fcn(x_pooled)
# --- Helper GNN Layer (Example: Simple GAT-like layer) ---
# For a full implementation, consider PyTorch Geometric's GATConv
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (B, N, in_features), adj: (B, N, N)
        # N = number of nodes (roi_num)
        # B = batch size

        # Linear transformation
        Wh = torch.einsum('bni,io->bno', h, self.W) # (B, N, out_features)

        # Compute attention scores
        # a_input: (B, N, N, 2*out_features)
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, Wh.shape[1], 1) # (B, N, N, out_features) for node_i
        Wh2 = Wh.unsqueeze(1).repeat(1, Wh.shape[1], 1, 1) # (B, N, N, out_features) for node_j
        a_input = torch.cat([Wh1, Wh2], dim=-1) # (B, N, N, 2*out_features)

        e = self.leakyrelu(torch.einsum('bnij,jk->bnik', a_input, self.a).squeeze(-1)) # (B, N, N)

        # Apply mask
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # Only apply attention to existing edges
        attention = F.softmax(attention, dim=2) # Softmax over neighbors
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate messages
        h_prime = torch.einsum('bni,bij->bnj', Wh, attention).permute(0,2,1) # This einsum looks incorrect for typical GAT.
                                                                           # Should be 'bij,bni->bnj'
        # Corrected aggregation:
        h_prime = torch.einsum('bjk,bji->bki', attention, Wh) # (B, N, out_features)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

# --- Simplified SAGPool Layer (Conceptual) ---
# For a full implementation, use PyTorch Geometric's SAGPooling
class SAGPooling(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(SAGPooling, self).__init__()
        self.ratio = ratio
        self.score_mlp = nn.Linear(in_channels, 1) # MLP to learn node importance scores

    def forward(self, x, adj):
        # x: (B, N, F), adj: (B, N, N)
        scores = self.score_mlp(x).squeeze(-1) # (B, N)
        
        # Select top-k nodes based on scores
        num_nodes = x.shape[1]
        num_selected_nodes = int(num_nodes * self.ratio)
        
        # Get top-k scores and indices (for each batch item)
        # Need to handle batch-wise selection, simplified for conceptual clarity
        # For actual implementation, use torch.topk
        # Here's a conceptual simplified idea:
        _, perm = scores.sort(dim=-1, descending=True)
        perm = perm[:, :num_selected_nodes] # (B, num_selected_nodes)

        # Select features of top-k nodes
        # Need to gather for each batch item separately
        # Simplified:
        # selected_x = x[:, perm] # This won't work directly due to perm's shape
        
        # A more correct conceptual approach for batching:
        selected_x_list = []
        selected_adj_list = []
        for i in range(x.shape[0]): # Iterate over batch
            batch_perm = perm[i]
            selected_x_list.append(x[i, batch_perm])
            # Construct new adjacency matrix for selected nodes
            new_adj = adj[i, batch_perm, :][:, batch_perm]
            selected_adj_list.append(new_adj)

        selected_x = torch.stack(selected_x_list) # (B, num_selected_nodes, F)
        selected_adj = torch.stack(selected_adj_list) # (B, num_selected_nodes, num_selected_nodes)
        
        return selected_x, selected_adj, perm # perm for unpooling if needed

# --- Improved GNNPredictor ---
class ImprovedGNNPredictor(nn.Module):
    def __init__(self, node_input_dim, roi_num=200, gnn_hidden_dim=64, num_gnn_layers=3, dropout=0.5):
        super().__init__()
        self.roi_num = roi_num
        self.gnn_hidden_dim = gnn_hidden_dim # New parameter for hidden dimension
        self.num_gnn_layers = num_gnn_layers

        # Initial Linear transformation of node features
        self.initial_node_transform = nn.Sequential(
            nn.Linear(node_input_dim, gnn_hidden_dim), # (8 -> 64)
            nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm1d(gnn_hidden_dim) # BN after linear, before (or after) activation
        )

        # Stack GNN layers (e.g., GAT layers)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            # Input to GAT will always be gnn_hidden_dim, output too if concat=True or if not final layer
            # For simplicity, using a basic GAT-like structure here
            self.gnn_layers.append(nn.Sequential(
                # Use a custom GATLayer or nn.Linear if sticking to einsum-based GCN
                # If using GATLayer, in_features and out_features will be gnn_hidden_dim
                Linear(gnn_hidden_dim, gnn_hidden_dim), # Placeholder for GNN aggregation and transformation
                nn.LeakyReLU(negative_slope=0.2),
                torch.nn.BatchNorm1d(gnn_hidden_dim) # BN after linear, before (or after) activation
            ))
            # Add Residual Connection (optional but recommended for deeper nets)
            # You would typically add a bypass path around the GNN layer.

        # Optional: Graph Pooling Layer
        # self.pooling = SAGPooling(gnn_hidden_dim, ratio=0.5) # Example, requires SAGPooling implementation

        # Final Classification Head (FCN)
        # Input to FCN depends on whether pooling is used
        # If no pooling: gnn_hidden_dim * roi_num (e.g., 64 * 200 = 12800)
        # If pooling: gnn_hidden_dim * (roi_num * pooling_ratio) or a fixed size after global pooling
        
        # For now, let's assume no pooling for the initial FCN input calculation,
        # but the conceptual structure allows for it.
        fcn_input_dim = gnn_hidden_dim * roi_num 
        self.fcn = nn.Sequential(
            nn.Linear(fcn_input_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

    def forward(self, m, node_feature):
        bz = m.shape[0]

        # Squeeze graph adjacency matrix if it has a singleton last dim
        if m.dim() == 4 and m.shape[-1] == 1:
            m = m.squeeze(-1) # Shape (B, K, K)

        # Initial node feature transformation
        # Reshape for BatchNorm1d
        x = node_feature.reshape((bz * self.roi_num, -1)) # (B*K, node_input_dim)
        x = self.initial_node_transform(x) # (B*K, gnn_hidden_dim)
        x = x.reshape((bz, self.roi_num, -1)) # (B, K, gnn_hidden_dim)


        # Stacked GNN layers (e.g., GAT layers)
        for i, gnn_layer_module in enumerate(self.gnn_layers):
            # Aggregation (similar to your original einsum, but ideally from a GAT/GIN layer)
            # If using custom GATLayer: x = gnn_layer_module(x, m)
            # If sticking to basic GCN-like:
            aggregated_x = torch.einsum('ijk,ijp->ijp', m, x) # (B, K, gnn_hidden_dim)
            
            # Transformation and Normalization within the layer module
            x = gnn_layer_module(aggregated_x) # (B, K, gnn_hidden_dim)

            # Residual Connection (conceptual)
            # if i > 0 and self.add_residual_conn:
            #     x = x + previous_layer_output # Ensure dimension matching

        # Optional: Graph Pooling
        # if hasattr(self, 'pooling'):
        #     x, m, _ = self.pooling(x, m) # x: (B, new_N, F), m: (B, new_N, new_N)
        #     self.roi_num = x.shape[1] # Update roi_num for flattened layer

        # Global pooling (e.g., mean pooling over nodes if pooling is not used or before FCN)
        # If no pooling layers, and we want a fixed-size graph embedding before flattening:
        # graph_embedding = x.mean(dim=1) # (B, gnn_hidden_dim)
        # Then flatten: graph_embedding.view(bz, -1) if needed, or use a smaller FCN input

        # Flatten for FCN (assuming no pooling for this FCN input setup)
        x = x.view(bz, -1) # (B, gnn_hidden_dim * current_roi_num)

        return self.fcn(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class AdvancedGNNPredictorV2(nn.Module):
    def __init__(self, node_input_dim, hidden_dim=64, num_heads=4, dropout_rate=0.5):
        """
        Version 2 of the AdvancedGNNPredictor with key improvements.

        Args:
            node_input_dim (int): Dimensionality of input node features.
            hidden_dim (int): The base hidden dimension for GAT layers.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        # Layer 1
        self.conv1 = GATConv(node_input_dim, hidden_dim, heads=num_heads, dropout=dropout_rate, add_self_loops=False)
        self.res1 = nn.Linear(node_input_dim, hidden_dim * num_heads) # For residual connection
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Layer 2
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout_rate, add_self_loops=False)
        self.res2 = nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads) # For residual connection
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Final Classifier
        self.fcn = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(128, 2)
        )

    def forward(self, data):
        """
        Forward pass. Now uses edge_attr for attention weighting.

        Args:
            data: PyTorch Geometric Batch object with x, edge_index, and edge_attr.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Layer 1 with Residual Connection
        x_res = self.res1(x) # Project original features
        x = self.conv1(x, edge_index, edge_attr=edge_attr) + x_res # Add residual
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Layer 2 with Residual Connection
        x_res = self.res2(x) # Project features from previous layer
        x = self.conv2(x, edge_index, edge_attr=edge_attr) + x_res # Add residual
        x = self.bn2(x)
        x = F.relu(x)

        # Global Pooling and Final Classification
        x_pooled = global_mean_pool(x, batch)
        return self.fcn(x_pooled)
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        b, k, d = raw.shape

        x = raw.view((b*k, -1, self.kernel_size))

        x, h = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=360, time_series=512):
        super().__init__()

        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series, pool_size=4, )
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'], dropout=model_config['dropout'])

        self.linear = nn.Sequential(
            Linear(model_config['embedding_size']*roi_num, 256),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(256, 32),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(32, 2)
        )

    def forward(self, x):
        x = self.extract(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x
class Embed2GraphByLowRankAttention(nn.Module):
    def __init__(self, input_dim, proj_dim=32):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)

    def forward(self, x):
        x_proj = self.proj(x)  # [B, N, proj_dim]
        A = torch.matmul(x_proj, x_proj.transpose(1, 2))  # [B, N, N]
        A = F.softmax(A / (x_proj.shape[-1] ** 0.5), dim=-1)
        return A.unsqueeze(-1)

class Embed2GraphByThresholdCorr(nn.Module):
    def __init__(self, threshold=0.3):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        B, N, D = x.size()
        x = x - x.mean(dim=2, keepdim=True)
        corr = torch.matmul(x, x.transpose(1, 2)) / (x.shape[-1] - 1)
        std = x.std(dim=2, keepdim=True)
        corr = corr / (std @ std.transpose(1, 2) + 1e-8)
        corr = torch.where(torch.abs(corr) > self.threshold, corr, torch.zeros_like(corr))
        return corr.unsqueeze(-1)

class Embed2GraphByLearnableAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.a = nn.Parameter(torch.randn(input_dim * 2, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        B, N, D = x.size()
        x_i = x.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, D]
        x_j = x.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, D]
        x_ij = torch.cat([x_i, x_j], dim=-1)     # [B, N, N, 2D]
        e = self.leaky_relu(torch.matmul(x_ij, self.a).squeeze(-1))  # [B, N, N]
        attention = F.softmax(e, dim=-1)
        return attention.unsqueeze(-1)

class Embed2GraphByCosine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)  # [B, N, D]
        A = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, N, N]
        return A.unsqueeze(-1)

class Embed2GraphByGaussian(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.size()
        x1 = x.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, D]
        x2 = x.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, D]
        dist_sq = ((x1 - x2)**2).sum(-1)
        A = torch.exp(-dist_sq / (2 * self.sigma**2))  # [B, N, N]
        return A.unsqueeze(-1)
class Embed2GraphEnsemble(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        self.att_fusion = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, D = x.shape
        device = x.device

        # rel_rec and rel_send dynamically
        off_diag = torch.ones([N, N], device=device) - torch.eye(N, device=device)
        rel_rec = F.one_hot(torch.where(off_diag)[0], num_classes=N).float()
        rel_send = F.one_hot(torch.where(off_diag)[1], num_classes=N).float()

        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        x_cat = torch.cat([senders, receivers], dim=2)

        x_lin = F.relu(self.fc_out(x_cat))
        x_lin = F.relu(self.fc_cat(x_lin))

        # fill linear into matrix
        edge_idx = torch.where(off_diag > 0)
        m_lin = torch.zeros((B, N, N, 1), device=device)
        for b in range(B):
            m_lin[b, edge_idx[0], edge_idx[1], 0] = x_lin[b, :, 0]

        # product path
        m_prod = torch.einsum('bij,bkj->bik', x, x).unsqueeze(-1)

        # dynamic attention fusion
        fused_input = torch.cat([m_lin, m_prod], dim=-1)
        w = self.att_fusion(fused_input)
        m = w * m_lin + (1 - w) * m_prod

        # optional residual and normalization
        m = m + torch.mean(m, dim=-2, keepdim=True)
        m = F.layer_norm(m, m.shape[1:])

        return m


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)

        m = torch.unsqueeze(m, -1)

        return m

class Embed2GraphByAttention(nn.Module):
    def __init__(self, input_dim, roi_num=264):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: [B, N, D]
        Q = self.query_proj(x)
        K = self.key_proj(x)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        attention = F.softmax(attention, dim=-1)
        return attention.unsqueeze(-1)


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):

        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(
            x, (batch_sz, region_num, region_num, -1))
        return m



class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )


    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        x = x.view(bz,-1)

        return self.fcn(x)


class FBNETGEN(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        self.graph_generation = model_config['graph_generation']
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'])
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "attention":
            self.emb2graph = SparseAttentionGraph(
                model_config['embedding_size']) 
        elif self.graph_generation == "clique":
            self.emb2graph = Embed2GraphByClique(model_config['embedding_size'])

        elif self.graph_generation == "clique_and_fusion":
            self.emb2graph = Embed2GraphByFusion(model_config['embedding_size'])
                
        
        elif self.graph_generation == "ensemble":
            self.emb2graph = Embed2GraphEnsemble(model_config['embedding_size'])

        elif self.graph_generation == "gaussian":
            self.emb2graph = Embed2GraphByGaussian(model_config['embedding_size'])

        elif self.graph_generation == "cosine":
            self.emb2graph = Embed2GraphByCosine()

        elif self.graph_generation == "learnable_attention":
            self.emb2graph = Embed2GraphByLearnableAttention(model_config['embedding_size'])

        elif self. graph_generation == "threshold":
            self.emb2graph = Embed2GraphByThresholdCorr()
        
        elif self.graph_generation == "low_rank_attention":
            self.emb2graph = Embed2GraphByLowRankAttention(model_config['embedding_size'])

            
        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)

    def forward(self, t, nodes):
        x = self.extract(t)
        x = F.softmax(x, dim=-1)
        m = self.emb2graph(x)

        m = m[:, :, :, 0]

        bz, _, _ = m.shape  

        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))

        return self.predictor(m, nodes), m, edge_variance
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embed2GraphByClique(nn.Module):
    def __init__(self, input_dim, roi_num=360):
        super().__init__()
        self.fc_sim = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        self.roi_num = roi_num

    def forward(self, x):
        # x shape: [B, N, D]
        B, N, D = x.shape

        # Create all pairs of node embeddings (sender & receiver)
        x_i = x.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, D]
        x_j = x.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, D]

        # Concatenate pairs: [B, N, N, 2D]
        edge_input = torch.cat([x_i, x_j], dim=-1)

        # Compute similarity for each edge
        edge_weights = self.fc_sim(edge_input)  # [B, N, N, 1]

        # Optional: mask diagonal (self-loops)
        eye = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        edge_weights = edge_weights * (1 - eye)

        # Normalize over each row (softmax for stability)
        edge_weights = F.softmax(edge_weights.squeeze(-1), dim=-1).unsqueeze(-1)

        return edge_weights  # [B, N, N, 1]


class Embed2GraphByClique_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))  # learnable importance

    def forward(self, x):
        B, N, _ = x.shape
        A = torch.ones(B, N, N, device=x.device)
        A = A - torch.eye(N, device=x.device).unsqueeze(0)
        A = A / (N - 1)
        return (self.alpha * A).unsqueeze(-1)

class Embed2GraphByFusion(nn.Module):
    def __init__(self, input_dim, alpha=0.5):
        super().__init__()
        self.attn_graph = Embed2GraphByAttention(input_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha))  # learnable fusion weight

    def forward(self, x):
        A_attn = self.attn_graph(x)  # [B, N, N, 1]

        # Construct clique
        B, N, _ = x.shape
        A_clique = torch.ones(B, N, N, device=x.device)
        A_clique = A_clique - torch.eye(N, device=x.device).unsqueeze(0)
        A_clique = A_clique / (N - 1)
        A_clique = A_clique.unsqueeze(-1)

        # Combine
        A = self.alpha * A_attn + (1 - self.alpha) * A_clique
        return A
    
class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, roi_num):
        super().__init__()
        self.in_planes = 1
        self.d = roi_num

        self.e2econv1 = E2EBlock(1, 32, roi_num, bias=True)
        self.e2econv2 = E2EBlock(32, 64, roi_num, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out


class FCNet(nn.Module):

    def __init__(self, node_size, seq_len, kernel_size=3):
        super().__init__()

        self.ind1, self.ind2 = torch.triu_indices(node_size, node_size, offset=1)

        seq_len -= kernel_size//2*2
        channel1 = 32
        self.block1 = nn.Sequential(
            Conv1d(in_channels=1, out_channels=channel1,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2  

        seq_len -= kernel_size//2*2
        channel2 = 64
        self.block2 = nn.Sequential(
            Conv1d(in_channels=channel1, out_channels=channel2,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2 

        seq_len -= kernel_size//2*2
        channel3 = 96
        self.block3 = nn.Sequential(
            Conv1d(in_channels=channel2, out_channels=channel3,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU()
        )

        channel4 = 64
        self.block4 = nn.Sequential(
            Conv1d(in_channels=channel3, out_channels=channel4,
                            kernel_size=kernel_size),
            Conv1d(in_channels=channel4, out_channels=channel4,
                            kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2)  
        )
        seq_len -= kernel_size//2*2
        seq_len -= kernel_size//2*2
        seq_len //= 2  
        
               
        self.fc = nn.Linear(in_features=seq_len*channel4, out_features=32)

        self.diff_mode = nn.Sequential(
            nn.Linear(in_features=32*2, out_features=32),
            nn.Linear(in_features=32, out_features=32),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        bz, _, time_series = x.shape

        x = x.reshape((bz*2, 1, time_series))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        
        x = x.reshape((bz, 2, -1))

        x = self.fc(x)

        x = x.reshape((bz, -1))

        diff = self.diff_mode(x)

        return diff
    
class SparseAttentionGraph(nn.Module):
    def __init__(self, input_dim, topk=10):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.topk = topk

    def forward(self, x):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        B, N, _ = attn.size()

        # Top-k sparsification
        topk_vals, topk_idx = torch.topk(attn, self.topk, dim=-1)
        mask = torch.zeros_like(attn)
        mask.scatter_(-1, topk_idx, 1.0)
        sparse_attn = attn * mask

        sparse_attn = F.softmax(sparse_attn, dim=-1)
        return sparse_attn.unsqueeze(-1)
    
class Embed2GraphByAttention(nn.Module):
    def __init__(self, input_dim, topk=20):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.topk = topk

    def forward(self, x):
        # x: [B, N, F]
        Q = self.query_proj(x)  # [B, N, F]
        K = self.key_proj(x)    # [B, N, F]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # [B, N, N]

        B, N, _ = attn.size()

        # Top-k sparsification
        topk_vals, topk_idx = torch.topk(attn, self.topk, dim=-1)
        mask = torch.zeros_like(attn)
        mask.scatter_(-1, topk_idx, 1.0)
        sparse_attn = attn * mask  # [B, N, N]

        sparse_attn = F.softmax(sparse_attn, dim=-1)
        return sparse_attn.unsqueeze(-1)  # [B, N, N, 1]

class FBNETGEN_MultiView(nn.Module):
    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        self.attn_graph = Embed2GraphByAttention(node_input_dim)
        self.product_graph = Embed2GraphByProduct()
        self.predictor = GNNPredictor(node_input_dim, roi_num=roi_num)

    def forward(self, inputs, x):
        # inputs: [B, N, F]
        A1 = self.attn_graph(inputs)      # [B, N, N, 1]
        A2 = self.product_graph(inputs)   # [B, N, N, 1]
        A = (A1 + A2) / 2.0               # fused adjacency: [B, N, N, 1]
        return self.predictor(inputs, A)  # logits: [B, 2]
    
    
import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import torch # Just for torch.tensor type in previous context, not strictly needed for SNA funcs themselves

# --- Model Definitions (from previous responses, assume these are defined elsewhere or imported) ---
# For the full executable script, these would be included.
# class GruKRegion(nn.Module): ...
# class Embed2GraphByProduct(nn.Module): ...
# class GNNPredictor(nn.Module): ...
# class FBNETGEN(nn.Module): ...


class SNAMetricsCalculator:
    """
    A utility class to calculate various Social Network Analysis (SNA) metrics
    from adjacency matrices, typically representing brain functional connectivity graphs.
    """

    def __init__(self, default_threshold: float = 0.0, default_is_weighted: bool = False):
        """
        Initializes the SNAMetricsCalculator with default parameters for graph creation.

        Args:
            default_threshold (float): Default threshold for binarizing graphs.
                                       Edges with weights <= this will be set to 0.
            default_is_weighted (bool): Default flag for creating weighted graphs.
                                        If True, graph edges will have weights; otherwise, they're binary.
        """
        self.default_threshold = default_threshold
        self.default_is_weighted = default_is_weighted

    def _get_param(self, param_value, default_value):
        """Helper to get parameter value, prioritizing argument over default."""
        return param_value if param_value is not None else default_value

    def _create_networkx_graph(self, adj_matrix: np.ndarray, threshold: float = None, is_weighted: bool = None) -> nx.Graph:
        """
        Internal helper to convert a NumPy adjacency matrix into a NetworkX graph object.

        Args:
            adj_matrix (np.ndarray): A 2D NumPy array (K, K) representing the adjacency matrix.
            threshold (float, optional): Overrides default_threshold. If provided, graph is binarized.
            is_weighted (bool, optional): Overrides default_is_weighted. If True, creates a weighted graph.

        Returns:
            nx.Graph: A NetworkX graph object.
        """
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        # Ensure no self-loops for most SNA metrics
        adj_matrix_copy = adj_matrix.copy() # Work on a copy to avoid modifying original
        np.fill_diagonal(adj_matrix_copy, 0)

        use_threshold = self._get_param(threshold, self.default_threshold)
        use_is_weighted = self._get_param(is_weighted, self.default_is_weighted)

        if use_threshold is not None:
            # Create a binary (unweighted) graph based on threshold
            binary_adj_matrix = (adj_matrix_copy > use_threshold).astype(int)
            G = nx.from_numpy_array(binary_adj_matrix)
        elif use_is_weighted:
            # Create a weighted graph
            G = nx.from_numpy_array(adj_matrix_copy)
        else:
            # Create an unweighted graph where all existing non-zero edges have weight 1
            binary_adj_matrix = (adj_matrix_copy != 0).astype(int)
            G = nx.from_numpy_array(binary_adj_matrix)
            
        return G

    # --- Global SNA Metrics ---

    def get_global_efficiency(self, adj_matrix: np.ndarray, threshold: float = None) -> float:
        """
        Calculates the global efficiency of a graph.
        Requires the graph to be connected for a meaningful interpretation across all nodes.
        Uses a binarized graph based on the threshold.
        """
        G_binary = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=False)
        
        if G_binary.number_of_nodes() == 0:
            return np.nan

        if not nx.is_connected(G_binary):
            try:
                largest_cc = G_binary.subgraph(max(nx.connected_components(G_binary), key=len))
                if len(largest_cc.nodes()) > 1:
                    return nx.global_efficiency(largest_cc)
                else: # Single node connected component
                    return np.nan 
            except ValueError: # handle case where max() fails on empty iter
                 return np.nan
        else:
            return nx.global_efficiency(G_binary)

    def get_average_shortest_path_length(self, adj_matrix: np.ndarray, threshold: float = None) -> float:
        """
        Calculates the average shortest path length of a graph.
        Requires the graph to be connected. Uses a binarized graph based on the threshold.
        """
        G_binary = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=False)
        
        if nx.is_connected(G_binary):
            try:
                return nx.average_shortest_path_length(G_binary)
            except nx.NetworkXError: # Catches error if graph has no edges etc.
                return np.nan
        else:
            return np.nan

    def get_transitivity(self, adj_matrix: np.ndarray, is_weighted: bool = None, threshold: float = None) -> float:
        """
        Calculates the transitivity (global clustering coefficient) of a graph.
        Can be calculated for weighted or unweighted graphs.
        """
        G = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=is_weighted)
        return nx.transitivity(G)

    def get_density(self, adj_matrix: np.ndarray, is_weighted: bool = None, threshold: float = None) -> float:
        """
        Calculates the density of a graph.
        NetworkX density is for unweighted graphs.
        """
        # Density is typically calculated for unweighted graphs
        G = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=False) 
        return nx.density(G)

    def get_modularity(self, adj_matrix: np.ndarray, threshold: float = None) -> float:
        """
        Calculates the modularity of a graph after community detection (using Louvain method).
        Requires the 'python-louvain' library. Uses a binarized graph.
        """
        try:
            import community as co # pip install python-louvain
        except ImportError:
            print("Error: 'python-louvain' library not found. Modularity cannot be calculated.")
            return np.nan
        
        G_binary = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=False)
        
        if G_binary.number_of_nodes() == 0 or G_binary.number_of_edges() == 0:
            return np.nan

        try:
            partition = co.best_partition(G_binary)
            modularity = co.modularity(partition, G_binary)
            return modularity
        except Exception as e:
            # print(f"Error calculating modularity: {e}") # Uncomment for debugging
            return np.nan

    # --- Local SNA Metrics ---

    def get_degree_centrality(self, adj_matrix: np.ndarray, is_weighted: bool = None, threshold: float = None) -> dict:
        """
        Calculates the degree centrality for each node (ROI).
        For unweighted, it's the number of connections. For weighted, it's the sum of edge weights.
        """
        G = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=is_weighted)
        use_is_weighted = self._get_param(is_weighted, self.default_is_weighted)
        use_threshold = self._get_param(threshold, self.default_threshold)

        if use_is_weighted and use_threshold is None: # Only apply weight if it's explicitly weighted and not binarized
            return dict(G.degree(weight='weight'))
        else:
            return nx.degree_centrality(G) # This is for unweighted, normalized degree


    def get_clustering_coefficient(self, adj_matrix: np.ndarray, is_weighted: bool = None, threshold: float = None) -> dict:
        """
        Calculates the local clustering coefficient for each node (ROI).
        Measures how interconnected a node's neighbors are.
        """
        G = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=is_weighted)
        use_is_weighted = self._get_param(is_weighted, self.default_is_weighted)
        use_threshold = self._get_param(threshold, self.default_threshold)

        if use_is_weighted and use_threshold is None:
            return nx.clustering(G, weight='weight')
        else:
            return nx.clustering(G)


    def get_betweenness_centrality(self, adj_matrix: np.ndarray, is_weighted: bool = None, threshold: float = None) -> dict:
        """
        Calculates the betweenness centrality for each node (ROI).
        Measures how often a node lies on the shortest path between other pairs of nodes.
        Can be slow for large graphs.
        """
        G = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=is_weighted)
        use_is_weighted = self._get_param(is_weighted, self.default_is_weighted)
        use_threshold = self._get_param(threshold, self.default_threshold)

        if use_is_weighted and use_threshold is None:
            return nx.betweenness_centrality(G, weight='weight') 
        else:
            return nx.betweenness_centrality(G)

    # You can add more local metrics here:
    # def get_closeness_centrality(self, adj_matrix, is_weighted=None, threshold=None):
    #     G = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=is_weighted)
    #     if is_weighted and threshold is None:
    #         return nx.closeness_centrality(G, distance='weight')
    #     else:
    #         return nx.closeness_centrality(G)

    # def get_eigenvector_centrality(self, adj_matrix, is_weighted=None, threshold=None):
    #     G = self._create_networkx_graph(adj_matrix, threshold=threshold, is_weighted=is_weighted)
    #     if is_weighted and threshold is None:
    #         return nx.eigenvector_centrality(G, weight='weight')
    #     else:
    #         return nx.eigenvector_centrality(G)