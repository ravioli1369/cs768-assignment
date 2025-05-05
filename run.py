import re
import torch
import shutil
import pickle
import numpy as np
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from pathlib import Path
from glob import glob
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from matplotlib import pyplot as plt
mlp_params = {
    "figure.figsize": [9, 6],
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "axes.titlepad": 15,
    "figure.titlesize": 24,
    "axes.labelpad": 10,
    "font.size": 16,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True if shutil.which("latex") else False,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
}
plt.rcParams.update(mlp_params)


def extract_titles_and_paths(root):
    papers = {}
    folders = []
    for folder in Path(root).iterdir():
        if folder.is_dir():
            folders.append(folder)
            title_file = folder / "title.txt"
            if title_file.exists():
                with open(title_file, "r", encoding="utf-8") as f:
                    title = f.read().strip().lower()
                    title = re.sub(r"[^a-zA-Z0-9]", "", title)
                    title = re.sub(r"\s+", " ", title).strip()
                papers[title] = {
                    "path": folder,
                    "citations": [],
                }
    return papers, folders


class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGELinkPredictor, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.link_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        row, col = edge_index
        z_src = z[row]
        z_dst = z[col]
        z_combined = torch.cat([z_src, z_dst], dim=1)
        return self.link_predictor(z_combined).squeeze()
    
    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_pred = self.decode(z, pos_edge_index)
        neg_pred = self.decode(z, neg_edge_index)
        return pos_pred, neg_pred

def extract_features(paper_folders):
    texts = []
    paper_ids = []
    
    for folder in paper_folders:
        try:
            with open(f"{folder}/title.txt", "r", encoding="utf-8") as f:
                title = f.read().strip().lower()
                title = re.sub(r"[^a-zA-Z0-9]", "", title)
                title = re.sub(r"\s+", " ", title).strip()
            
            with open(f"{folder}/abstract.txt", "r", encoding="utf-8") as f:
                abstract = f.read().strip().lower()
                abstract = re.sub(r"[^a-zA-Z0-9]", "", abstract)
                abstract = re.sub(r"\s+", " ", abstract).strip()
            
            texts.append(title + " " + abstract)
            paper_ids.append(title)
        except:
            continue
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=300)
    features = vectorizer.fit_transform(texts).toarray()
    
    return features, paper_ids

def prepare_data(G, features, paper_ids):    
    # Create mapping from paper IDs to indices
    id_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    
    # Get edges from the graph
    edges = []
    for u, v in G.edges():
        if u in id_to_idx and v in id_to_idx:
            edges.append((id_to_idx[u], id_to_idx[v]))

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    # Split edges for training, validation, and testing
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    
    train_edges = edge_index[:, perm[:int(0.8 * num_edges)]]
    val_edges = edge_index[:, perm[int(0.8 * num_edges):int(0.9 * num_edges)]]
    test_edges = edge_index[:, perm[int(0.9 * num_edges):]]
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=train_edges)
    
    return data, train_edges, val_edges, test_edges

def train(model, data, train_edges, optimizer, batch_size=2048):
    model.train()
    
    # Generate negative samples
    neg_edge_index = negative_sampling(
        edge_index=train_edges,
        num_nodes=data.x.size(0),
        num_neg_samples=train_edges.size(1)
    )
    
    # Process in batches to handle large graphs
    total_loss = 0
    for i in range(0, train_edges.size(1), batch_size):
        optimizer.zero_grad()
        
        batch_pos_edge_index = train_edges[:, i:i+batch_size]
        batch_neg_edge_index = neg_edge_index[:, i:i+batch_size]
        
        pos_pred, neg_pred = model(data.x, data.edge_index, batch_pos_edge_index, batch_neg_edge_index)
        
        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_pos_edge_index.size(1)
    
    return total_loss / train_edges.size(1)

def evaluate(model, data, val_edges, edge_index, k_values=[10, 20, 50, 100]):
    model.eval()
    
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    
    results = {}
    for k in k_values:
        recall = calculate_recall_at_k(model, z, val_edges, k, edge_index)
        results[f'recall@{k}'] = recall
    
    return results

def calculate_recall_at_k(model, z, true_edges, k, edge_index):
    # For each source node, get top-k predictions
    src_nodes = torch.unique(true_edges[0]).tolist()
    correct_predictions = 0
    total_edges = 0
    
    for src in src_nodes:
        # Get all true target nodes for this source
        true_targets = true_edges[1][true_edges[0] == src].tolist()
        if not true_targets:
            continue
        
        # Calculate scores for all possible target nodes
        src_embedding = z[src].repeat(z.size(0), 1)
        all_embeddings = torch.cat([src_embedding, z], dim=1)
        scores = model.link_predictor(all_embeddings).squeeze()
        
        # Exclude existing edges in training graph
        existing_edges = edge_index[1][edge_index[0] == src].tolist()
        for idx in existing_edges:
            scores[idx] = -float('inf')
        
        # Get top-k predictions
        _, top_indices = scores.topk(k)
        top_indices = top_indices.tolist()
        
        # Calculate recall
        hits = len(set(top_indices) & set(true_targets))
        correct_predictions += hits
        total_edges += len(true_targets)
    
    return correct_predictions / total_edges if total_edges > 0 else 0

def predict_citations(model, data, new_paper_features, k=10):
    model.eval()
    
    with torch.no_grad():
        # Get node embeddings from the model
        z = model.encode(data.x, data.edge_index)
        
        # Convert new paper features to tensor
        new_paper_tensor = torch.tensor(new_paper_features, dtype=torch.float).unsqueeze(0)
        
        # For simplicity, we'll use a simple approach: compute similarity directly
        # In a real scenario, you might want to add the new node to the graph and run inference
        new_paper_embedding = model.conv1(new_paper_tensor, torch.zeros((2, 0), dtype=torch.long))
        new_paper_embedding = F.relu(new_paper_embedding)
        new_paper_embedding = model.conv2(new_paper_embedding, torch.zeros((2, 0), dtype=torch.long))
        
        # Compute scores for all potential citations
        new_embedding_repeated = new_paper_embedding.repeat(z.size(0), 1)
        all_embeddings = torch.cat([new_embedding_repeated, z], dim=1)
        scores = model.link_predictor(all_embeddings).squeeze()
        
        # Get top-k predictions
        _, top_indices = scores.topk(k)
        
        return top_indices.tolist()

def main(G, paper_folders):
    # Extract features
    features, paper_ids = extract_features(paper_folders)
    
    # Prepare data
    data, train_edges, val_edges, test_edges = prepare_data(G, features, paper_ids)
    
    # Initialize model
    model = GraphSAGELinkPredictor(in_channels=features.shape[1], hidden_channels=128, out_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    best_val_recall = 0
    for epoch in range(100):
        loss = train(model, data, train_edges, optimizer)
        val_metrics = evaluate(model, data, val_edges)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Recall@10: {val_metrics["recall@10"]:.4f}')
        
        # Save best model
        if val_metrics["recall@10"] > best_val_recall:
            best_val_recall = val_metrics["recall@10"]
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Test the model
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = evaluate(model, data, test_edges)
    print(f'Test Recall@10: {test_metrics["recall@10"]:.4f}')
    
    # Example of predicting citations for a new paper
    # Assume we have features for a new paper
    new_paper_features = extract_features_for_new_paper("path/to/new/paper")
    predicted_citations = predict_citations(model, data, new_paper_features)
    print(f'Predicted citations: {predicted_citations}')

def extract_features_for_new_paper(paper_path):
    # Similar to extract_features but for a single paper
    try:
        with open(f"{paper_path}/title.txt", "r", encoding="utf-8") as f:
            title = f.read().strip()
        
        with open(f"{paper_path}/abstract.txt", "r", encoding="utf-8") as f:
            abstract = f.read().strip()
        
        text = title + " " + abstract
        
        # Use the same vectorizer as in training
        # This is simplified; in practice, you'd need to save the vectorizer
        vectorizer = TfidfVectorizer(max_features=300)
        vectorizer.fit([text])  # This is just a placeholder
        features = vectorizer.transform([text]).toarray()
        
        return features[0]
    except:
        return np.zeros(300)  # Default features if extraction fails

G = pickle.load(open("graph.pickle", "rb"))
papers, folders = extract_titles_and_paths("dataset_papers")

main(G, folders)