import re
import torch
import pickle
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from run import GraphSAGELinkPredictor
import argparse

################################################
#               IMPORTANT                      #
################################################
# 1. Do not print anything other than the ranked list of papers.
# 2. Do not forget to remove all the debug prints while submitting.

def extract_folders(root):
    folders = []
    for folder in Path(root).iterdir():
        if folder.is_dir():
            folders.append(folder)
    return folders

def extract_features(paper_folders):
    texts = []
    paper_ids = []
    
    for folder in paper_folders:
        # try:
        with open(f"{folder}/title.txt", "r", encoding="utf-8") as f:
            title = f.read().strip().lower()
            title = re.sub(r"[^a-zA-Z0-9]\s", "", title)
            title = re.sub(r"\s+", " ", title).strip()
        
        with open(f"{folder}/abstract.txt", "r", encoding="utf-8") as f:
            abstract = f.read().strip().lower()
            abstract = re.sub(r"[^a-zA-Z0-9]\s", "", abstract)
            abstract = re.sub(r"\s+", " ", abstract).strip()
        
        texts.append(title + ";" + abstract)
        paper_ids.append(title)
        # except:
            # continue
    
    # create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=300)
    features = vectorizer.fit_transform(texts).toarray()
    
    return features, paper_ids, vectorizer

def prepare_data(G, features, paper_ids):
    id_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    
    edges = []
    for u, v in G.edges():
        if u in id_to_idx and v in id_to_idx:
            edges.append((id_to_idx[u], id_to_idx[v]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    # print(args)

    ################################################
    #               YOUR CODE START                #
    ################################################
    G = pickle.load(open("graph.pickle", "rb"))
    folders = extract_folders("dataset_papers")
    features, paper_ids, vectorizer = extract_features(folders)

    model = GraphSAGELinkPredictor(in_channels=features.shape[1], hidden_channels=128, out_channels=64)
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()

    # Use the same vectorizer from training for the test paper
    title = re.sub(r"[^a-zA-Z0-9]\s", "", args.test_paper_title.strip().lower())
    title = re.sub(r"\s+", " ", title).strip()
    abstract = re.sub(r"[^a-zA-Z0-9]\s", "", args.test_paper_abstract.strip().lower())
    abstract = re.sub(r"\s+", " ", abstract).strip()
    text = title + ";" + abstract
    new_features = vectorizer.transform([text]).toarray()[0]

    data = prepare_data(G, features, paper_ids)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        
        new_paper_tensor = torch.tensor(new_features, dtype=torch.float).unsqueeze(0)
        new_paper_embedding = model.conv1(new_paper_tensor, torch.zeros((2, 0), dtype=torch.long))
        new_paper_embedding = F.relu(new_paper_embedding)
        new_paper_embedding = model.conv2(new_paper_embedding, torch.zeros((2, 0), dtype=torch.long))
        
        # compute scores for all potential citations
        new_embedding_repeated = new_paper_embedding.repeat(z.size(0), 1)
        all_embeddings = torch.cat([new_embedding_repeated, z], dim=1)
        scores = model.link_predictor(all_embeddings).squeeze()
        
        # get top-k predictions
        _, top_indices = scores.topk(10)
        result = [paper_ids[i] for i in top_indices.tolist()]    


    ################################################
    #               YOUR CODE END                  #
    ################################################


    
    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()
