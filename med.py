import os
import json
import pickle
from collections import defaultdict
from typing import List

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel
from llama_index.llms.groq import Groq

# -------------------------- Constants and File Paths --------------------------

TRIPLETS_FILE = "triplets.json"
SYMPTOM_EMBEDDINGS_FILE = "symptom_embeddings.pkl"
LEVEL2_CLUSTERS_FILE = "level2_clusters.json"
LEVEL1_CLUSTERS_FILE = "level1_clusters.json"
GRAPH_FILE = "knowledge_graph.pkl"
EXCEL_FILE = "data.xlsx"

# ---------------------------- Load Medical Dataset ----------------------------

df = pd.read_excel(EXCEL_FILE)

# ------------------------- Initialize LLM and BioBERT --------------------------

llm = Groq(
    model="llama3-70b-8192",
    api_key="Groq_api_key"
)

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert.eval()

# ---------------------------- Utility: Save/Load ------------------------------

def save_json(obj, filename):
    def convert_keys_and_values(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys_and_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys_and_values(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(filename, 'w') as f:
        json.dump(convert_keys_and_values(obj), f, indent=2)

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# -------------------------- Triplet Creation (QA → Fact) ----------------------

def convert_to_fact_llama_groq(question: str, answer: str) -> str:
    prompt = f"""Convert the following medical question and answer into a natural language sentence.

Q: {question.strip()}
A: {answer.strip()}

Fact:"""
    response = llm.complete(prompt)
    return response.text.strip()

def generate_or_load_triplets():
    triplets = load_json(TRIPLETS_FILE)
    if triplets:
        print("Loaded saved triplets.")
        return triplets

    print("Generating triplets...")
    triplets = []
    for _, row in df.iterrows():
        subject, relation, qa_string = map(str.strip, row.iloc[:3])
        obj = {"fact": convert_to_fact_llama_groq(*qa_string.split("?:", 1))} if "?:" in qa_string else {"fact": qa_string}
        triplets.append({"subject": subject, "relation": relation, "object": obj})

    save_json(triplets, TRIPLETS_FILE)
    return triplets

# --------------------------- Embedding + Clustering ---------------------------

def get_embedding(text,tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = biobert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

def generate_or_load_symptom_embeddings(graph_nodes):
    symptom_embeddings = load_pickle(SYMPTOM_EMBEDDINGS_FILE)
    if symptom_embeddings:
        print("Loaded saved symptom embeddings.")
        return symptom_embeddings

    print("Generating symptom embeddings...")
    symptom_embeddings = {}
    for node in tqdm(graph_nodes):
        symptom_embeddings[node] = get_embedding(node)
    
    save_pickle(symptom_embeddings, SYMPTOM_EMBEDDINGS_FILE)
    return symptom_embeddings

def name_cluster(text_list: List[str], level: str = "2") -> str:
    diseases = ", ".join(text_list)
    prompt = f"""
    You are a medical expert. The following is a list of {'diseases' if level == '2' else 'disease categories'}:

    {diseases}

    Suggest a concise and medically accurate name that categorizes this group at level {level}.
    Only output the name, nothing else.
    """
    return llm.complete(prompt).text.strip()

def cluster_making(subjects: list, n_clusters: int, level: int):
    print(f"Clustering level {level} with {n_clusters} clusters...")
    disease_embeddings = [(d, get_embedding(d)) for d in tqdm(subjects)]
    X = np.array([emb for _, emb in disease_embeddings])

    agglo = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = agglo.fit_predict(X)

    level_clusters = defaultdict(list)
    for (disease, _), label in zip(disease_embeddings, labels):
        level_clusters[label].append(disease)

    level_named = {}
    for label, diseases in level_clusters.items():
        name = name_cluster(diseases, level=str(level))
        level_named[label] = {"name": name, "diseases": diseases}

    return level_named, [v["name"] for v in level_named.values()]

# --------------------------- Knowledge Graph Creation -------------------------

def generate_or_load_graph(triplets, level2_named, level1_named):
    if os.path.exists(GRAPH_FILE):
        print("Loading saved knowledge graph...")
        return load_pickle(GRAPH_FILE)

    print("Creating new knowledge graph...")
    G = nx.MultiDiGraph()

    for triplet in triplets:
        subj, rel, obj_text = triplet["subject"], triplet["relation"], triplet["object"]["fact"]
        if(rel=='has_symptomatology'):
            G.add_edge(subj, obj_text, label=rel)
            G.add_edge(obj_text, subj, label=f"{rel}_reverse")

    for lvl2_info in level2_named.values():
        lvl2_name = lvl2_info["name"]
        for disease in lvl2_info["diseases"]:
            G.add_edge(disease, lvl2_name, label="is_a")
            G.add_edge(lvl2_name, disease, label="is_a_reverse")

    for lvl1_info in level1_named.values():
        lvl1_name = lvl1_info["name"]
        for lvl2_cat in lvl1_info.get("diseases", []):  # assuming nested cluster names passed here
            G.add_edge(lvl2_cat, lvl1_name, label="is_a")
            G.add_edge(lvl1_name, lvl2_cat, label="is_a_reverse")

    save_pickle(G, GRAPH_FILE)
    return G

# ------------------------------ Main Pipeline ---------------------------------

def build_knowledge():
    triplets = generate_or_load_triplets()
    unique_subjects = list(set(t['subject'] for t in triplets))

    # remove `is_a` targets from subject pool
    is_a_objects = {str(t['object']) for t in triplets if t['relation'] == "is_a"}
    filtered_subjects = [s for s in unique_subjects if s.lower() not in is_a_objects]

    # Load or build level 2 clusters
    level2_named = load_json(LEVEL2_CLUSTERS_FILE)
    if not level2_named:
        level2_named, unique_level2 = cluster_making(filtered_subjects, 16, 2)
        save_json(level2_named, LEVEL2_CLUSTERS_FILE)
    else:
        unique_level2 = [v["name"] for v in level2_named.values()]

    # Load or build level 1 clusters
    level1_named = load_json(LEVEL1_CLUSTERS_FILE)
    if not level1_named:
        level1_named, _ = cluster_making(unique_level2, 7, 1)
        save_json(level1_named, LEVEL1_CLUSTERS_FILE)

    # Load or generate graph
    G = generate_or_load_graph(triplets, level2_named, level1_named)
    symptom_nodes = [triplet["object"]["fact"] for triplet in triplets if triplet["relation"] == "has_symptomatology"]
    symptom_embeddings = generate_or_load_symptom_embeddings(symptom_nodes)
    return {
        "triplets": triplets,
        "level2_clusters": level2_named,
        "level1_clusters": level1_named,
        "graph": G,
        "embeddings": symptom_embeddings,
    }
    
