import pandas as pd
import networkx as nx
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
"""This part of the code is mainly the implementation of the prediction function"""

# Build graph
def build_graph(train_df):
    """Build an undirected graph from training data"""
    if train_df is None:
        return None
    G = nx.Graph()
    edges = train_df[['node1', 'node2']].values
    G.add_edges_from(edges)
    print("Graph built with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges")
    return G


# Compute similarity scores
def compute_scores(G, edges, method):
    """Compute similarity scores for edges"""
    if method == 'jaccard':
        scores = nx.jaccard_coefficient(G, edges)
    elif method == 'resource_allocation':
        scores = nx.resource_allocation_index(G, edges)
    elif method == 'adamic_adar':
        scores = nx.adamic_adar_index(G, edges)
    elif method == 'preferential_attachment':
        scores = nx.preferential_attachment(G, edges)
    elif method == 'shortest_path':
        scores = []
        for u, v in edges:
            try:
                path_length = nx.shortest_path_length(G, u, v)
                score = 1.0 / (path_length + 1)  # Avoid division by zero
            except nx.NetworkXNoPath:
                score = 0.0
            scores.append((u, v, score))
        return scores
    elif method == 'clustering_coefficient':
        clustering = nx.clustering(G)
        scores = [(u, v, (clustering.get(u, 0) + clustering.get(v, 0)) / 2) for u, v in edges]
        return scores
    else:
        raise ValueError("Unsupported method / 不支持的方法")
    return [(u, v, score) for u, v, score in scores]


# Compute node-level features
def compute_node_features(G, edges):
    """Compute node degrees and centrality for edges"""
    degrees = dict(G.degree())
    centrality = nx.betweenness_centrality(G, k=100)  # Approximate centrality for efficiency
    features = []
    for u, v in edges:
        deg_u, deg_v = degrees.get(u, 0), degrees.get(v, 0)
        cent_u, cent_v = centrality.get(u, 0), centrality.get(v, 0)
        features.append({
            'degree_u': deg_u,
            'degree_v': deg_v,
            'centrality_u': cent_u,
            'centrality_v': cent_v
        })
    return features


# Generate training data for supervised learning
def generate_training_data(G, train_edges, neg_samples):
    """Generate training features and labels"""
    # Positive samples
    pos_scores = {
        method: compute_scores(G, train_edges, method)
        for method in ['jaccard', 'resource_allocation', 'adamic_adar', 'preferential_attachment', 'shortest_path',
                       'clustering_coefficient']
    }
    pos_node_features = compute_node_features(G, train_edges)
    pos_df = pd.DataFrame({
        'edge': [(u, v) for u, v, _ in pos_scores['jaccard']],
        'jaccard': [score for _, _, score in pos_scores['jaccard']],
        'resource_allocation': [score for _, _, score in pos_scores['resource_allocation']],
        'adamic_adar': [score for _, _, score in pos_scores['adamic_adar']],
        'preferential_attachment': [score for _, _, score in pos_scores['preferential_attachment']],
        'shortest_path': [score for _, _, score in pos_scores['shortest_path']],
        'clustering_coefficient': [score for _, _, score in pos_scores['clustering_coefficient']],
        'degree_u': [f['degree_u'] for f in pos_node_features],
        'degree_v': [f['degree_v'] for f in pos_node_features],
        'centrality_u': [f['centrality_u'] for f in pos_node_features],
        'centrality_v': [f['centrality_v'] for f in pos_node_features],
        'label': 1
    })

    # Negative samples
    neg_scores = {
        method: compute_scores(G, neg_samples, method)
        for method in ['jaccard', 'resource_allocation', 'adamic_adar', 'preferential_attachment', 'shortest_path',
                       'clustering_coefficient']
    }
    neg_node_features = compute_node_features(G, neg_samples)
    neg_df = pd.DataFrame({
        'edge': [(u, v) for u, v, _ in neg_scores['jaccard']],
        'jaccard': [score for _, _, score in neg_scores['jaccard']],
        'resource_allocation': [score for _, _, score in neg_scores['resource_allocation']],
        'adamic_adar': [score for _, _, score in neg_scores['adamic_adar']],
        'preferential_attachment': [score for _, _, score in neg_scores['preferential_attachment']],
        'shortest_path': [score for _, _, score in neg_scores['shortest_path']],
        'clustering_coefficient': [score for _, _, score in neg_scores['clustering_coefficient']],
        'degree_u': [f['degree_u'] for f in neg_node_features],
        'degree_v': [f['degree_v'] for f in neg_node_features],
        'centrality_u': [f['centrality_u'] for f in neg_node_features],
        'centrality_v': [f['centrality_v'] for f in neg_node_features],
        'label': 0
    })

    # Combine data
    train_data = pd.concat([pos_df, neg_df], ignore_index=True)
    X = train_data[['jaccard', 'resource_allocation', 'adamic_adar', 'preferential_attachment', 'shortest_path',
                    'clustering_coefficient', 'degree_u', 'degree_v', 'centrality_u', 'centrality_v']]
    y = train_data['label']
    return X, y


# Predict links using supervised learning
def predict_links_supervised(G, test_edges, train_edges, neg_samples, output_file='49034004.csv'):
    """Predict top-100 edges using supervised learning"""
    start_time = time.time()

    # Generate training data
    X_train, y_train = generate_training_data(G, train_edges, neg_samples)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train logistic regression model
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Compute test set features
    test_scores = {
        method: compute_scores(G, test_edges, method)
        for method in ['jaccard', 'resource_allocation', 'adamic_adar', 'preferential_attachment', 'shortest_path',
                       'clustering_coefficient']
    }
    test_node_features = compute_node_features(G, test_edges)
    test_df = pd.DataFrame({
        'edge': [(u, v) for u, v, _ in test_scores['jaccard']],
        'jaccard': [score for _, _, score in test_scores['jaccard']],
        'resource_allocation': [score for _, _, score in test_scores['resource_allocation']],
        'adamic_adar': [score for _, _, score in test_scores['adamic_adar']],
        'preferential_attachment': [score for _, _, score in test_scores['preferential_attachment']],
        'shortest_path': [score for _, _, score in test_scores['shortest_path']],
        'clustering_coefficient': [score for _, _, score in test_scores['clustering_coefficient']],
        'degree_u': [f['degree_u'] for f in test_node_features],
        'degree_v': [f['degree_v'] for f in test_node_features],
        'centrality_u': [f['centrality_u'] for f in test_node_features],
        'centrality_v': [f['centrality_v'] for f in test_node_features]
    })

    # Scale test features
    X_test = test_df[['jaccard', 'resource_allocation', 'adamic_adar', 'preferential_attachment', 'shortest_path',
                      'clustering_coefficient', 'degree_u', 'degree_v', 'centrality_u', 'centrality_v']]
    X_test_scaled = scaler.transform(X_test)

    # Predict probabilities
    test_df['prob'] = model.predict_proba(X_test_scaled)[:, 1]

    # Post-processing: Boost scores for edges involving hub nodes
    degrees = dict(G.degree())
    hub_threshold = np.percentile(list(degrees.values()), 95)  # Top 5% degree nodes as hubs
    test_df['hub_boost'] = test_df.apply(
        lambda row: 1.2 if (degrees.get(row['edge'][0], 0) > hub_threshold or degrees.get(row['edge'][1],
                                                                                          0) > hub_threshold) else 1.0,
        axis=1
    )
    test_df['final_score'] = test_df['prob'] * test_df['hub_boost']

    # Select top-100 edges
    top_edges = test_df.sort_values(by='final_score', ascending=False).head(100)
    pred_df = pd.DataFrame(top_edges['edge'].tolist(), columns=['node1', 'node2'])
    pred_df.to_csv(output_file, index=False, header=False)

    end_time = time.time()
    print("Predictions saved to", output_file)
    print("Prediction time:", end_time - start_time, "seconds")


# Main prediction function
def predict_links(G, test_df, train_edges, output_file='49034004.csv'):
    """Predict links using supervised learning"""
    if test_df is None or G is None:
        return
    test_edges = test_df[['node1', 'node2']].values.tolist()

    # Generate negative samples
    neg_samples = []
    all_nodes = list(G.nodes())
    while len(neg_samples) < len(train_edges):
        u, v = np.random.choice(all_nodes, 2, replace=False)
        if not G.has_edge(u, v):
            neg_samples.append((u, v))

    # Predict links
    predict_links_supervised(G, test_edges, train_edges, neg_samples, output_file)