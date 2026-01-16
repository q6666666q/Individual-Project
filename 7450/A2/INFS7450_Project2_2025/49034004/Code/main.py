from text import load_data, display_data_info
from utilities import build_graph, predict_links
"""Main program"""

def main():
    # Load and inspect data
    train_df, test_df = load_data()
    display_data_info(train_df, test_df)

    # Build graph
    G = build_graph(train_df)

    # Predict links using supervised learning
    train_edges = list(G.edges())
    predict_links(G, test_df, train_edges, output_file='49034004.csv')


if __name__ == "__main__":
    main()