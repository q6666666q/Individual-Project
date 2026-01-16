import pandas as pd
"""This part of the code is mainly the data preprocessing section, used to view the data and check whether there are any errors in the data"""



def load_data(train_file='Data/trainingset.csv', test_file='Data/testset.csv'):
    """Load datasets and return DataFrames"""
    train_df = pd.read_csv(train_file, header=None, names=['node1', 'node2'])
    test_df = pd.read_csv(test_file, header=None, names=['node1', 'node2'])
    return train_df, test_df


def display_data_info(train_df, test_df):
    """Display basic dataset information"""
    if train_df is None or test_df is None:
        return
    print("Training Set (first 5 rows):")
    print(train_df.head())
    print("\nTest Set (first 5 rows):")
    print(test_df.head())
    print("\nTraining edges", len(train_df))
    print("Test edges", len(test_df))
    print("Training nodes range",
          train_df['node1'].min(), "to", train_df['node1'].max(),
          "and", train_df['node2'].min(), "to", train_df['node2'].max())
    print("Test nodes range",
          test_df['node1'].min(), "to", test_df['node1'].max(),
          "and", test_df['node2'].min(), "to", test_df['node2'].max())


def main():
    train_df, test_df = load_data()
    display_data_info(train_df, test_df)


if __name__ == '__main__':
    main()