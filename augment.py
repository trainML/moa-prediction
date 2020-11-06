"""
Data augmentation. For now we are doing oversampling via MLSMOTE algorithm.
"""
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

def get_tail_labels(df: pd.DataFrame, ql=[0.03, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_labels = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_labels


def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.03, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_labels(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub


def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Given dataframe X and N number of neighbors, 
    returns indices of N nearest neighbors for each target/row.
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_samples, n_neighbors=5):
    """
    Give the augmented data using MLSMOTE algorithm.
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_samples: int, number of newly generated sample
    n_neighbors: number of neighbors to consider when MLSMOTE draws new data points.
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=n_neighbors)
    n = len(indices2)
    new_X = np.zeros((n_samples, X.shape[1]))
    target = np.zeros((n_samples, y.shape[1]))
    for i in range(n_samples):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbor,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target


def augment_data(X, y, oversample_args: tuple):
    """
    Main function to handle all data augmentation steps.
    For now we are only doing oversampling using the MLSMOTE algorithm.

    args
    X - original feature dataset
    y - original targets dataset
    oversample_args - tuple consisting of (n_samples, n_neighbors)
    (see MLSMOTE function comment)

    return
    X_augmented - augmented feature dataset
    y_augmented - augmented targets dataset
    """
    n_samples, n_neighbors = oversample_args

    X_sub, y_sub = get_minority_samples(X, y)
    X_res, y_res = MLSMOTE(X_sub, y_sub, n_samples, n_neighbors)
    X_augmented = pd.concat([X, X_res])
    y_augmented = pd.concat([y, y_res])
    return X_augmented, y_augmented

