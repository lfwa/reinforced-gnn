"""Data preparation.

This modules provides function to load and prepare the data from raw csv files.
"""

import numpy as np
import pandas as pd
import surprise
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_data(file_path: str, train_val_split: bool, random_seed: int = 0, train_size: float = 0):
    """Load data from a csv file into a pandas dataframe.

    Args:
        file_path (str): Path to the data file.
        train_val_split (bool): True if the data should be split into a
        training and validation set. False otherwise.
        random_seed (int, optional): Random state for splitting the data if
        train_val_split is true. Defaults to 0.
        train_size (float, optional): Proportion of training data if
        train_val_split is set. Defaults to 0.

    Returns:
        pd.DataFrame: Pandas DataFrame with data from the csv file.
    """
    data_pd = pd.read_csv(file_path)

    if train_val_split:
        train_pd, val_pd = train_test_split(
            data_pd, train_size=train_size, random_state=random_seed
        )
        return train_pd, val_pd
    else:
        return data_pd


def load_reinforcements(file_path: str):
    """Load reinforcements from file path.

    Args:
        file_path (str): Path to reinforcements csv file.

    Returns:
        np.ndarray: Numpy array of reinforcement values.
    """
    data_pd = pd.read_csv(file_path)
    return np.expand_dims(data_pd.Reinforcement.values, axis=1)


def __extract_users_items_predictions(data_pd: pd.DataFrame):
    """Extract users, items, and predictions from dataframe into numpy arrays.

    Args:
        data_pd (pd.DataFrame): Pandas dataframe with the data loaded.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Triple of users, items, and
        predictions as numpy arrays.
    """
    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            data_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1
        )
    ]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def __get_tensors_from_dataframe(data_pd: pd.DataFrame):
    """Retrieve users, items, and predictions from dataframe into tensors.

    Args:
        data_pd (pd.DataFrame): Pandas DataFrame with the data loaded.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Triple of users, items,
        and predictions as tensors.
    """
    users, movies, predictions = __extract_users_items_predictions(data_pd)
    users_torch = torch.tensor(users, dtype=torch.int64)
    movies_torch = torch.tensor(movies, dtype=torch.int64)
    predictions_torch = torch.tensor(predictions, dtype=torch.int64)

    return users_torch, movies_torch, predictions_torch


def create_dataset(data_pd: pd.DataFrame, test_dataset: bool = False):
    """Creates a TensorDataset from the pandas dataframe.

    Args:
        data_pd (pd.DataFrame): Pandas dataframe with the data loaded.
        test_dataset (bool, optional): True if it is the test dataset. False
        otherwise. Defaults to False.

    Returns:
        torch.TensorDataset: TensorDataset with the data.
    """
    users_torch, movies_torch, predictions_torch = __get_tensors_from_dataframe(data_pd)

    if not test_dataset:
        return TensorDataset(users_torch, movies_torch, predictions_torch)
    else:
        test_ids = data_pd.Id
        return test_ids, TensorDataset(users_torch, movies_torch)


def create_dataset_with_reinforcements(
    data_pd: pd.DataFrame, reinforcements: np.array, test_dataset: bool = False
):
    """Creates a TensorDataset from DataFrame with additional reinforcements.

    Args:
        data_pd (pd.DataFrame): Pandas DataFrame with the data loaded.
        reinforcements (np.array): Reinforcements in a numpy array.
        test_dataset (bool, optional): True if it is the test dataset. False
        otherwise. Defaults to False.

    Returns:
        torch.TensorDataset: TensorDataset with the data and reinforcements.
    """
    users_torch, movies_torch, ratings_torch = __get_tensors_from_dataframe(data_pd)
    reinforcements_torch = torch.tensor(reinforcements, dtype=torch.float)

    if not test_dataset:
        return TensorDataset(users_torch, movies_torch, reinforcements_torch, ratings_torch)
    else:
        test_ids = data_pd.Id
        return test_ids, TensorDataset(users_torch, movies_torch, reinforcements_torch)


def create_laplacian_matrix(data_pd: pd.DataFrame, number_of_users: int, number_of_items: int):
    """Create laplacian matrix used in GNN.

    Args:
        data_pd (pd.DataFrame): Pandas dataframe with the data loaded.
        number_of_users (int): Number of users in the data.
        number_of_items (int): Number of items in the data.

    Returns:
        torch.Tensor: Laplacian matrix.
    """
    users_torch, movies_torch, predictions_torch = __get_tensors_from_dataframe(data_pd)

    user_item_matrix = torch.sparse_coo_tensor(
        torch.vstack((users_torch, movies_torch)), predictions_torch
    )
    top_zero_matrix = torch.zeros(
        (user_item_matrix.shape[0], user_item_matrix.shape[0])
    ).to_sparse()
    bottom_zero_matrix = torch.zeros(
        (user_item_matrix.shape[1], user_item_matrix.shape[1])
    ).to_sparse()

    top_a = torch.cat((top_zero_matrix, user_item_matrix), dim=1)
    bottom_a = torch.cat((torch.transpose(user_item_matrix, 0, 1), bottom_zero_matrix), dim=1)
    matrix_a = torch.vstack((top_a, bottom_a))

    degree = (matrix_a.to_dense() > 0).sum(axis=1)
    degree_matrix = torch.diag(torch.pow(degree, -0.5))

    laplacian_matrix = torch.sparse.mm(
        degree_matrix, torch.sparse.mm(matrix_a, degree_matrix)
    ).to_sparse()
    return laplacian_matrix


def create_surprise_data(data_pd):
    """Create dataset used in baselines with the surprise library.

    Args:
        data_pd (pd.DataFrame): Pandas dataframe with the data loaded.

    Returns:
        surprise.Dataset: Surprise dataset.
    """
    users, movies, ratings = __extract_users_items_predictions(data_pd)

    df = pd.DataFrame({"users": users, "movies": movies, "ratings": ratings})
    reader = surprise.Reader(rating_scale=(1, 5))
    return surprise.Dataset.load_from_df(df[["users", "movies", "ratings"]], reader=reader)
