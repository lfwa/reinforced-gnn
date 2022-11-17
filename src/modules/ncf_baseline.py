"""NCF.

This module contains the NCF baseline model.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utilities.evaluation_functions import get_score

# Hyper Parameters used for the Model
hyper_parameters = {
    "batch_size": 1024,
    "num_epochs": 25,
    "number_of_users": 10000,
    "number_of_movies": 1000,
    "user_embedding_size": 10000,
    "movie_embedding_size": 1000,
    "learning_rate": 1e-3,
    "train_size": 0.9,
    "dropout": 0.5,
}


class NCF(pl.LightningModule):
    """NCF model."""

    def __init__(self, train_data, val_data, test_data, test_ids, args, config):
        """Initialize model.

        Args:
            train_data (torch.TensorDataSet): Training data.
            val_data (torch.TensorDataSet): Validation data.
            test_data (torch.TensorDataset): Test data.
            test_ids (pd.DataFrame): Testing IDs.
            args: args.
            config (dict): config.
        """
        super().__init__()

        self.args = args

        # Configuration used for execution
        self.config = config

        # Parameters of the network
        self.number_of_users = config["number_of_users"]
        self.number_of_movies = config["number_of_movies"]
        self.user_embedding_size = config["user_embedding_size"]
        self.movie_embedding_size = config["movie_embedding_size"]
        self.dropout = config["dropout"]

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        # Loss function for training and evaluation
        self.loss = nn.MSELoss()

        # Layer for one-hot encoding of the users
        self.one_hot_encoding_users = nn.Embedding(self.number_of_users, self.number_of_users)
        self.one_hot_encoding_users.data = torch.eye(self.number_of_users)
        # Layer for one-hot encoding of the movies
        self.one_hot_encoding_movies = nn.Embedding(self.number_of_movies, self.number_of_movies)
        self.one_hot_encoding_movies.data = torch.eye(self.number_of_movies)

        # Dense layers for getting embedding of users and movies
        self.embedding_layer_users = nn.Linear(self.number_of_users, self.user_embedding_size)
        self.embedding_layer_movies = nn.Linear(self.number_of_movies, self.movie_embedding_size)

        # Neural network used for training on concatenation of users and movies embedding
        self.feed_forward = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=self.user_embedding_size + self.movie_embedding_size, out_features=64
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
            nn.ReLU(),
        )

    def forward(self, users, movies):
        """Forward pass through the model.

        Args:
            users (torch.Tensor): User tensor.
            movies (torch.Tensor): Item tensor.

        Returns:
            torch.Tensor: Result tensor.
        """
        # Transform users and movies to one-hot encodings
        users_one_hot = self.one_hot_encoding_users(users)
        movies_one_hot = self.one_hot_encoding_movies(movies)

        # Compute embedding of users and movies
        users_embedding = self.embedding_layer_users(users_one_hot)
        movies_embedding = self.embedding_layer_movies(movies_one_hot)

        # Train rest of neural network on concatenation of user and movie embeddings
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))

    def training_step(self, batch, batch_idx):
        """Run a training step.

        Args:
            batch (torch.TensorDataset): Batch.
            batch_idx (int): Index of batch.

        Returns:
            float: Loss after taking a step.
        """
        users, movies, ratings = batch

        predictions = self(users, movies)
        loss = self.loss(predictions, ratings.float())
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Run a validation step.

        Args:
            batch (torch.TensorDataset): Batch.
            batch_idx (int): Index of batch.

        Returns:
            float: Validation loss after taking a step.
        """
        users, movies, ratings = batch

        predictions = self(users, movies)
        val_loss = self.loss(predictions, ratings.float())
        score = get_score(predictions.cpu().numpy(), ratings.cpu().numpy())
        self.log("val_loss", val_loss)
        self.log("score", score)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Run a test step.

        Args:
            batch (torch.TensorDataset): Batch.
            batch_idx (int): Index of batch.

        Returns:
            torch.Tensor: Predictions.
        """
        users, movies = batch
        predictions = self(users, movies)
        return predictions

    def test_epoch_end(self, outputs):
        """Log output after epoch ends.

        Args:
            outputs (torch.Tensor): Model output.
        """
        predictions = torch.cat(outputs, dim=0).cpu().numpy()
        predictions = np.clip(predictions.astype(np.float), a_min=1, a_max=5)
        prediction_output = np.stack((self.test_ids, predictions), axis=1)

        self.logger.experiment.log_table(
            filename="predictions.csv", tabular_data=prediction_output, headers=["Id", "Prediction"]
        )

    def configure_optimizers(self):
        """Setup and retrieve optimizer.

        Returns:
            torch.optim.Adam: Adam optimizer.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def train_dataloader(self):
        """Retrieve dataloader for training data.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.args.dataloader_workers,
        )

    def val_dataloader(self):
        """Retrieve dataloader for validation data.

        Returns:
            torch.utils.data.DataLoader: Validation dataloader.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.args.dataloader_workers,
        )

    def test_dataloader(self):
        """Retrieve dataloader for test data.

        Returns:
            torch.utils.data.DataLoader: Testing dataloader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.args.dataloader_workers,
        )
