"""GNN NCF.

This module contains the GNN NCF model.
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
    "num_epochs": 60,
    "number_of_users": 10000,
    "number_of_movies": 1000,
    "embedding_size": 64,
    "num_embedding_propagation_layers": 2,
    "learning_rate": 5e-4,
    "train_size": 0.9,
    "patience": 3,
    "dropout": 0,
}


class GNN(pl.LightningModule):
    """GNN NCF model."""

    def __init__(self, train_data, val_data, test_data, test_ids, args, laplacian_matrix, config):
        """Initialize model.

        Args:
            train_data (torch.TensorDataSet): Training data.
            val_data (torch.TensorDataSet): Validation data.
            test_data (torch.TensorDataset): Test data.
            test_ids (pd.DataFrame): Testing IDs.
            args: args.
            laplacian_matrix (torch.Tensor): Laplacian matrix.
            config (dict): config.
        """
        super().__init__()

        self.args = args

        # Configuration used for execution
        self.config = config

        # Parameters of the network
        self.number_of_users = config["number_of_users"]
        self.number_of_movies = config["number_of_movies"]
        self.embedding_size = config["embedding_size"]
        self.num_embedding_propagation_layers = config["num_embedding_propagation_layers"]
        self.dropout = config["dropout"]

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        # Loss function for training and evaluation
        self.loss = nn.MSELoss()

        # Layers for embedding users and movies
        self.embedding_users = nn.Embedding(self.number_of_users, self.embedding_size)
        self.embedding_movies = nn.Embedding(self.number_of_movies, self.embedding_size)

        # Laplacian and Identity Matrices for the Embedding Propagation Layers
        self.laplacian_matrix = laplacian_matrix.to(self.device)
        self.identity = torch.eye(self.number_of_users + self.number_of_movies).to_sparse()

        # List of Embedding Propagation Layers
        self.embedding_propagation_layers = torch.nn.ModuleList(
            [
                self.EmbeddingPropagationLayers(
                    self.laplacian_matrix,
                    self.identity,
                    in_features=self.embedding_size,
                    out_features=self.embedding_size,
                )
                for i in range(self.num_embedding_propagation_layers)
            ]
        )

        # Feedforward network used to make predictions from
        # the embedding propagation layers.
        input_size = 2 * self.num_embedding_propagation_layers * self.embedding_size
        self.feed_forward = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=input_size, out_features=64),
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

    def get_initial_embeddings(self):
        """Retrieve initial embeddings.

        Returns:
            torch.LongTensor: Initial embeddings.
        """
        users = torch.LongTensor([i for i in range(self.number_of_users)]).to(self.device)
        movies = torch.LongTensor([i for i in range(self.number_of_movies)]).to(self.device)

        users_embedding = self.embedding_users(users)
        movies_embedding = self.embedding_movies(movies)
        return torch.cat((users_embedding, movies_embedding), dim=0)

    def forward(self, users, movies):
        """Forward pass through the model.

        Args:
            users (torch.Tensor): User tensor.
            movies (torch.Tensor): Item tensor.

        Returns:
            torch.Tensor: Result tensor.
        """
        current_embedding = self.get_initial_embeddings()
        final_embedding = None
        for layer in self.embedding_propagation_layers:
            current_embedding = layer(current_embedding, self.device)
            if final_embedding is None:
                final_embedding = current_embedding
            else:
                final_embedding = torch.cat((final_embedding, current_embedding), dim=1)

        users_embedding = final_embedding[users]
        movies_embedding = final_embedding[movies + self.number_of_users]
        concat = torch.cat((users_embedding, movies_embedding), dim=1)

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

    class EmbeddingPropagationLayers(nn.Module):
        """Internal embedding propagation layers for the GNN."""

        def __init__(self, laplacian_matrix, identity, in_features, out_features):
            """Initialize propagation layers.

            Args:
                laplacian_matrix (torch.Tensor): Laplacian matrix.
                identity (torch.Tensor): Identity matrix.
                in_features (int): Number of input features.
                out_features (int): Number of output features.
            """
            super().__init__()

            # Laplacian Matrix used in the Embedding Layer
            self.laplacian_matrix = laplacian_matrix

            # Identity Matrix used in the Embedding Layer
            self.identity = identity

            # Linear transformation Layers used internally
            self.transformation_layer_1 = nn.Linear(in_features, out_features)
            self.transformation_layer_2 = nn.Linear(in_features, out_features)

        def forward(self, previous_embedding, device):
            """Forward pass through embedding layers.

            Args:
                previous_embedding (torch.Tensor): Previous embeddings.
                device: Device.

            Returns:
                torch.Tensor: Result tensor.
            """
            self.laplacian_matrix = self.laplacian_matrix.to(device)
            self.identity = self.identity.to(device)

            embedding_1 = torch.sparse.mm(
                (self.laplacian_matrix + self.identity), previous_embedding
            )
            embedding_2 = torch.mul(
                torch.sparse.mm(self.laplacian_matrix, previous_embedding), previous_embedding
            )

            transformed_embedding_1 = self.transformation_layer_1(embedding_1)
            transformed_embedding_2 = self.transformation_layer_2(embedding_2)

            return nn.LeakyReLU()(transformed_embedding_1 + transformed_embedding_2)
