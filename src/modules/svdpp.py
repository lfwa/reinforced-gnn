"""SVD++.

This module contains a class for the SVD++ model.
"""

import numpy as np
import surprise

# Hyper Parameters used for the Model
hyper_parameters = {
    "n_factors": 101,
    "n_epochs": 24,
    "init_std_dev": 0.08495244489981713,
    "lr_all": 0.0009125851561880716,
    "reg_all": 0.06340961885100015,
}


class SVDpp:
    """SVD++ model."""

    def __init__(self, train_data, test_data, test_ids, args, config, logger):
        """Initialize model.

        Args:
            train_data (surprise.DataSet): Training data.
            test_data (torch.TensorDataset): Test data.
            test_ids (pd.DataFrame): Testing IDs.
            args: args.
            config (dict): config.
            logger (CometLogger): Experiment logger.
        """
        self.args = args

        # Configuration used for execution
        self.config = config

        self.logger = logger

        self.train_data = train_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        self.n_factors = config["n_factors"]
        self.n_epochs = config["n_epochs"]
        self.init_std_dev = config["init_std_dev"]
        self.lr_all = config["lr_all"]
        self.reg_all = config["reg_all"]

        self.algo = surprise.SVDpp(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            init_std_dev=self.init_std_dev,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=7,
        )

    def fit(self):
        """Fit model to training data."""
        self.algo.fit(self.train_data)

    def predict(self, data):
        """Predict from data.

        Args:
            data (torch.TensorDataSet): Data.

        Returns:
            list: Predictions in a list.
        """
        predictions = []
        for user, movie in data:
            prediction = self.algo.predict(user.item(), movie.item()).est
            predictions.append(prediction)
        return predictions

    def test(self):
        """Test model by generating predictions and logging to logger."""
        predictions = self.predict(self.test_data)
        prediction_output = np.stack((self.test_ids, predictions), axis=1)
        self.logger.experiment.log_table(
            filename="predictions.csv", tabular_data=prediction_output, headers=["Id", "Prediction"]
        )
