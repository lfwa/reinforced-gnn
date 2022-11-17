"""SlopeOne.

This module contains a class for the SlopeOne model.
"""

import numpy as np
import surprise


class SlopeOne:
    """SlopeOne model."""

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

        self.algo = surprise.SlopeOne()

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
