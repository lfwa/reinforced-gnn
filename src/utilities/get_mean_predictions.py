"""Retrieve mean predictions for ensembles.

This module retrieves the mean prediction for ensembles when run as main.
"""

import os

import numpy as np
import pandas as pd
from comet_ml import Experiment

from utilities.helper import create_argument_parser, create_comet_logger


def main():
    """Compute mean predictions from ensemble."""
    parser = create_argument_parser()
    args = parser.parse_args()
    comet_logger = create_comet_logger(args)

    if args.ensemble_directory is None:
        print("Please specify the directory with the predictions you want to combine ...")
        exit()

    directory = args.ensemble_directory
    if not os.path.exists(directory):
        print("The provided directory path does not exists ...")
        exit()

    ids, predictions = None, None
    # Search for all files ending with '.csv' and combine them in one numpy array
    for file in os.listdir(directory):
        if file.endswith(".csv") and file.startswith("predictions"):
            cur_predictions_pd = pd.read_csv(directory + "/" + file)
            cur_predictions = np.expand_dims(cur_predictions_pd.Prediction.to_numpy(), axis=1)

            if ids is None:
                ids = cur_predictions_pd.Id
                predictions = cur_predictions
            else:
                predictions = np.concatenate((predictions, cur_predictions), axis=1)

    if ids is None:
        print("No files ending with '.csv' were found in the specified directory ...")
        exit()

    # Compute mean predictions and log them using comet logger
    mean_predictions = np.mean(predictions, axis=1)
    prediction_output = np.stack((ids, mean_predictions), axis=1)
    comet_logger.experiment.log_table(
        filename="predictions.csv", tabular_data=prediction_output, headers=["Id", "Prediction"]
    )


if __name__ == "__main__":
    main()
