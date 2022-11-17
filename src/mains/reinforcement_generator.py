"""Reinforcement generator.

This module generates reinforcements to use with the reinforced GNN NCF model
when executed as main.
"""

import numpy as np
import pandas as pd
from comet_ml import Experiment

from modules import nmf, slopeone, svd_unbiased, svdpp
from utilities.data_preparation import create_dataset, create_surprise_data, load_data
from utilities.helper import (
    create_argument_parser,
    create_cache_directories,
    create_comet_logger,
    get_config,
)

hyper_parameters = {
    "generate_svd": True,
    "generate_nmf": True,
    "generate_slopeone": True,
    "generate_svdpp": True,
    "train_size": 0.9,
}


def main():
    """Generate reinforcements."""
    parser = create_argument_parser()
    args = parser.parse_args()

    np.random.seed(7)

    config = get_config(args, hyper_parameters)
    create_cache_directories(config)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(config)

    train_pd, val_pd = load_data(
        file_path=args.data_dir + args.train_data,
        train_val_split=True,
        random_seed=args.random_seed,
        train_size=config["train_size"],
    )
    test_pd = load_data(file_path=args.data_dir + args.test_data, train_val_split=False)

    # Write the split data to the cache for later use
    train_pd.to_csv("cache/train_data.csv", index=False)
    val_pd.to_csv("cache/val_data.csv", index=False)
    test_pd.to_csv("cache/test_data.csv", index=False)

    _, train_data = create_dataset(train_pd, test_dataset=True)
    _, val_data = create_dataset(val_pd, test_dataset=True)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)

    surprise_train_data = create_surprise_data(train_pd).build_full_trainset()

    # Add all models for which predictions will be made
    models = []
    if config["generate_svd"]:
        config_svd = get_config(args, svd_unbiased.hyper_parameters)
        svd_predictor = svd_unbiased.SVDUnbiased(
            surprise_train_data, test_data, test_ids, args, config_svd, comet_logger
        )
        cache = "cache/svd/"
        models.append((svd_predictor, cache))

    if config["generate_nmf"]:
        config_nmf = get_config(args, nmf.hyper_parameters)
        nmf_predictor = nmf.NMF(
            surprise_train_data, test_data, test_ids, args, config_nmf, comet_logger
        )
        cache = "cache/nmf/"
        models.append((nmf_predictor, cache))

    if config["generate_slopeone"]:
        config_slopeone = get_config(args, {})
        slopeone_predictor = slopeone.SlopeOne(
            surprise_train_data, test_data, test_ids, args, config_slopeone, comet_logger
        )
        cache = "cache/slopeone/"
        models.append((slopeone_predictor, cache))

    if config["generate_svdpp"]:
        config_svdpp = get_config(args, svdpp.hyper_parameters)
        svdpp_predictor = svdpp.SVDpp(
            surprise_train_data, test_data, test_ids, args, config_svdpp, comet_logger
        )
        cache = "cache/svdpp/"
        models.append((svdpp_predictor, cache))

    # Predict Reinforcements and save them in corresponding files
    for model in models:
        predictor = model[0]
        cache = model[1]

        predictor.fit()
        train_reinforcement = pd.DataFrame({"Reinforcement": predictor.predict(train_data)})
        val_reinforcement = pd.DataFrame({"Reinforcement": predictor.predict(val_data)})
        test_reinforcement = pd.DataFrame({"Reinforcement": predictor.predict(test_data)})

        train_reinforcement.to_csv(cache + "train_reinforcement.csv", index=False)
        val_reinforcement.to_csv(cache + "val_reinforcement.csv", index=False)
        test_reinforcement.to_csv(cache + "test_reinforcement.csv", index=False)


if __name__ == "__main__":
    main()
