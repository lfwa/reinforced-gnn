"""Helper functionality.

This module provides helper functions for creating an argument parser,
retrieving config, creating directories needed, loggers, etc.
"""

import copy
import hashlib
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from pytorch_lightning.loggers import CometLogger


def create_argument_parser() -> ArgumentParser:
    """Create argument parser.

    Returns:
        ArgumentParser: Argument parser.
    """
    parser = ArgumentParser(
        description="Main entry point for this project",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../data/",
        help="path to the directory containing the unprocessed data",
    )
    parser.add_argument(
        "--train-data", type=str, default="data_train.csv", help="name of the training data file"
    )
    parser.add_argument(
        "--test-data", type=str, default="data_test.csv", help="name of the testing data file"
    )
    parser.add_argument("--random-seed", type=int, default=7, help="random seed used")
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="flag indicating whether the experiment is logged in comet ml",
    )
    parser.add_argument(
        "--comet-key",
        type=str,
        default="../../comet.json",
        help="path to the comet api key directory",
    )
    parser.add_argument(
        "--comet-directory",
        type=str,
        default="./logs",
        help="path to log directory when comet can not be run online",
    )
    parser.add_argument(
        "--dataloader-workers", type=int, default=8, help="number of dataloader workers used"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="path to non-default config for testing"
    )
    parser.add_argument(
        "--ensemble-learning",
        action="store_true",
        help="flag indicating whether ensemble learning is enables",
    )
    parser.add_argument(
        "--ensemble-directory",
        type=str,
        default=None,
        help="path to the directory containing the predictions from the ensemble",
    )
    return parser


def get_config(args: Namespace, hyper_parameters):
    """Retrieve config.

    Args:
        args (Namespace): Args namespace.
        hyper_parameters: Hyper parameters.

    Returns:
        Config.
    """
    config = copy.deepcopy(hyper_parameters)

    if args.config is not None:
        try:
            new_config = json.load(open(args.config))

            for key in new_config.keys():
                config[key] = new_config[key]

        except:
            print("New config not found ... Continue with default config of model ...")
    return config


def create_comet_logger(args: Namespace) -> CometLogger:
    """Create a CometLogger instance used to log experiments.

    Args:
        args (Namespace): json with comet configurations.

    Returns:
        CometLogger: CometLogger used to log the experiment.
    """
    comet_api_key = None
    try:
        comet_api_key = json.load(open(args.comet_key))
    except:
        print("Comet API Key not found ... Continue by logging the experiment offline ...")

    if comet_api_key is None:
        return CometLogger(save_dir=args.comet_directory)
    else:
        return CometLogger(
            api_key=comet_api_key["api_key"],
            project_name=comet_api_key["project_name"],
            workspace=comet_api_key["workspace"],
            disabled=args.disable_logging,
            offline=False,
            save_dir="/cluster/scratch/" + comet_api_key["workspace"],
        )


def create_cache_directories(config):
    """Create directories for cache.

    Args:
        config: Configuration.
    """
    if not os.path.exists("cache"):
        os.mkdir("cache")

    if config["generate_svd"] and not os.path.exists("cache/svd"):
        os.mkdir("cache/svd")

    if config["generate_nmf"] and not os.path.exists("cache/nmf"):
        os.mkdir("cache/nmf")

    if config["generate_slopeone"] and not os.path.exists("cache/slopeone"):
        os.mkdir("cache/slopeone")

    if config["generate_svdpp"] and not os.path.exists("cache/svdpp"):
        os.mkdir("cache/svdpp")


def check_caches_exist(reinforcement_types):
    """Check if cache exists.

    Args:
        reinforcement_types: Reinforcement types used.
    """
    if not os.path.exists("cache"):
        print(
            "Could not find the cache directory. Please generate it using the "
            "reinfocement generator first ..."
        )
        exit()

    for reinforcement_type in reinforcement_types:
        if not os.path.exists("cache/" + reinforcement_type):
            print(
                "Could not find the cache directory for "
                + reinforcement_type
                + ". Please generate it using the reinfocement generator first ..."
            )
            exit()


def create_checkpoint_directory():
    """Create directory used to save model checkpoints."""
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")


def create_ensemble_learning_directory(model_name):
    """Create directory for storing ensemble results.

    Args:
        model_name (str): Name of model.
    """
    if not os.path.exists("ensemble"):
        os.mkdir("ensemble")

    if not os.path.exists("ensemble/" + model_name):
        os.mkdir("ensemble/" + model_name)


def get_hash(config, args):
    """Retrieve hash.

    Args:
        config: Configuration.
        args: Arguments.

    Returns:
        str: Hash.
    """
    encoded_config = json.dumps(config, sort_keys=True).encode()
    encoded_args = json.dumps(vars(args), sort_keys=True).encode()

    dhash = hashlib.md5()
    dhash.update(encoded_config)
    dhash.update(encoded_args)
    return dhash.hexdigest()
