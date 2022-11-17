"""Reinforced GNN NCF main.

This modules trains and tests the reinforced GNN NCF model.
"""

import numpy as np
from comet_ml import Experiment
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from modules import reinforced_gnn_ncf
from utilities.data_preparation import (
    create_dataset_with_reinforcements,
    create_laplacian_matrix,
    load_data,
    load_reinforcements,
)
from utilities.helper import (
    check_caches_exist,
    create_argument_parser,
    create_checkpoint_directory,
    create_comet_logger,
    create_ensemble_learning_directory,
    get_config,
    get_hash,
)


def main():
    """Run reinforced GNN NCF model."""
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, reinforced_gnn_ncf.hyper_parameters)

    check_caches_exist(config["reinforcement_type"])
    create_checkpoint_directory()
    if args.ensemble_learning:
        create_ensemble_learning_directory("reinforced_gnn_ncf")

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(config)

    train_pd = load_data(file_path="cache/train_data.csv", train_val_split=False)
    val_pd = load_data(file_path="cache/val_data.csv", train_val_split=False)
    test_pd = load_data(file_path="cache/test_data.csv", train_val_split=False)

    reinforcement_cache = None
    train_reinforcements, val_reinforcements, test_reinforcements = None, None, None

    for reinforcement_type in config["reinforcement_type"]:
        reinforcement_cache = "cache/" + reinforcement_type + "/"
        if train_reinforcements is None:
            train_reinforcements = load_reinforcements(
                reinforcement_cache + "train_reinforcement.csv"
            )
            val_reinforcements = load_reinforcements(reinforcement_cache + "val_reinforcement.csv")
            test_reinforcements = load_reinforcements(
                reinforcement_cache + "test_reinforcement.csv"
            )
        else:
            cur_train_reinforcements = load_reinforcements(
                reinforcement_cache + "train_reinforcement.csv"
            )
            cur_val_reinforcements = load_reinforcements(
                reinforcement_cache + "val_reinforcement.csv"
            )
            cur_test_reinforcements = load_reinforcements(
                reinforcement_cache + "test_reinforcement.csv"
            )

            train_reinforcements = np.concatenate(
                (train_reinforcements, cur_train_reinforcements), axis=1
            )
            val_reinforcements = np.concatenate(
                (val_reinforcements, cur_val_reinforcements), axis=1
            )
            test_reinforcements = np.concatenate(
                (test_reinforcements, cur_test_reinforcements), axis=1
            )

    train_data = create_dataset_with_reinforcements(train_pd, train_reinforcements)
    val_data = create_dataset_with_reinforcements(val_pd, val_reinforcements)
    test_ids, test_data = create_dataset_with_reinforcements(
        test_pd, test_reinforcements, test_dataset=True
    )

    laplacian_matrix = create_laplacian_matrix(
        train_pd, config["number_of_users"], config["number_of_movies"]
    )

    graph_neural_network = reinforced_gnn_ncf.GNN(
        train_data, val_data, test_data, test_ids, args, laplacian_matrix, config
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=config["patience"])

    checkpoint_filename = "reinforced_gnn_ncf_" + str(get_hash(config, args))
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=checkpoint_filename,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=config["num_epochs"],
        logger=comet_logger,
        callbacks=[early_stopping, checkpoint_callback],
    )

    trainer.fit(graph_neural_network)

    best_graph_neural_network = reinforced_gnn_ncf.GNN.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        test_ids=test_ids,
        args=args,
        laplacian_matrix=laplacian_matrix,
        config=config,
    )

    trainer.test(best_graph_neural_network)


if __name__ == "__main__":
    main()
