"""GNN NCF main.

This modules trains and tests the GNN NCF model.
"""

import numpy as np
from comet_ml import Experiment
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from modules import gnn_ncf
from utilities.data_preparation import create_dataset, create_laplacian_matrix, load_data
from utilities.helper import create_argument_parser, create_comet_logger, get_config, get_hash


def main():
    """Run GNN NCF model."""
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, gnn_ncf.hyper_parameters)

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(config)

    train_pd, val_pd = load_data(
        file_path=args.data_dir + args.train_data,
        train_val_split=True,
        random_seed=args.random_seed,
        train_size=config["train_size"],
    )
    test_pd = load_data(file_path=args.data_dir + args.test_data, train_val_split=False)
    train_data, val_data = create_dataset(train_pd), create_dataset(val_pd)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)

    laplacian_matrix = create_laplacian_matrix(
        train_pd, config["number_of_users"], config["number_of_movies"]
    )

    graph_neural_network = gnn_ncf.GNN(
        train_data, val_data, test_data, test_ids, args, laplacian_matrix, config
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=config["patience"])

    checkpoint_filename = "gnn_ncf_" + str(get_hash(config, args))
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

    best_graph_neural_network = gnn_ncf.GNN.load_from_checkpoint(
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
