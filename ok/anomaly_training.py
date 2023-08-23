"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

import logging
import warnings
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything
from anomalib.models import get_model
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.mvtec import MVTec
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
from anomalib.data import MVTec
logger = logging.getLogger("anomalib")

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--category", type=str, required=True, help="Category of the dataset")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--task", type=str, default="segmentation", help="Task type: classification, detection, or segmentation")
    parser.add_argument("--log_level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser

def train(args: Namespace):
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    # Create an instance of the MVTec data module
    datamodule = MVTec(
        root=args.dataset_path,
        category=args.category,
        image_size=(256, 256),  # Specify the desired image size
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        task=args.task,
        transform_config_train=config.dataset.transform_config.train,  # Provide your transform configuration
        test_split_mode=config.dataset.test_split_mode,
        test_split_ratio=config.dataset.test_split_ratio,
        val_split_mode=config.dataset.val_split_mode,
        val_split_ratio=config.dataset.val_split_ratio,
    )

    # Instantiate your custom anomaly model
    model = get_model(config)  # Pass any necessary configuration

    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)

    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)  # Use your custom data module

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)

    logger.info("Testing the model.")
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)