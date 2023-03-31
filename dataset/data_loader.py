import logging
import os
import pickle
import random
import cv2

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from .avm_dataset import AVMDataset
from .offset_dataset import OffsetDataset
from .basis_dataset import HomoTrainData, HomoTestData

_logger = logging.getLogger(__name__)


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    _logger.info(
        "Dataset type: {}, transform type: {}".format(
            params.dataset_type, params.transform_type
        )
    )

    if params.dataset_type == "basic":
        train_ds = HomoTrainData(params)
        test_ds = HomoTestData(params)
    elif params.dataset_type == "avm":
        train_ds = AVMDataset(params.train_data_dir)
        test_ds = AVMDataset(params.test_data_dir)
    elif params.dataset_type == "offset":
        train_ds = OffsetDataset(params.train_data_dir)
        test_ds = OffsetDataset(params.test_data_dir)

    dataloaders = {}
    # add train data loader
    train_dl = DataLoader(
        train_ds,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
        drop_last=True,
        # prefetch_factor=3, # for pytorch >=1.5.0
    )
    dataloaders["train"] = train_dl

    # chose test data loader for evaluate

    if params.eval_type in ["test", 'valid']:
        dl = DataLoader(
            test_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.cuda
            # prefetch_factor=3, # for pytorch >=1.5.0
        )
    else:
        dl = None
        raise ValueError("Unknown eval_type in params, should in [val, test]")

    dataloaders[params.eval_type] = dl

    return dataloaders
