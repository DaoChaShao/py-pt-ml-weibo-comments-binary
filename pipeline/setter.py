#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 20:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   setter.py
# @Desc     :

from random import randint

from pipeline.preprocessor import process_data
from src.configs.cfg_rnn import CONFIG4RNN
from src.dataloaders.base4pt import TorchDataLoader
from src.datasets.seq_classification import TorchDataset4Seq2Classification
from src.utils.helper import Timer


def prepare_data() -> tuple[TorchDataLoader, TorchDataLoader, dict]:
    """ Main Function """
    with Timer("Data Preparation"):
        # Get preprocessed data
        X_train, X_valid, y_train, y_valid, dictionary, max_len = process_data()

        # Create PyTorch datasets
        train_dataset = TorchDataset4Seq2Classification(X_train, y_train.values.tolist(), max_len)
        valid_dataset = TorchDataset4Seq2Classification(X_valid, y_valid.values.tolist(), max_len)
        # idx_train: int = randint(0, len(train_dataset) - 1)
        # print(f"Train Sample at index {idx_train:>05d}: {train_dataset[idx_train][1]} | {train_dataset[idx_train][0]}")
        # idx_valid: int = randint(0, len(valid_dataset) - 1)
        # print(f"Valid Sample at index {idx_valid:>05d}: {valid_dataset[idx_valid][1]} | {valid_dataset[idx_valid][0]}")

        # Create PyTorch data loaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=CONFIG4RNN.PREPROCESSOR.BATCHES,
            is_shuffle=CONFIG4RNN.PREPROCESSOR.SHUFFLE
        )
        valid_loader = TorchDataLoader(
            valid_dataset,
            batch_size=CONFIG4RNN.PREPROCESSOR.BATCHES,
            is_shuffle=CONFIG4RNN.PREPROCESSOR.SHUFFLE
        )
        # batch_train = next(iter(train_loader))
        # print(f"Train Batch - Features: {batch_train[0].shape}, Labels: {batch_train[1].shape}")
        # batch_valid = next(iter(valid_loader))
        # print(f"Valid Batch - Features: {batch_valid[0].shape}, Labels: {batch_valid[1].shape}")

        return train_loader, valid_loader, dictionary


if __name__ == "__main__":
    prepare_data()
