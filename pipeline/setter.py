#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 20:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   setter.py
# @Desc     :

from pipeline import process_data

from pipeline.preprocessor import process_data
from src.utils.helper import Timer


def prepare_data() -> None:
    """ Main Function """
    with Timer("Data Preparation"):
        # Get preprocessed data
        X_train_seqs, X_valid_seqs, y_train, y_valid = process_data()

        # TODO： Further data preparation steps can be added here


if __name__ == "__main__":
    prepare_data()
