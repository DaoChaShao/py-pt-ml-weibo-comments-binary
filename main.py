#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 13:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from argparse import ArgumentParser

from pipeline import prepare_data


def main() -> None:
    """ Main Function """
    pass


"""
import argparse
from src.configs import trainer_config, dataset_config
from src.nets.unet.standard4 import Standard4LayersUNet
from src.datasets.unet.binary_dataset import BinarySegDataset
from src.trainers.unet.binary_trainer import UNetSegmentationTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=trainer_config.TRAINER_CONFIG["batch_size"])
parser.add_argument("--lr", type=float, default=trainer_config.TRAINER_CONFIG["learning_rate"])
parser.add_argument("--device", type=str, default=trainer_config.TRAINER_CONFIG["device"])
parser.add_argument("--model_save_path", type=str, default="./logs/unet_binary/best_model.pth")
args = parser.parse_args()
"""

if __name__ == "__main__":
    main()
