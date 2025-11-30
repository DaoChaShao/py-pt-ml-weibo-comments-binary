#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 13:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from argparse import ArgumentParser
from sys import argv
from torch import nn, optim

from pipeline import prepare_data
from src.configs.cfg_rnn import CONFIG4RNN
from src.trainers.rnn_seq_classification import TorchTrainer4Seq2Classification
from src.nets.rnn_lstm_classification import LSTMRNNForClassification
from src.utils.helper import Beautifier
from src.utils.PT import TorchRandomSeed


def main() -> None:
    """ Main Function """
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training times")
    args = parser.parse_args()
    """
    Display hyperparameters You need to set before training using command lines:
    1.Running: python main.py -h
    2.Or running: python main.py --help
    """

    with TorchRandomSeed("Weibo Seq Classification"):
        # Prepare data
        train_loader, valid_loader, dictionary = prepare_data()

        # Get the input size and number of classes
        vocab_size: int = len(dictionary)

        # Initialize model
        model = LSTMRNNForClassification(
            vocab_size=vocab_size,
            embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
            hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
            num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
            num_classes=CONFIG4RNN.PARAMETERS.CLASSES,
            dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT,
        )
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.alpha, weight_decay=CONFIG4RNN.HYPERPARAMETERS.DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss()
        model.summary()

        # Setup trainer
        trainer = TorchTrainer4Seq2Classification(
            model=model,
            optimiser=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
        )
        # Train the model
        trainer.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=args.epochs,
            model_save_path=str(CONFIG4RNN.FILEPATHS.SAVED_NET)
        )


if __name__ == "__main__":
    main()
