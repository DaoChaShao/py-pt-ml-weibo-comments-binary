#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/30 22:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   evaluator.py
# @Desc     :   

from pandas import DataFrame, Series, read_csv
from pathlib import Path
from pprint import pprint
from random import randint
from torch import Tensor, load, no_grad, device

from src.configs.cfg_rnn import CONFIG4RNN
from src.dataloaders.base4pt import TorchDataLoader
from src.datasets.seq_classification import TorchDataset4Seq2Classification
from src.nets.rnn_lstm_classification import LSTMRNNForClassification
from src.utils.helper import Timer
from src.utils.highlighter import red, green
from src.utils.nlp import spacy_single_tokeniser, regular_chinese, build_word2id_seqs
from src.utils.stats import split_data, load_json
from src.utils.THU import cut_only


def main() -> None:
    """ Main Function """
    with Timer("Test Data Evaluation"):
        # Get test data
        params: Path = Path(CONFIG4RNN.FILEPATHS.SAVED_NET)
        path: Path = Path(CONFIG4RNN.FILEPATHS.DATA4ALL)
        dic: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY)
        key: Path = Path(CONFIG4RNN.FILEPATHS.API_KEY)
        # print(params)
        # print(path)
        # print(dic)
        # print(key)

        if params.exists() and path.exists() and dic.exists():
            # print("Data file found!")
            dictionary: dict = load_json(dic)
            # MAX_LEN: int = 32086  # Spacy Tokeniser
            MAX_LEN: int = 29163  # THULAC Tokeniser

            # Get raw data
            raw: DataFrame = read_csv(path)
            # print(raw.head())
            # print(raw.shape)

            # Separate features and labels
            X: Series = raw.iloc[:, -1]
            X: DataFrame = X.to_frame()
            y: DataFrame = raw.iloc[:, :-1]
            # print(X.head())
            # print(y.head())
            # print(X.shape, y.shape)
            # print(X.shape, y.shape)

            # Split data into training and validation sets
            _, _, X_raw, _, _, y_raw = split_data(X, y)

            # Tokenise text data
            # amount: int | None = 100
            amount: int | None = None
            X_contents: list[str] = X_raw.iloc[:, 0].tolist()
            if amount is not None:
                # lines: list[list[str]] = spacy_batch_tokeniser(X_contents[:amount], lang="zh")
                lines: list[list[str]] = [cut_only(line) for line in X_contents[:amount]]
            else:
                # lines: list[list[str]] = spacy_batch_tokeniser(X_raw.iloc[:, 0].tolist(), lang="zh")
                lines: list[list[str]] = [cut_only(line) for line in X_contents]
            # pprint(lines[:1])
            lines: list[list[str]] = [regular_chinese(words) for words in lines]
            # pprint(lines[:1])

            # Transform content to sequence of ids
            X_prove: list[list[int]] = build_word2id_seqs(lines, dictionary)
            # print(X_prove[:1])

            # Create Dataset
            dataset = TorchDataset4Seq2Classification(
                feature_seqs=X_prove,
                lbl_seqs=y_raw.values.tolist(),
                seq_max_len=MAX_LEN,
            )
            # Create DataLoader
            dataloader = TorchDataLoader(
                dataset=dataset,
                batch_size=CONFIG4RNN.PREPROCESSOR.BATCHES,
                is_shuffle=False,
            )
            idx: int = randint(0, len(dataloader) - 1)
            # print(f"Randomly selected batch index for evaluation: {idx}")
            # print(f"Total number of evaluation samples: {len(X_raw)}")
            # print(f"Features: {dataloader[idx][0]}")
            # print(f"Labels: {dataloader[idx][1]}")

            # Set the model
            model = LSTMRNNForClassification(
                vocab_size=len(dictionary),
                embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
                hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
                num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
                num_classes=CONFIG4RNN.PARAMETERS.CLASSES,
                dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT,
            )
            dict_state = load(CONFIG4RNN.FILEPATHS.SAVED_NET)
            model.load_state_dict(dict_state)
            model.to(device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
            model.eval()
            print("Model loaded successfully!")

            # Evaluation
            total: float = 0.0
            correct: float = 0.0
            for i, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.to(device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
                y_batch = y_batch.squeeze().to(device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
                # print(y_batch)

                # Start to predict
                with no_grad():
                    predictions: Tensor = model(X_batch)
                    # print(predictions)
                    y_predictions = predictions.argmax(dim=1)
                    # print(f"Predicted class: {y_predictions}")

                    correct += (y_predictions == y_batch).sum().item()
                    total += y_batch.size(0)

                    _acc = correct / total
                    print(
                        f"[{i:<05d}/{len(dataloader)}] Cumulative Accuracy: {green(f'{_acc:.4f}')} | failure: {red(f'{1.0 - _acc:.4f}')}"
                    )

            accuracy: float = correct / total
            failure: float = 1.0 - accuracy
            print(f"Final Accuracy: {green(f"{accuracy:.4f}")} | failure: {red(f"{failure:.4f}")}")
            """
            THULAC Tokeniser Results:
            - 
            """
        else:
            print(f"Sorry! {params.name}, {path.name} and {dic.name} do not exist. Please train the model first.")


if __name__ == "__main__":
    main()
