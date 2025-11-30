#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/30 22:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   evaluator.py
# @Desc     :   

from pandas import DataFrame, Series, read_csv
from pathlib import Path
from torch import Tensor, load, no_grad, device
from tqdm import tqdm

from src.configs.cfg_rnn import CONFIG4RNN
from src.nets.rnn_lstm_classification import LSTMRNNForClassification
from src.utils.helper import Timer
from src.utils.highlighter import red, green
from src.utils.nlp import spacy_single_tokeniser, regular_chinese, build_word2id_seqs
from src.utils.PT import series2tensor, sequences2tensors
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
            acc: list[bool] = []
            for i in tqdm(range(len(X_raw)), total=len(X_raw), desc="Evaluating Samples"):
                X_content: Series = X_raw.iloc[i]
                y_prove: Series = y_raw.iloc[i]
                # print(X_content)
                # print(y_prove)

                # Tokenise the selected data
                # words: list[str] = spacy_single_tokeniser(X_content.squeeze(), lang="zh")
                words: list[str] = cut_only(X_content.squeeze())
                # print(words)
                tokens: list[str] = regular_chinese(words)
                # print(tokens)

                # Transform content to sequence of ids
                X_prove: list[list[int]] = build_word2id_seqs([tokens], dictionary)
                # print(X_prove)

                # Pad and convert to tensor
                X_prove: Tensor = sequences2tensors(X_prove, MAX_LEN).to(device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
                y_prove: Tensor = series2tensor(
                    [y_prove],
                    label=True,
                    accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR
                )
                # print(X_prove)

                # Start to predict
                with no_grad():
                    predictions: Tensor = model(X_prove)
                    # print(predictions)
                    y_pred: int = predictions.argmax(dim=1).item()
                    # print(f"Predicted class: {y_pred}")
                    acc.append(y_pred == y_prove.item())

            if acc:
                accuracy: float = sum(acc) / len(acc)
                failure: float = 1.0 - accuracy
                print(f"Evaluation completed! Accuracy: {green(f"{accuracy:.4f}")} | failure: {red(f"{failure:.4f}")}")
        else:
            print(f"Sorry! {params.name}, {path.name} and {dic.name} do not exist. Please train the model first.")


if __name__ == "__main__":
    main()
