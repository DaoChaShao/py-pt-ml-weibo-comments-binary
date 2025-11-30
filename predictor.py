#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 13:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from pandas import DataFrame, Series, read_csv
from pathlib import Path
from random import randint
from torch import Tensor, load, no_grad, device

from src.configs.cfg_rnn import CONFIG4RNN
from src.nets.rnn_lstm_classification import LSTMRNNForClassification
from src.utils.apis import OpenAITextCompleter
from src.utils.helper import Timer, read_yaml
from src.utils.highlighter import red, green, blue
from src.utils.nlp import spacy_single_tokeniser, regular_chinese, build_word2id_seqs
from src.utils.PT import series2tensor, sequences2tensors
from src.utils.stats import split_data, load_json
from src.utils.THU import cut_only


def main() -> None:
    """ Main Function """
    with Timer("Test Data Inference"):
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

            # Select a piece of data
            idx: int = randint(0, len(X_raw) - 1)
            X_content: Series = X_raw.iloc[idx]
            y_prove: Series = y_raw.iloc[idx]
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
            y_prove: Tensor = series2tensor([y_prove], label=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)
            # print(X_prove)

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

            # Start to predict
            with no_grad():
                predictions: Tensor = model(X_prove)
                # print(predictions)
                y_pred: int = predictions.argmax(dim=1).item()
                # print(f"Predicted class: {y_pred}")

                rate: str = green("Positive") if y_prove.item() == 1 else red("Negative")
                print(f"The {blue(idx)} comment is {rate}.")
                print(X_content)

                # Prompt Engineering with OpenAI API
                cfg: dict = read_yaml(key)
                API_KEY: str = cfg["openai"]["api_key"]
                opener = OpenAITextCompleter(API_KEY)
                role: str = "You are a professional Chinese Language Expert with Chinese expertise."
                prompt: str = f"""
                    Give a brief explanation in Chinese why the following review is {'positive' if y_pred == 1 else 'negative'}:
                    {X_content}
                    Explanation:
                    """
                explanation = opener.client(role, prompt)
                print(explanation)

                correct: bool = (y_pred == y_prove.item())
                result = green("Bingo") if correct else red("Shit")
                print(f"The prediction: {result} !")
                print()
        else:
            print(f"Sorry! {params.name}, {path.name} and {dic.name} do not exist. Please train the model first.")


if __name__ == "__main__":
    main()
