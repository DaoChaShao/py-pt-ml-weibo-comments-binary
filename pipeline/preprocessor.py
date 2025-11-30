#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 20:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :   

from pathlib import Path
from pandas import DataFrame, Series, read_csv
from pprint import pprint

from src.configs.cfg_rnn import CONFIG4RNN
from src.utils.helper import Timer
from src.utils.highlighter import starts
from src.utils.nlp import spacy_batch_tokeniser, regular_chinese, count_frequency, build_word2id_seqs
from src.utils.stats import split_data, save_json
from src.utils.THU import cut_only


def process_data() -> tuple[list[list[int]], list[list[int]], DataFrame, DataFrame, dict, int]:
    """ Main Function """
    with Timer("Data Preprocessing"):
        path: Path = Path(CONFIG4RNN.FILEPATHS.DATA4ALL)
        # print(path)

        if path.exists():
            # print("Data file found!")

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
            X_train, X_valid, _, y_train, y_valid, _ = split_data(X, y)
            # print(X_train.head())
            # print(y_train.head())

            # Tokenise text data
            # amount: int | None = 300
            amount: int | None = None
            X_train_contents: list[str] = X_train.iloc[:, 0].tolist()
            X_valid_contents: list[str] = X_valid.iloc[:, 0].tolist()
            if amount is not None:
                # lines_train: list[list[str]] = spacy_batch_tokeniser(X_train_contents[:amount], lang="zh")
                # lines_valid: list[list[str]] = spacy_batch_tokeniser(X_valid_contents[:amount], lang="zh")
                lines_train: list[list[str]] = [cut_only(line) for line in X_train_contents[:amount]]
                lines_valid: list[list[str]] = [cut_only(line) for line in X_valid_contents[:amount]]
            else:
                # lines_train: list[list[str]] = spacy_batch_tokeniser(X_train.iloc[:, 0].tolist(), lang="zh")
                # lines_valid: list[list[str]] = spacy_batch_tokeniser(X_valid.iloc[:, 0].tolist(), lang="zh")
                lines_train: list[list[str]] = [cut_only(line) for line in X_train_contents]
                lines_valid: list[list[str]] = [cut_only(line) for line in X_valid_contents]
            # pprint(lines_train[:5])
            lines_train: list[list[str]] = [regular_chinese(words) for words in lines_train]
            # pprint(lines_train[:5])
            # print(len(lines_train))

            # Count word frequencies
            if amount is not None:
                tokens: list[str] = [word for line in lines_train[:amount] for word in line]
                freq, _ = count_frequency(tokens, top_k=10, freq_threshold=3)
            else:
                tokens: list[str] = [word for line in lines_train for word in line]
                freq, _ = count_frequency(tokens, top_k=10, freq_threshold=3)

            # Create a dictionary/word2id mapping words to indices
            special: list[str] = ["<PAD>", "<UNK>"]
            dictionary: dict[str, int] = {word: idx for idx, word in enumerate(special + freq)}
            # print(dictionary)
            # print(len(dictionary))
            save_json(dictionary, CONFIG4RNN.FILEPATHS.DICTIONARY)
            # print(f"The size of saved dictionary is {len(dictionary)}!")

            # Build 2D index representation of texts
            sequences: list[list[int]] = build_word2id_seqs(lines_train, dictionary)
            # print(sequences)
            # print(len(sequences))

            # Padding the sequences to a fixed length
            lengths: list[int] = [len(seq) for seq in sequences]
            max_len: int = max(lengths)
            min_len: int = min(lengths)
            avg_len: float = sum(lengths) / len(lengths)
            # print(f"Max Length: {max_len}, Min Length: {min_len}, Avg Length: {avg_len:.2f}")

            # Setup features and labels
            if amount is not None:
                X_train = build_word2id_seqs(lines_train[:amount], dictionary)
                X_valid = build_word2id_seqs(lines_valid[:amount], dictionary)
                y_train = y_train[:amount]
                y_valid = y_valid[:amount]
            else:
                X_train = build_word2id_seqs(lines_train, dictionary)
                X_valid = build_word2id_seqs(lines_valid, dictionary)

            starts("Data Preprocessing Summary")
            print(f"Train dataset: {len(X_train)} Samples")
            print(f"Valid dataset: {len(X_valid)} Samples")
            print(f"Dictionary Size: {len(dictionary)}")
            print(f"The min length of the sequence: {min_len}")
            print(f"The average length of the sequence: {avg_len:.1f}")
            print(f"The max length of the sequence: {max_len}")
            print(f"Train Labels: {y_train.values.sum()} poses, {len(y_train) - y_train.values.sum()} negs")
            print(f"Valid Labels: {y_valid.values.sum()} poses, {len(y_valid) - y_valid.values.sum()} negs")
            starts()
            print()
            """
            Spacy
            *******************Data Preprocessing Summary*******************
            Train dataset: 83991 Samples
            Valid dataset: 17998 Samples
            Dictionary Size: 32086
            The min length of the sequence: 0
            The average length of the sequence: 14.1
            The max length of the sequence: 58
            Train Labels: 42014 poses, 41977 negs
            Valid Labels: 9040 poses, 8958 negs
            ****************************************************************
            THULAC
            *******************Data Preprocessing Summary*******************
            Train dataset: 83991 Samples
            Valid dataset: 17998 Samples
            Dictionary Size: 29163
            The min length of the sequence: 1
            The average length of the sequence: 28.7
            The max length of the sequence: 111
            Train Labels: 42014 poses, 41977 negs
            Valid Labels: 9040 poses, 8958 negs
            ****************************************************************
            """
        else:
            raise FileNotFoundError(f"Data file {path.name} NOT found!")

        return X_train, X_valid, y_train, y_valid, dictionary, max_len


if __name__ == "__main__":
    process_data()
