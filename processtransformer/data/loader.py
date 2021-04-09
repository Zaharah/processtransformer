import io
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing 

from ..constants import Task

class LogsDataLoader:
    def __init__(self, name, dir_path = "./datasets"):
        """Provides support for reading and 
            pre-processing examples from processed logs.
        Args:
            name: str: name of the dataset as used during processing raw logs
            dir_path: str: Path to dataset directory
        """
        self._dir_path = f"{dir_path}/{name}/processed"

    def prepare_data_next_activity(self, df, 
        x_word_dict, y_word_dict, 
        max_case_length, shuffle=True):
        
        x = df["prefix"].values
        y = df["next_act"].values
        if shuffle:
            x, y = utils.shuffle(x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        # token_x = np.array(token_x, dtype = np.float32)

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])
        # token_y = np.array(token_y, dtype = np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)

        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)

        return token_x, token_y

    def prepare_data_next_time(self, df, 
        x_word_dict,  max_case_length, 
        time_scaler = None, y_scaler = None, 
        shuffle = True):

        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", 
            "time_passed"]].values.astype(np.float32)
        y = df["next_time"].values.astype(np.float32)
        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(
                time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(
                time_x).astype(np.float32)            

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(
                y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(
                y.reshape(-1, 1)).astype(np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)
        
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return token_x, time_x, y, time_scaler, y_scaler

    def prepare_data_remaining_time(self, df, x_word_dict, max_case_length, 
        time_scaler = None, y_scaler = None, shuffle = True):

        x = df["prefix"].values
        time_x = df[["recent_time",	"latest_time", 
            "time_passed"]].values.astype(np.float32)
        y = df["remaining_time_days"].values.astype(np.float32)

        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(
                time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(
                time_x).astype(np.float32)            

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(
                y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(
                y.reshape(-1, 1)).astype(np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)
        
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        return token_x, time_x, y, time_scaler, y_scaler

    def get_max_case_length(self, train_x):
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x)

    def load_data(self, task):
        if task not in (Task.NEXT_ACTIVITY,
            Task.NEXT_TIME,
            Task.REMAINING_TIME):
            raise ValueError("Invalid task.")

        train_df = pd.read_csv(f"{self._dir_path}/{task.value}_train.csv")
        test_df = pd.read_csv(f"{self._dir_path}/{task.value}_test.csv")

        with open(f"{self._dir_path}/metadata.json", "r") as json_file:
            metadata = json.load(json_file)

        x_word_dict = metadata["x_word_dict"]
        y_word_dict = metadata["y_word_dict"]
        max_case_length = self.get_max_case_length(train_df["prefix"].values)
        vocab_size = len(x_word_dict) 
        total_classes = len(y_word_dict)

        return (train_df, test_df, 
            x_word_dict, y_word_dict, 
            max_case_length, vocab_size, 
            total_classes)
