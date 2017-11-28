#!/usr/bin/env python
# -*- coding: utf-8 -*-
# by vellhe 2017/11/23
import keras

from audio_processor import load_xy
from audio_tagger_cnn import AudioTaggerCNN
from audio_tagger_crnn import AudioTaggerCRNN

TRAIN_DATASET_PATH = "dataset/audio_tagger_accurate_train_data.hdf5"
TEST_DATASET_PATH = "dataset/audio_tagger_accurate_test_data.hdf5"
CRNN_MODEL_WEIGHTS_PATH = "out/crnn_weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"
CNN_MODEL_WEIGHTS_PATH = "out/cnn_weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"


def train_model(model_type="crnn"):
    if model_type == "cnn":
        model = AudioTaggerCNN()
        model_weigths_out_path = CNN_MODEL_WEIGHTS_PATH
    elif model_type == "crnn":
        model = AudioTaggerCRNN()
        model_weigths_out_path = CRNN_MODEL_WEIGHTS_PATH
    else:
        print("error model_type", model_type)
        return None

    x_train, y_train = load_xy(TRAIN_DATASET_PATH)
    x_test, y_test = load_xy(TEST_DATASET_PATH)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, 20, 1000, validation_data=(x_test, y_test), callbacks=[
        keras.callbacks.ModelCheckpoint(model_weigths_out_path,
                                        monitor="val_acc", verbose=1,
                                        save_best_only=True, save_weights_only=True)])


if __name__ == "__main__":
    train_model("crnn")
