#!/usr/bin/env python
# -*- coding: utf-8 -*-
# by vellhe 2017/11/23
import keras

from audio_processor import audios_to_x, tags
from audio_tagger_cnn import AudioTaggerCNN
from audio_tagger_crnn import AudioTaggerCRNN

CRNN_WEIGHTS_PATH = "model_weights/crnn_weights.108-0.39-0.90.hdf5"
CNN_WEIGHTS_PATH = "model_weights/cnn_weights.07-0.43-0.86.hdf5"


def predict(audio_paths, model_type="crnn"):
    if model_type == "cnn":
        model = AudioTaggerCNN(CNN_WEIGHTS_PATH)
    elif model_type == "crnn":
        model = AudioTaggerCRNN(CRNN_WEIGHTS_PATH)
    else:
        print("error model_type", model_type)
        return None
    x = audios_to_x(audio_paths)
    y_pre = model.predict(x)
    for ret in y_pre:
        print(tags[ret.argmax()])


if __name__ == "__main__":
    predict(["data/Z999@1050615.wav", "data/f001.wav", "data/m001.wav", "data/c001.wav"],
            "crnn")
    predict(["data/Z999@1050615.wav", "data/f001.wav", "data/m001.wav", "data/c001.wav"],
            "cnn")
