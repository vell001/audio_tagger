#!/usr/bin/env python
# -*- coding: utf-8 -*-
# by vellhe 2017/11/23
import keras

from audio_processor import load_train_data, load_xy
from audio_tagger_cnn import AudioTaggerCNN

if __name__ == "__main__":
    tags = ['c', 'm', 'f']
    # x_train, y_train = load_train_data("/home/vell/workspace/audio_tagger_data/性别分类/accurate_train_data", tags)
    # x_test, y_test = load_train_data("/home/vell/workspace/audio_tagger_data/性别分类/accurate_test_data", tags)
    x_train, y_train = load_xy("/home/vell/workspace/audio_tagger/data/audio_tagger_accurate_train_data.hdf5")
    x_test, y_test = load_xy("/home/vell/workspace/audio_tagger/data/audio_tagger_accurate_test_data.hdf5")
    model = AudioTaggerCNN(None)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, 10, 100, validation_data=(x_test, y_test), callbacks=[
        keras.callbacks.ModelCheckpoint("out/cnn_weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", monitor="val_acc", verbose=1,
                                        save_best_only=True, save_weights_only=True)])
    model.save("model_test.h5py")
