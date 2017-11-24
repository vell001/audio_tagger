import codecs

import h5py
import keras
import librosa
import numpy as np
import os


def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

    '''

    # mel-spectrogram parameters
    SR = 16000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 21.85  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) / 2:(n_sample + n_sample_fit) / 2]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS) ** 2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


def audio_to_melgram_dataset(audio_path_tag):
    x = np.zeros((0, 1, 96, 1366))
    y = np.zeros((0, 3))
    for audio_path, tag in audio_path_tag:
        melgram = compute_melgram(audio_path)
        x = np.concatenate((x, melgram), axis=0)
        tag_cat = keras.utils.to_categorical(tag, 3)
        y = np.concatenate((y, tag_cat), axis=0)

    y = np.array(y)
    return x, y


def load_train_data(audio_manifest_path, tags):
    audio_path_tag = list()
    with codecs.open(audio_manifest_path, "r", "utf-8") as audio_manifest_f:
        for line in audio_manifest_f.readlines():
            if not line or not line.strip():
                continue
            tmp = line.split("  ")
            if not tmp or len(tmp) != 2:
                continue
            audio_path = os.path.join(os.path.dirname(audio_manifest_path), tmp[0].strip())
            if not os.path.exists(audio_path):
                continue

            tag = tags.index(tmp[1].strip())
            if tag < 0:
                continue
            audio_path_tag.append((audio_path, tag))
    return audio_to_melgram_dataset(audio_path_tag)


if __name__ == "__main__":
    # print(compute_melgram("data/Z999@1050615.wav"))
    # audio_to_melgram_dataset([
    #     ("data/Z999@1050615.wav", 1),
    #     ("data/Z999@1050615.wav", 1),
    #     ("data/Z999@1050615.wav", 1),
    #     ("data/Z999@1050615.wav", 1),
    # ])
    tags = ['c', 'm', 'f']
    load_train_data("/home/vell/workspace/audio_tagger_data/性别分类/accurate_test_data", tags)
