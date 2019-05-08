# ==================================================================================
# Baseline Model
# date : 2019/05/05
# reference : https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch
# ==================================================================================

import gc
import os
import sys
import random
import time
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from pathlib import Path
from psutil import cpu_count
from functools import wraps, partial
from tqdm import tqdm

import numpy as np
import pandas as pd
from numba import jit

import librosa
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# ================= #
#  paramas section  #
# ================= #
# kaggleのkernelで実行する場合は以下
# IS_KERNEL = True
IS_KERNEL = False
VERSION = "0000" if IS_KERNEL else os.path.basename(__file__)[0:4]
ROOT_PATH = Path("..") if IS_KERNEL else Path(__file__).parents[1]
DataLoader = partial(DataLoader, nom_workers=cpu_count())
SEED = 1116

# 基礎数値他
SAMPLING_RATE = 44100  # 44.1[kHz]
SAMPLE_DURATION = 2  # 2[sec]
N_MEL = 128  # spectrogram y axis size


# =============== #
#  util function  #
# =============== #
def get_logger(is_torch=False):
    return getLogger("torch" + VERSION) if is_torch else getLogger(VERSION)


def stop_watch(*dargs, **dkargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            method_name = dargs[0]
            start = time.time()
            get_logger().info("[Start] {}".format(method_name))

            result = func(*args, **kargs)

            elapsed_time = int(time.time() - start)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            get_logger().info("[Finish] {}: [elapsed_time] {:0>2}:{:0>2}:{:0>2}".format(method_name, hour, minits, sec))
            return result
        return wrapper
    return decorator


# ======================= #
#  preprocessing section  #
# ======================= #
# ref : https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
# >> data select section
@stop_watch("select train data")
def select_train_data():
    input_dir = ROOT_PATH / "input"
    tag_list = pd.read_csv(input_dir / "sample_submission.csv").columns[1:].to_list()  # 80 tag list

    # train curated
    train_curated_df = pd.read_csv(input_dir / "train_curated.csv")
    train_curated_df["fpath"] = str(input_dir.absolute()) + "/train_curated/" + train_curated_df["fname"]

    # train noisy
    train_noisy_df = pd.read_csv(input_dir / "train_noisy.csv")
    single_tag_train_noisy_df = train_noisy_df[~train_noisy_df["labels"].str.contains(",")]
    train_noisy_df = None
    for tag in tag_list:  # 80 tags
        temp_df = single_tag_train_noisy_df.query("labels == '{}'".format(tag)).iloc[:50, :]
        if train_noisy_df is None:
            train_noisy_df = temp_df
        else:
            train_noisy_df = pd.concat([train_noisy_df, temp_df])
    train_noisy_df["fpath"] = str(input_dir.absolute()) + "/train_noisy/" + train_noisy_df["fname"]

    train_df = pd.concat([train_curated_df, train_noisy_df])[["fpath", "labels"]]
    return train_df
# << data select section


# >> audio convert section
def read_audio(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLING_RATE)

    # trim silence : https://librosa.github.io/librosa/generated/librosa.effects.trim.html
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)

    # padding short data
    sample_size = SAMPLE_DURATION * SAMPLING_RATE
    if len(y) < sample_size:
        padding = sample_size - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, sample_size - len(y) - offset), "constant")

    return y, sr  # np.ndarrya, shape=(sample_size,), SAMPLING_RATE


def audio_to_melspectrogram(audio, sr):
    spectrogram = librosa.feature.melspectrogram(
        audio,
        sr=sr,
        n_mels=N_MEL,  # https://librosa.github.io/librosa/generated/librosa.filters.mel.html#librosa.filters.mel
        hop_length=347 * SAMPLE_DURATION,  # to make time steps 128? 恐らくstftのロジックを理解すれば行ける
        n_fft=N_MEL * 20,  # n_mels * 20
        fmin=20,  # Filterbank lowest frequency, Audible range 20[Hz]
        fmax=sr / 2  # Nyquist frequency
    )
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def mono_to_color(mono):
    # stack X as [mono, mono, mono]
    x = np.stack([mono, mono, mono], axis=-1)

    # Standardize
    x_std = (x - x.mean()) / (x.std() + 1e-6)

    if (x_std.max() - x_std.min()) > 1e-6:
        color = 255 * (x_std - x_std.min()) / (x_std.max() - x_std.min())
        color = color.astype(np.uint8)
    else:
        color = np.zeros_like(x_std, dtype=np.uint8)

    return color
# << audio convert section


# >> dataset section(WIP)
def df_to_labeldata(df):
    """
    df : columns=["fpath", "labels"]
    """
    @jit
    def calc(arr):
        for idx in tqdm(range(len(arr))):
            y, sr = read_audio(arr[idx, 0])
            spec_mono = audio_to_melspectrogram(y, sr)
            spec_color = mono_to_color(spec_mono)
    calc(df.values)
"""
    for idx in tqdm(range(len(df))):
        y, sr = read_audio(df["fpath"].values[idx])
        spec_mono = audio_to_melspectrogram(y, sr)
        spec_color = mono_to_color(spec_mono)
"""


class TrainDataset(Dataset):
    """
    train_df :columns=["fpath", "labels"]
    """
    def __init__(self, train_df, transforms):
        super().__init__()
        self.transforms = transforms
        melspectrograms = []
        labels = []

    def __len__():
        pass

    def __getitem__():
        pass
# << dataset section


# ================ #
#  logger section  #
# ================ #
def create_logger():
    # formatter
    fmt = Formatter("[%(levelname)s] %(asctime)s >> \t%(message)s")

    # stream handler
    sh = StreamHandler()
    sh.setLevel(INFO)
    sh.setFormatter(fmt)

    # logger
    _logger = getLogger(VERSION)
    _logger.setLevel(DEBUG)
    _logger.addHandler(sh)

    _torch_logger = getLogger("torch" + VERSION)
    _torch_logger.setLevel(DEBUG)
    _torch_logger.addHandler(sh)

    # file output
    if not IS_KERNEL:
        log_dir = ROOT_PATH / "log"
        log_dir.mkdir(exist_ok=True, parents=True)

        _logfile = log_dir / "{}.log".format(VERSION)
        _logfile.touch()
        fh = FileHandler(_logfile)
        fh.setLevel(DEBUG)
        fh.setFormatter(fmt)
        _logger.addHandler(fh)

        _torch_logfile = log_dir / "{}_torch.log".format(VERSION)
        _torch_logfile.touch()
        torch_fh = FileHandler(_torch_logfile)
        torch_fh.setLevel(DEBUG)
        torch_fh.setFormatter("%(message)s")
        _torch_logger.addHandler(torch_fh)


# ===================== #
#  pre setting section  #
# ===================== #
def seed_everything():
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.daterministic = True


def jobs_manage(n_jobs=cpu_count()):
    os.environ["MKL_NUM_THREADS"] = str(n_jobs)
    os.environ["OMP_NUM_THREADS"] = str(n_jobs)


# ============== #
#  main section  #
# ============== #
@stop_watch("main function")
def main():
    # presetting
    seed_everything()
    jobs_manage()
    train_df = select_train_data()
    df_to_labeldata(train_df)


if __name__ == "__main__":
    gc.enable()
    create_logger()
    try:
        main()
    except Exception as e:
        print()
        get_logger().error("Exception Occured. \n>> \n {}".format(e))
