# ==================================================================================
# Dataset Maker
# date : 2019/05/18
# reference : https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch
# ==================================================================================

import gc
import os
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

# ================= #
#  paramas section  #
# ================= #
# kaggleのkernelで実行する場合は以下
# IS_KERNEL = True
IS_KERNEL = False
VERSION = "0000" if IS_KERNEL else os.path.basename(__file__)[0:4]
ROOT_PATH = Path("..") if IS_KERNEL else Path(__file__).parents[1]
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
def path_loading():
    input_dir = ROOT_PATH / "input"
    save_dir = ROOT_PATH if IS_KERNEL else ROOT_PATH / "image" / VERSION

    # make dir
    (save_dir / "train_curated").mkdir(parents=True, exist_ok=True)
    (save_dir / "train_noisy").mkdir(parents=True, exist_ok=True)
    (save_dir / "test").mkdir(parents=True, exist_ok=True)

    # train curated
    train_curated_df = pd.read_csv(input_dir / "train_curated.csv")
    train_curated_df["audio_path"] = str(input_dir.absolute()) + "/train_curated/" + train_curated_df["fname"]
    train_curated_df["image_path"] = str(save_dir.absolute()) + "/train_curated/" + train_curated_df["fname"].str[:-4] + ".png"
    train_curated_df = train_curated_df[["audio_path", "image_path"]]

    # train noisy
    train_noisy_df = pd.read_csv(input_dir / "train_noisy.csv")
    train_noisy_df["audio_path"] = str(input_dir.absolute()) + "/train_noisy/" + train_noisy_df["fname"]
    train_noisy_df["image_path"] = str(save_dir.absolute()) + "/train_noisy/" + train_noisy_df["fname"].str[:-4] + ".png"
    train_noisy_df = train_noisy_df[["audio_path", "image_path"]]

    # test
    test_fname = [p.name for p in list((input_dir / "test").glob("*.wav"))]
    test_df = pd.DataFrame(data=list(test_fname), columns=["fname"])
    test_df["audio_path"] = str(input_dir.absolute()) + "/test/" + test_df["fname"]
    test_df["image_path"] = str(save_dir.absolute()) + "/test/" + test_df["fname"].str[:-4] + ".png"
    test_df = test_df[["audio_path", "image_path"]]

    df = pd.concat([train_curated_df, train_noisy_df, test_df])[["audio_path", "image_path"]]
    df.reset_index(drop=True, inplace=True)
    return df
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

        # file handler
        _logfile = log_dir / "{}.log".format(VERSION)
        _logfile.touch()
        fh = FileHandler(_logfile, mode="w")
        fh.setLevel(DEBUG)
        fh.setFormatter(fmt)
        _logger.addHandler(fh)

        _torch_logfile = log_dir / "{}_torch.log".format(VERSION)
        _torch_logfile.touch()
        torch_fh = FileHandler(_torch_logfile, mode="w")
        torch_fh.setLevel(DEBUG)
        torch_fh.setFormatter(Formatter("%(message)s"))
        _torch_logger.addHandler(torch_fh)


# ===================== #
#  pre setting section  #
# ===================== #
def seed_everything():
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)


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

    # preprocessing
    path_df = path_loading()

    for idx in tqdm(range(len(path_df))):
        y, sr = read_audio(path_df.loc[idx, "audio_path"])
        mono_spec = audio_to_melspectrogram(y, sr)
        color = mono_to_color(mono_spec)
        Image.fromarray(color).save(path_df.loc[idx, "image_path"])


if __name__ == "__main__":
    gc.enable()
    create_logger()
    try:
        main()
    except Exception as e:
        get_logger().error("Exception Occured. \n>> \n {}".format(e))
        raise e
