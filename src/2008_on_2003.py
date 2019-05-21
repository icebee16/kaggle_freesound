# ==================================================================================
# Baseline Model
# date : 2019/05/19
# reference: https://www.kaggle.com/peining/simple-cnn-classifier-with-pytorch
# ==================================================================================

import gc
import os
import random
import time
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from pathlib import Path
from psutil import cpu_count
from functools import wraps, partial
from fastprogress import master_bar, progress_bar

import numpy as np
import pandas as pd
from numba import jit

import librosa
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from imgaug import augmenters as iaa


# ================= #
#  paramas section  #
# ================= #
IS_KERNEL = False
VERSION = "0000" if IS_KERNEL else os.path.basename(__file__)[0:4]
ROOT_PATH = Path("..") if IS_KERNEL else Path(__file__).parents[1]
DataLoader = partial(DataLoader, num_workers=cpu_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Const
SEED = 1129
SAMPLING_RATE = 44100  # 44.1[kHz]
SAMPLE_DURATION = 2  # 2[sec]
HOP_LENGTH = 345
N_MEL = 128  # spectrogram y axis size
SPEC_AUGMENTATION_RATE = 2

# SPEC_AUGMENTATION
NUM_MASK = 2
FREQ_MASKING_MAX_PERCENTAGE = 0.15
TIME_MASKING_MAX_PERCENTAGE = 0.30


# Directory
DEBUG_MODE = True
HEAD = "debug_" if DEBUG_MODE else ""
CURATED_DIR = HEAD + "train_curated"
NOISY_DIR = HEAD + "train_noisy"
TEST_DIR = HEAD + "test"
SAMPLE_SUBMISSION = HEAD + "sample_submission"

# Training params
num_epochs = 80
train_batch_size = 128
valid_batch_size = 256
test_batch_size = 256
optimizer_params = {
    "lr": 3e-3,
    "amsgrad": False
}
scheduler_params = {
    "eta_min": 1e-5,
    "T_max": 10
}


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
def select_train_data():
    input_dir = ROOT_PATH / "input"
    tag_list = pd.read_csv(input_dir / "{}.csv".format(SAMPLE_SUBMISSION)).columns[1:].tolist()  # 80 tag list

    # train curated
    train_curated_df = pd.read_csv(input_dir / "{}.csv".format(CURATED_DIR))
    train_curated_df["fpath"] = str(input_dir.absolute()) + "/" + CURATED_DIR + "/" + train_curated_df["fname"]

    train_df = train_curated_df
    # train noisy
    if IS_KERNEL is False:
        train_noisy_df = pd.read_csv(input_dir / "{}.csv".format(NOISY_DIR))
        single_tag_train_noisy_df = train_noisy_df[~train_noisy_df["labels"].str.contains(",")]
        train_noisy_df = None
        for tag in tag_list:  # 80 tags
            temp_df = single_tag_train_noisy_df.query("labels == '{}'".format(tag)).iloc[:50, :]
            if train_noisy_df is None:
                train_noisy_df = temp_df
            else:
                train_noisy_df = pd.concat([train_noisy_df, temp_df])
        train_noisy_df["fpath"] = str(input_dir.absolute()) + "/" + NOISY_DIR + "/" + train_noisy_df["fname"]
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


def spec_augment(spec: np.ndarray, num_mask=2,
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    """Simple augmentation using cross masks
    Reference: https://www.kaggle.com/davids1992/specaugment-quick-implementation
    """
    spec = spec.copy()

    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0

    return spec


def audio_to_melspectrogram(audio, sr):
    spectrogram = librosa.feature.melspectrogram(
        audio,
        sr=sr,
        n_mels=N_MEL,           # https://librosa.github.io/librosa/generated/librosa.filters.mel.html#librosa.filters.mel
        hop_length=HOP_LENGTH,  # to make time steps 128? 恐らくstftのロジックを理解すれば行ける
        n_fft=N_MEL * 20,       # n_mels * 20
        fmin=20,                # Filterbank lowest frequency, Audible range 20[Hz]
        fmax=sr / 2             # Nyquist frequency
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


# >> label convert section
@jit("i1[:](i1[:])")
def label_to_array(label):
    tag_list = pd.read_csv(ROOT_PATH / "input" / "{}.csv".format(SAMPLE_SUBMISSION)).columns[1:].tolist()  # 80 tag list
    array = np.zeros(len(tag_list)).astype(int)
    for tag in label.split(","):
        array[tag_list.index(tag)] = 1
    return array
# << label convert section


# >> dataset section
@stop_watch("load train image")
def df_to_labeldata(fpath_arr, labels):
    """
    fpath   : arr
    labels  : arr
    """

    spec_list = []
    label_list = []

    aug = iaa.ContrastNormalization((0.9, 1.1))

    @jit
    def calc(fpath_arr, labels):
        for idx in range(len(fpath_arr)):
            mod = int(idx % SPEC_AUGMENTATION_RATE)
            # melspectrogram
            y, sr = read_audio(fpath_arr[idx])
            spec_mono = audio_to_melspectrogram(y, sr)
            if mod != 0:
                # spec_mono = spec_augment(spec_mono, num_mask=NUM_MASK,
                #                         freq_masking_max_percentage=FREQ_MASKING_MAX_PERCENTAGE,
                #                         time_masking_max_percentage=TIME_MASKING_MAX_PERCENTAGE)
                spec_mono = aug.augment_image(spec_mono)
            spec_color = mono_to_color(spec_mono)
            spec_list.append(spec_color)

            # labels
            label_list.append(label_to_array(labels[idx]))

    calc(fpath_arr, labels)

    return spec_list, label_list


class TrainDataset(Dataset):
    """
    train_df :columns=["fpath", "labels"]
    """
    def __init__(self, train_path_arr, train_y, transform):
        super().__init__()
        self.transforms = transform
        self.melspectrograms, self.labels = df_to_labeldata(train_path_arr, train_y)

    def __len__(self):
        return len(self.melspectrograms)

    def __getitem__(self, idx):
        # crop
        image = Image.fromarray(self.melspectrograms[idx], mode="RGB")
        time_dim, base_dim = image.size

        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)

        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        return image, label


def load_testdata(fname):
    """
    fname : list
    """
    input_dir = ROOT_PATH / "input"

    spec_list = []

    @jit
    def calc(fname_list):
        for idx in range(len(fname_list)):
            # melspectrogram
            y, sr = read_audio(input_dir / TEST_DIR / fname_list[idx])
            spec_mono = audio_to_melspectrogram(y, sr)
            spec_color = mono_to_color(spec_mono)
            spec_list.append(spec_color)

    calc(fname)

    return spec_list


class TestDataset(Dataset):
    """
    """
    def __init__(self, transform, tta=5):
        super().__init__()
        self.transforms = transform
        self.fnames = pd.read_csv(ROOT_PATH / "input" / "{}.csv".format(SAMPLE_SUBMISSION))["fname"].values.tolist()
        self.melspectrograms = load_testdata(self.fnames)
        self.tta = tta

    def __len__(self):
        return len(self.melspectrograms) * self.tta

    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)

        # crop
        image = Image.fromarray(self.melspectrograms[new_idx], mode="RGB")
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)

        fname = self.fnames[new_idx]
        return image, fname
# << dataset section


# =============== #
#  model section  #
# =============== #
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x


# ================ #
#  metric section  #
# ================ #
# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
@jit
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (retrieved_cumulative_hits[class_rankings[pos_class_indices]] / (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


@jit
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) / np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# =============== #
#  train section  #
# =============== #
@stop_watch("train section")
def train_model(train_df, train_transforms):
    num_classes = len(pd.read_csv(ROOT_PATH / "input" / "{}.csv".format(SAMPLE_SUBMISSION)).columns[1:])

    trn_x, val_x, trn_y, val_y = train_test_split(train_df["fpath"].values, train_df["labels"].values, test_size=0.2, random_state=SEED)

    trn_x = pd.DataFrame(trn_x, columns=["fpath"])
    trn_y = pd.DataFrame(trn_y, columns=["labels"])
    aug_trn_x = trn_x
    aug_trn_y = trn_y
    for i in range(SPEC_AUGMENTATION_RATE - 1):
        aug_trn_x = pd.concat([aug_trn_x, trn_x])
        aug_trn_y = pd.concat([aug_trn_y, trn_y])

    aug_trn_x.sort_index(ascending=False, inplace=True)
    aug_trn_y.sort_index(ascending=False, inplace=True)

    train_dataset = TrainDataset(aug_trn_x["fpath"].values.tolist(), aug_trn_y["labels"].values.tolist(), train_transforms)
    valid_dataset = TrainDataset(val_x, val_y, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    model = Classifier(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = Adam(params=model.parameters(), **optimizer_params)
    scheduler = CosineAnnealingLR(optimizer, **scheduler_params)

    best_epoch = -1
    best_lwlrap = 0.
    mb = master_bar(range(num_epochs))

    for epoch in mb:
        start = time.time()

        # train
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in progress_bar(train_loader, parent=mb):
            preds = model(x_batch.to(device))
            loss = criterion(preds, y_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        # valid
        model.eval()
        valid_preds = np.zeros((len(val_x), num_classes))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            preds = model(x_batch.to(device)).detach()
            loss = criterion(preds, y_batch.to(device))

            preds = torch.sigmoid(preds)
            valid_preds[i * valid_batch_size: (i + 1) * valid_batch_size] = preds.cpu().numpy()

            avg_val_loss += loss.item() / len(valid_loader)

        val_labels = np.array([label_to_array(l) for l in val_y.tolist()])
        score, weight = calculate_per_class_lwlrap(val_labels, valid_preds)
        lwlrap = (score * weight).sum()

        scheduler.step()

        elapsed = time.time() - start
        get_logger(is_torch=True).debug("{}\t{}\t{}\t{}".format(avg_loss, avg_val_loss, lwlrap, elapsed))
        if (epoch + 1) % 5 == 0:
            mb.write(f"Epoch {epoch + 1} -\tavg_train_loss: {avg_loss:.4f}\tavg_val_loss: {avg_val_loss:.4f}\tval_lwlrap: {lwlrap:.6f}\ttime: {elapsed:.0f}s")

        if lwlrap > best_lwlrap:
            best_epoch = epoch + 1
            best_lwlrap = lwlrap
            model_path = "weight_best.pt" if IS_KERNEL else (ROOT_PATH / "model" / "{}_weight_best.pt".format(VERSION)).resolve()
            torch.save(model.state_dict(), model_path)

    return {
        "best_epoch": best_epoch,
        "best_lwlrap": best_lwlrap
    }


# ================= #
#  predict section  #
# ================= #
def predict_model(test_transforms, *, tta=5):
    model_path = "weight_best.pt" if IS_KERNEL else (ROOT_PATH / "model" / "{}_weight_best.pt".format(VERSION)).resolve()

    num_classes = len(pd.read_csv(ROOT_PATH / "input" / "{}.csv".format(SAMPLE_SUBMISSION)).columns[1:])

    test_dataset = TestDataset(test_transforms, tta)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = Classifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_outputs, all_fnames = [], []

    pb = progress_bar(test_loader)
    for images, fnames in pb:
        preds = torch.sigmoid(model(images.to(device)).detach())
        all_outputs.append(preds.cpu().numpy())
        all_fnames.extend(fnames)

    test_preds = pd.DataFrame(data=np.concatenate(all_outputs),
                              index=all_fnames,
                              columns=map(str, range(num_classes)))
    test_preds = test_preds.groupby(level=0).mean()

    return test_preds


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
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    result = train_model(train_df, train_transforms)
    get_logger().info("best_epoch : {},\tbest_lwlrap : {}".format(result["best_epoch"], result["best_lwlrap"]))

    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    test_preds = predict_model(test_transforms)

    test_df = pd.read_csv(ROOT_PATH / "input" / "{}.csv".format(SAMPLE_SUBMISSION))
    test_df.iloc[:, 1:] = test_preds.values

    submission_path = "submission.csv" if IS_KERNEL else (ROOT_PATH / "data" / "submission" / "{}.csv".format(VERSION)).resolve()
    test_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    gc.enable()
    create_logger()
    try:
        main()
    except Exception as e:
        get_logger().error("Exception Occured. \n>> \n {}".format(e))
        raise e
