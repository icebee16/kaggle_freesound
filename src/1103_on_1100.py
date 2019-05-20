# ==================================================================================
# Baseline Model
# date : 2019/05/05
# reference : https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch
# comment : [change point] IMAGE_VERSION {1000 > 1003}
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
from fastprogress import master_bar, progress_bar

import numpy as np
import pandas as pd
from numba import jit

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
IMAGE_VERSION = "1003"
FOLD_NUM = 5
ROOT_PATH = Path("..") if IS_KERNEL else Path(__file__).parents[1]
DataLoader = partial(DataLoader, num_workers=cpu_count())
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
def select_train_data():
    fold_dir = ROOT_PATH / "data" / "fold"
    img_dir = ROOT_PATH / "image" / IMAGE_VERSION

    # train curated
    train_curated_df = pd.read_csv(fold_dir / "train_curated_sfk.csv")
    train_curated_df = train_curated_df[["fname", "labels", "fold"]]
    train_curated_df["fpath"] = str(img_dir.absolute()) + "/train_curated/" + train_curated_df["fname"].str[:-4] + ".png"

    # train noisy

    # df concat
    train_df = train_curated_df

    return train_df[["fpath", "labels", "fold"]]
# << data select section


# >> label convert section
@jit("i1[:](i1[:])")
def label_to_array(label):
    tag_list = pd.read_csv(ROOT_PATH / "input" / "sample_submission.csv").columns[1:].tolist()  # 80 tag list
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

    @jit
    def calc(fpath_arr, labels):
        for idx in tqdm(range(len(fpath_arr))):
            # melspectrogram
            img = Image.open(fpath_arr[idx])
            spec_color = np.asarray(img)
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
        # zero padding by time axis
        color_spec = self.melspectrograms[idx]
        (mel_dim, time_dim, channel_dim) = color_spec.shape
        color_spec = np.pad(color_spec, [(0, 0), (mel_dim, mel_dim), (0, 0)], "constant")

        # peak crop
        peak_timing = color_spec.sum(axis=0).sum(axis=1).argmax()
        peak_timing = max([peak_timing, mel_dim])

        image = Image.fromarray(color_spec, mode="RGB")
        crop = peak_timing - int(mel_dim / 2)  # (WIP)
        image = image.crop([crop, 0, crop + mel_dim, mel_dim])
        image = self.transforms(image).div_(255)

        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        return image, label
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
def train_model(train_df, train_transforms, fold):
    get_logger().info("[start] >> {} fold".format(fold))
    num_epochs = 80
    batch_size = 64
    test_batch_size = 256
    lr = 3e-3
    eta_min = 1e-5
    t_max = 10

    num_classes = len(pd.read_csv(ROOT_PATH / "input" / "sample_submission.csv").columns[1:])

    trn_data = train_df.query("fold != {}".format(fold))
    trn_x = trn_data["fpath"].values
    trn_y = trn_data["labels"].values
    val_data = train_df.query("fold == {}".format(fold))
    val_x = val_data["fpath"].values
    val_y = val_data["labels"].values

    train_dataset = TrainDataset(trn_x, trn_y, train_transforms)
    valid_dataset = TrainDataset(val_x, val_y, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False)

    model = Classifier(num_classes=num_classes).cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    best_epoch = -1
    best_lwlrap = 0.
    mb = master_bar(range(num_epochs))

    for epoch in mb:
        start = time.time()

        # train
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in progress_bar(train_loader, parent=mb):
            preds = model(x_batch.cuda())
            loss = criterion(preds, y_batch.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        # valid
        model.eval()
        valid_preds = np.zeros((len(val_x), num_classes))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            preds = model(x_batch.cuda()).detach()
            loss = criterion(preds, y_batch.cuda())

            preds = torch.sigmoid(preds)
            valid_preds[i * test_batch_size: (i + 1) * test_batch_size] = preds.cpu().numpy()

            avg_val_loss += loss.item() / len(valid_loader)

        val_labels = np.array([label_to_array(l) for l in val_y.tolist()])
        score, weight = calculate_per_class_lwlrap(val_labels, valid_preds)
        lwlrap = (score * weight).sum()

        scheduler.step()

        elapsed = time.time() - start
        get_logger(is_torch=True).debug("{}\t{}\t{}\t{}".format(avg_loss, avg_val_loss, lwlrap, elapsed))
        if (epoch + 1) % 10 == 0:
            mb.write(f"Epoch {epoch + 1} -\tavg_train_loss: {avg_loss:.4f}\tavg_val_loss: {avg_val_loss:.4f}\tval_lwlrap: {lwlrap:.6f}\ttime: {elapsed:.0f}s")

        if lwlrap > best_lwlrap:
            best_epoch = epoch + 1
            best_lwlrap = lwlrap
            model_path = "weight_best_{}.pt".format(fold) if IS_KERNEL else (ROOT_PATH / "model" / "{}_weight_best_{}.pt".format(VERSION, fold)).resolve()
            torch.save(model.state_dict(), model_path)

    get_logger().info("[ end ] >> {} fold".format(fold))
    get_logger(is_torch=True).info("")
    return {
        "best_epoch": best_epoch,
        "best_lwlrap": best_lwlrap
    }


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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    lwlrap_result = 0
    for i in range(FOLD_NUM):
        result = train_model(train_df, train_transforms, i)
        get_logger().info("[fold {}]best_epoch : {},\tbest_lwlrap : {}".format(i, result["best_epoch"], result["best_lwlrap"]))
    lwlrap_result += result["best_lwlrap"] / FOLD_NUM

    get_logger().info("[result]best_lwlrap : {}".format(lwlrap_result))


if __name__ == "__main__":
    gc.enable()
    create_logger()
    try:
        main()
    except Exception as e:
        get_logger().error("Exception Occured. \n>> \n {}".format(e))
        raise e
