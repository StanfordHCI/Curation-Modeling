import datetime
import os
import time
import pandas as pd
import torch
from superdebug import debug
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from data import get_model_input
from utils import get_config, get_ctr_model, load_model, save_model
from deepctr_torch.callbacks import ModelCheckpoint
from deepctr_torch.layers.utils import slice_arrays
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tensorflow.python.keras.callbacks import CallbackList
from tqdm import tqdm
import numpy as np

CONFIG_PATH = "configs/debug.yml"
config, log_path = get_config(CONFIG_PATH)
now = datetime.datetime.now()
with open(log_path, 'a') as log:
    log.write("-----------------" + f"{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}" + "-----------------\n")

def test_model_performance(model, test_model_input, test_data, feature_names, target):
    with open(log_path, 'a') as log:
        eval_all_test_data = model.evaluate(test_model_input, test_data[target].values, batch_size=config['batch_size'])
        debug(eval_all_test_data=eval_all_test_data)
        log.write("eval_all_test_data:" + str(eval_all_test_data)+"\n")
        test_upvote_data = pd.concat([test_data[test_data["VOTE"] == 1], test_data[test_data["VOTE"] == 0].iloc[0:1]], axis=0)
        test_model_upvote_input = {name:test_upvote_data[name] for name in feature_names}
        eval_all_upvote_data = model.evaluate(test_model_upvote_input, test_upvote_data[target].values, batch_size=config['batch_size'])
        debug(eval_all_upvote_data=eval_all_upvote_data)
        log.write("eval_all_upvote_data:" + str(eval_all_upvote_data)+"\n")
        test_downvote_data = pd.concat([test_data[test_data["VOTE"] == 0], test_data[test_data["VOTE"] == 1].iloc[0:1]], axis=0)
        test_model_downvote_input = {name:test_downvote_data[name] for name in feature_names}
        eval_all_downvote_data = model.evaluate(test_model_downvote_input, test_downvote_data[target].values, batch_size=config['batch_size'])
        debug(eval_all_downvote_data=eval_all_downvote_data)
        log.write("eval_all_downvote_data:" + str(eval_all_downvote_data)+"\n")


def train_model(model, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
        validation_data=None, shuffle=True):
    """

    :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
        dictionary mapping input names to Numpy arrays.
    :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
    :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
    :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
    :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
    :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
    :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
    :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
    """
    if isinstance(x, dict):
        x = [x[feature] for feature in model.feature_index]
    for i, _ in enumerate(x): 
        if _.isnull().values.any(): raise ValueError(i)
    do_validation = False
    if validation_data:
        do_validation = True
        if len(validation_data) == 2:
            val_x, val_y = validation_data
            val_sample_weight = None
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
        else:
            raise ValueError('When passing a `validation_data` argument, it must contain either 2 items (x_val, y_val), or 3 items (x_val, y_val, val_sample_weights), or alternatively it could be a dataset or a dataset or a dataset iterator. However we received `validation_data=%s`' % validation_data)
        if isinstance(val_x, dict):
            val_x = [val_x[feature] for feature in model.feature_index]

    elif validation_split and 0. < validation_split < 1.:
        do_validation = True
        if hasattr(x[0], 'shape'):
            split_at = int(x[0].shape[0] * (1. - validation_split))
        else:
            split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_arrays(x, 0, split_at),
                    slice_arrays(x, split_at))
        y, val_y = (slice_arrays(y, 0, split_at),
                    slice_arrays(y, split_at))

    else:
        val_x = []
        val_y = []
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)),torch.from_numpy(y))
    if torch.isnan(train_tensor_data.tensors[0]).any(): raise ValueError('nan')
    if batch_size is None:
        batch_size = 256

    model = model.train()
    loss_func = model.loss_func
    optim = model.optim
    best_eval_acc = 0
    if config["load_pretrained_model"]:
        model, optim, initial_epoch, best_eval_acc, save_dict = load_model(config["save_model_dir"], model, optim, initial_epoch, best_eval_acc)
    if model.gpus:
        print('parallel running on these gpus:', model.gpus)
        _model = torch.nn.DataParallel(model, device_ids=model.gpus)
        batch_size *= len(model.gpus)  # input `batch_size` is batch_size per gpu
    else:
        _model = model

    train_loader = DataLoader(
        dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

    sample_num = len(train_tensor_data)
    steps_per_epoch = (sample_num - 1) // batch_size + 1
    # Train
    print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
        len(train_tensor_data), len(val_y), steps_per_epoch))
    for epoch in range(initial_epoch, epochs):
        epoch_logs = {}
        start_time = time.time()
        loss_epoch = 0
        total_loss_epoch = 0
        train_result = {}
        for _, (x_train, y_train) in tqdm(enumerate(train_loader)):
            x = x_train.to(model.device).float()
            y = y_train.to(model.device).float()
            weight = (1. * (y == 1) + 3.5 * (y == 0))[:, 0]
            if torch.isnan(x).any(): raise ValueError("nan")

            y_pred = _model(x).squeeze()

            optim.zero_grad()
            loss = loss_func(y_pred, y.squeeze(), weight = weight, reduction='sum')
            reg_loss = model.get_regularization_loss()

            total_loss = loss + reg_loss + model.aux_loss

            loss_epoch += loss.item()
            total_loss_epoch += total_loss.item()
            total_loss.backward()
            optim.step()

            if verbose > 0:
                for name, metric_fun in model.metrics.items():
                    if name not in train_result:
                        train_result[name] = []
                    train_result[name].append(metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

        # Add epoch_logs
        epoch_logs["loss"] = total_loss_epoch / sample_num
        for name, result in train_result.items():
            epoch_logs[name] = np.sum(result) / steps_per_epoch

        if do_validation:
            eval_result = model.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                epoch_logs["val_" + name] = result
            best_eval_acc = max(best_eval_acc, eval_result["acc"])
        # verbose
        if verbose > 0:
            epoch_time = int(time.time() - start_time)
            print('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s - loss: {1: .4f}".format(
                epoch_time, epoch_logs["loss"])

            for name in model.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in model.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            print(eval_str)
            with open(log_path, 'a') as log:
                log.write(eval_str+"\n")
        save_model(model, epoch, eval_result["acc"], optim, config["save_model_dir"])
        if best_eval_acc == eval_result["acc"]:
            save_model(model, epoch, eval_result["acc"], optim, config["save_model_dir"], "best")
            

all_feature_columns, target, train_model_input, test_model_input, feature_names, train_data, test_data = get_model_input(config)
# from deepctr_torch.models import MLR as CTRModel # xDeepFM is best
CTRModel = get_ctr_model(config["model_type"])
model = CTRModel(all_feature_columns, all_feature_columns, task='binary', device=config["device"], gpus = config["gpus"])
model.compile(torch.optim.Adam(model.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])


if __name__ == "__main__":
    history = train_model(model, train_model_input, train_data[target].values, batch_size=config['batch_size'], epochs=config['num_epochs'], verbose=2, validation_split=0.2) # , callbacks=[ModelCheckpoint(config["save_model_path"])]
    model, _, _, _, _ = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    test_model_performance(model, test_model_input, test_data, feature_names, target)