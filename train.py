import argparse
from collections import Counter
import datetime
import os
import time
import pandas as pd
import sklearn
import torch
from superdebug import debug
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from data import get_model_input
from model import get_model
from utils import get_config, load_model, save_model, to_device
from deepctr_torch.callbacks import ModelCheckpoint
from deepctr_torch.layers.utils import slice_arrays
from deepctr_torch.models.basemodel import BaseModel
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tensorflow.python.keras.callbacks import CallbackList
from tqdm import tqdm
import numpy as np

def test_model_performance(model, test_model_input, test_data, feature_names, target, config):
    with open(config["log_path"], 'a') as log:
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

def get_train_normalization_weights(train_data:pd.DataFrame, config):
    upvote_downvote_weights = np.array(1 * (train_data["VOTE"] == 1) + config["downvote_weight"] * (train_data["VOTE"] == 0))
    # weight = (1. * (y == 1) + 3.5 * (y == 0))[:, 0]
    user_votes_counter = Counter(train_data["USERNAME"])
    if config["user_normalization"] == "equal":
        debug(user_normalization = config["user_normalization"])
        user_weights = np.array([100/user_votes_counter[x] for x in train_data["USERNAME"]])
    else:
        user_weights = np.ones([len(train_data)])
    debug(upvote_downvote_weights * user_weights)
    return upvote_downvote_weights * user_weights

def train_model(config, model, x=None, y=None, weights=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0., shuffle=True):
    # TODO:
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
    assert isinstance(x, dict)
    if model.lm_encoder is not None:
        assert "input_ids" in x, "Make sure train_model_input contains tokenized reddit text"
        text_input_ids = x["input_ids"]
        text_token_type_ids = x["token_type_ids"]
        text_attention_mask = x["attention_mask"]
    x = [x[feature] for feature in model.feature_index if feature in x] # turn into a list of numpy arrays
    for i, _ in enumerate(x): 
        if _.isnull().values.any(): raise ValueError(i)
    do_validation = False
    if validation_split and 0. < validation_split < 1.:
        do_validation = True
        if hasattr(x[0], 'shape'):
            split_at = int(x[0].shape[0] * (1. - validation_split))
        else:
            split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = slice_arrays(x, 0, split_at), slice_arrays(x, split_at)
        y, val_y = slice_arrays(y, 0, split_at), slice_arrays(y, split_at)
        weights, val_weights = slice_arrays(weights, 0, split_at), slice_arrays(weights, split_at)
        if model.lm_encoder is not None:
            text_input_ids, val_text_input_ids = text_input_ids[:split_at], text_input_ids[split_at:]
            text_token_type_ids, val_text_token_type_ids = text_token_type_ids[:split_at], text_token_type_ids[split_at:]
            text_attention_mask, val_text_attention_mask = text_attention_mask[:split_at], text_attention_mask[split_at:]


    else:
        val_x, val_y, val_weights, val_text_input_ids, val_text_token_type_ids, val_text_attention_mask = [], [], [], [], [], []
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    if model.lm_encoder is not None:
        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)),torch.from_numpy(y), torch.from_numpy(weights),text_input_ids,text_token_type_ids,text_attention_mask)
    else:
        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)),torch.from_numpy(y), torch.from_numpy(weights))

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

    train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

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
        for _, train_input in tqdm(enumerate(train_loader)):
            if model.lm_encoder is not None:
                (x_train, y_train, weight_train, text_input_ids_train, text_token_type_ids_train, text_attention_mask_train) = train_input
                text_input_ids, text_token_type_ids, text_attention_mask = to_device(model.device, False, text_input_ids_train, text_token_type_ids_train, text_attention_mask_train)
                encoder_hidden = model.lm_encoder(input_ids = text_input_ids, token_type_ids = text_token_type_ids, attention_mask = text_attention_mask).last_hidden_state
                encoder_hidden_pooled = encoder_hidden.sum(axis=1) / text_attention_mask.sum(axis = -1, keepdim = True)

            else:
                x_train, y_train, weight_train = train_input
            # x = x_train.to(model.device).float()
            # y = y_train.to(model.device).float()
            # weight = weight_train.to(model.device).float()
            x, y, weight = to_device(model.device, True, x_train, y_train, weight_train)
            if model.lm_encoder is not None:
                x = torch.cat([x, encoder_hidden_pooled], dim = -1)
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
                    if metric_fun != sklearn.metrics.accuracy_score and metric_fun != BaseModel._accuracy_score:
                        try:
                            train_result[name].append(metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"), labels = [0,1]))
                        except:
                            pass
                    else:
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
            with open(config["log_path"], 'a') as log:
                log.write(eval_str+"\n")
        save_model(model, epoch, eval_result["acc"], optim, config["save_model_dir"])
        if best_eval_acc == eval_result["acc"]:
            save_model(model, epoch, eval_result["acc"], optim, config["save_model_dir"], "best")
            
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to custom config file.")
    args = parser.parse_args()
    config = get_config(args.config)
    return config


if __name__ == "__main__":
    config = parse_config()
    all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, train_data, test_data = get_model_input(config)
    """
    CTRModel = get_ctr_model(config["model_type"])
    model = CTRModel(all_feature_columns, all_feature_columns, task='binary', device=config["device"], gpus = config["gpus"])
    model.compile(torch.optim.Adam(model.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])
    """
    model = get_model(config, all_feature_columns)
    train_weights = get_train_normalization_weights(train_data, config)
    history = train_model(config, model, x=train_model_input, y=train_data[target].values, weights = train_weights, batch_size=config['batch_size'], epochs=config['num_epochs'], verbose=2, validation_split=0.2) # , callbacks=[ModelCheckpoint(config["save_model_path"])]
    model, _, _, _, _ = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    test_model_performance(model, test_model_input, test_data, feature_names, target, config)