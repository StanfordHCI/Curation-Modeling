import argparse
from collections import Counter, OrderedDict
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
from utils import get_config, load_model, save_model, to_device, parse_config
from deepctr_torch.callbacks import ModelCheckpoint
from deepctr_torch.layers.utils import slice_arrays
from deepctr_torch.models.basemodel import BaseModel
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tensorflow.python.keras.callbacks import CallbackList
from tqdm import tqdm
import numpy as np
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def get_normalization_weights(data:pd.DataFrame, config):
    upvote_downvote_weights = np.array(1 * (data["VOTE"] == 1) + config["downvote_weight"] * (data["VOTE"] == 0))
    user_votes_counter = Counter(data["USERNAME"])
    if config["user_normalization"] == "equal":
        debug(user_normalization = config["user_normalization"])
        user_weights = np.array([100/user_votes_counter[x] for x in data["USERNAME"]])
    else:
        user_weights = np.ones([len(data)])
    debug(upvote_downvote_weights * user_weights)
    return upvote_downvote_weights * user_weights

def apply_metric(metric_func, y_true, y_pred, sample_weight = None):
    if metric_func != sklearn.metrics.accuracy_score and metric_func != BaseModel._accuracy_score:
        try:
            val = metric_func(y_true, y_pred, labels = [0,1], sample_weight=sample_weight)
        except Exception as e:
            debug(e)
            val = 1
    else:
        val = metric_func(y_true, y_pred, sample_weight=sample_weight)
    return val
def train_model(config, model, x=None, y=None, weights=None, batch_size=256, epochs=1, verbose=1, initial_epoch=0, validation_split=0., shuffle=True, max_voted_users=100, step_generator = False):
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
    if step_generator: assert epochs == 1 and batch_size == 1
    if model.lm_encoder is not None:
        assert "input_ids" in x, "Make sure train_model_input contains tokenized reddit text"
        text_input_ids = x["input_ids"]
        text_token_type_ids = x["token_type_ids"] if "token_type_ids" in x else None
        text_attention_mask = x["attention_mask"]
    else:
        text_input_ids, text_token_type_ids, text_attention_mask = None, None, None
    if "UPVOTED_USERS" in x:
        x["UPVOTED_USERS"] = pad_sequences(x["UPVOTED_USERS"], maxlen=max_voted_users, padding='post')
        x["DOWNVOTED_USERS"] = pad_sequences(x["DOWNVOTED_USERS"], maxlen=max_voted_users, padding='post')
    x = [x[feature] for feature in model.feature_index if feature in x] # turn into a list of numpy arrays
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
            text_token_type_ids, val_text_token_type_ids = (text_token_type_ids[:split_at], text_token_type_ids[split_at:]) if text_token_type_ids is not None else (torch.zeros([text_input_ids.shape[0], 0], dtype = int), torch.zeros([val_text_input_ids.shape[0], 0], dtype = int))
            text_attention_mask, val_text_attention_mask = text_attention_mask[:split_at], text_attention_mask[split_at:]
        else:
            val_text_input_ids, val_text_token_type_ids, val_text_attention_mask = None, None, None
    else:
        val_x, val_y, val_weights, val_text_input_ids, val_text_token_type_ids, val_text_attention_mask = [], [], [], [], [], []
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)
    train_tensor_data, train_loader = get_data_loader(model.lm_encoder is not None, x, text_input_ids,text_token_type_ids,text_attention_mask, y, weights, shuffle=shuffle, batch_size=batch_size)

    model = model.train()
    loss_func = model.loss_func
    optim = model.optim
    best_eval_acc = 0
    if config["load_pretrained_model"]:
        model, optim, initial_epoch, best_eval_acc, save_dict = load_model(config["save_model_dir"], model, optim, initial_epoch, best_eval_acc)
    if model.gpus:
        if not step_generator:
            print('parallel running on these gpus:', model.gpus)
        _model = torch.nn.DataParallel(model, device_ids=model.gpus)
        batch_size *= len(model.gpus)  # input `batch_size` is batch_size per gpu
    else:
        _model = model

    sample_num = len(train_tensor_data)
    steps_per_epoch = (sample_num - 1) // batch_size + 1
    # Train
    if not step_generator:
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(len(train_tensor_data), len(val_y), steps_per_epoch))
    for epoch in range(initial_epoch, epochs):
        # epoch_logs = {}
        start_time = time.time()
        loss_epoch = 0
        total_loss_epoch = 0
        train_result = {}
        if not step_generator:
            train_loader = tqdm(train_loader, total = steps_per_epoch, desc = "Training")
        for _, train_input in enumerate(train_loader):
            x, y, weight = convert_CTR_model_input(model, train_input, sample_voted_users=config["sample_part_voted_users"])
            # debug(x=x[0], y = y[0])
            y_pred = _model(x)

            optim.zero_grad()
            loss = loss_func(y_pred.reshape(y.shape), y, weight = weight.reshape(y.shape), reduction='sum')
            reg_loss = model.get_regularization_loss()

            total_loss = loss + reg_loss + model.aux_loss

            loss_epoch += loss.item()
            total_loss_epoch += total_loss.item()
            total_loss.backward()
            optim.step()

            if verbose > 0 and not step_generator:
                for name, metric_func in model.metrics.items():
                    if name not in train_result:
                        train_result[name] = []
                    metric_result = apply_metric(metric_func, y.cpu().numpy(), y_pred.reshape(y.shape).cpu().data.numpy())
                    train_result[name].append(metric_result)
            if step_generator:
                yield model

        if not step_generator:
            # Add epoch_logs
            # epoch_logs["loss"] = total_loss_epoch / sample_num
            # for name, result in train_result.items():
                # epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = evaluate_model(model, val_x, val_text_input_ids, val_text_token_type_ids, val_text_attention_mask, val_y, weights = val_weights, batch_size=batch_size)
                # for name, result in eval_result.items():
                #     epoch_logs["val_" + name] = result
                best_eval_acc = max(best_eval_acc, eval_result["acc"])
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, total_loss_epoch / sample_num)
                for name, result in train_result.items():
                    eval_str += " - " + name + ": {0: .4f}".format(np.sum(result) / steps_per_epoch)
                if do_validation:
                    for name, result in eval_result.items():
                        eval_str += " - " + "val_" + name + ": {0: .4f}".format(result)
                print(eval_str)
                with open(config["log_path"], 'a') as log:
                    log.write(eval_str+"\n")
            save_model(model, epoch, eval_result["acc"], optim, config["save_model_dir"])
            if best_eval_acc == eval_result["acc"]:
                save_model(model, epoch, eval_result["acc"], optim, config["save_model_dir"], "best")
    yield None
def sample_updown_voted_users(x:torch.tensor, model, vote = "upvote", interactive = False):
    if x is None: return None
    updown_voted_users_batch_orig = x[:,model.feature_index[f"{vote.upper()}D_USERS"][0]:model.feature_index[f"{vote.upper()}D_USERS"][1]]
    updown_voted_users_batch = updown_voted_users_batch_orig.new_zeros(updown_voted_users_batch_orig.shape)
    for batch_i in range(len(updown_voted_users_batch_orig)):
        updown_voted_users = set(tuple(updown_voted_users_batch_orig[batch_i].cpu().numpy()))
        if 0 in updown_voted_users:
            updown_voted_users.remove(0)
        if not interactive:
            sample_num = random.randint(0, len(updown_voted_users))
            sampled_updown_voted_users = random.sample(list(updown_voted_users), sample_num)
        else:
            print(f"Original {vote}d users:", list(updown_voted_users))
            selected_users = input(f"Please select {model.feature_index[f'{vote.upper()}D_USERS'][1] - model.feature_index[f'{vote.upper()}D_USERS'][0]} {vote}d users (input '.' to stop): ")
            if selected_users == ".":
                return None
            sampled_updown_voted_users = [int(user) for user in selected_users.split() if int(user) != 0]
            print(f"New {vote}d users:", sampled_updown_voted_users)
        sampled_updown_voted_users = torch.tensor(sampled_updown_voted_users)
        updown_voted_users_batch[batch_i, :len(sampled_updown_voted_users)] = sampled_updown_voted_users
    x_new = x.clone()
    x_new[:, model.feature_index[f"{vote.upper()}D_USERS"][0]:model.feature_index[f"{vote.upper()}D_USERS"][1]] = updown_voted_users_batch
    return x_new
def convert_CTR_model_input(model, dataloader_input, sample_voted_users = False, interactive = False):
    if model.lm_encoder is not None:
        (x, y, weight, text_input_ids, text_token_type_ids, text_attention_mask) = dataloader_input
        text_input_ids, text_token_type_ids, text_attention_mask = to_device(model.device, False, text_input_ids, text_token_type_ids, text_attention_mask)
        encoder_hidden = model.lm_encoder(input_ids = text_input_ids, token_type_ids = text_token_type_ids if text_token_type_ids.shape[-1] > 0 else None, attention_mask = text_attention_mask).last_hidden_state
        encoder_hidden_pooled = (text_attention_mask[:,:,None] * encoder_hidden).sum(axis=1) / text_attention_mask.sum(axis = 1, keepdim = True)
    else:
        x, y, weight = dataloader_input
    if "UPVOTED_USERS" in model.feature_index and sample_voted_users:
        x = sample_updown_voted_users(x, model, vote = "upvote", interactive = interactive)
        x = sample_updown_voted_users(x, model, vote = "downvote", interactive = interactive)
        if x is None:
            return None
    x, y, weight = to_device(model.device, True, x, y, weight)
    if model.lm_encoder is not None:
        x = torch.cat([x, encoder_hidden_pooled], dim = -1)
    return x,y,weight

def get_data_loader(has_lm_encoder, x, text_input_ids,text_token_type_ids,text_attention_mask, y, weights = None, shuffle=True, batch_size=256):
    if weights is None:
        weights = np.zeros(y.shape, dtype = float)
    if text_token_type_ids is None and text_input_ids is not None:
        text_token_type_ids = torch.zeros([text_input_ids.shape[0], 0], dtype = int)
    if has_lm_encoder:
        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)),torch.from_numpy(y), torch.from_numpy(weights),text_input_ids,text_token_type_ids,text_attention_mask)
    else:
        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)),torch.from_numpy(y), torch.from_numpy(weights))

    if torch.isnan(tensor_data.tensors[0]).any(): raise ValueError('nan')
    data_loader = DataLoader(dataset=tensor_data, shuffle=shuffle, batch_size=batch_size)
    return tensor_data, data_loader

def evaluate_model(model, x, text_input_ids, text_token_type_ids, text_attention_mask, y, data=None, weights = None, batch_size=256, sample_voted_users=False, max_voted_users=100, return_prediction = False, data_info = None, disable_tqdm = False):
    model = model.eval()
    if isinstance(x, dict):
        if "UPVOTED_USERS" in x:
            x["UPVOTED_USERS"] = pad_sequences(x["UPVOTED_USERS"], maxlen=max_voted_users, padding='post')
            x["DOWNVOTED_USERS"] = pad_sequences(x["DOWNVOTED_USERS"], maxlen=max_voted_users, padding='post')
        x = [x[feature] for feature in model.feature_index if feature in x]
    
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    tensor_data, test_loader = get_data_loader(model.lm_encoder is not None, x, text_input_ids,text_token_type_ids,text_attention_mask, y, weights, shuffle=False, batch_size=batch_size)
    pred_ans = []

    if not disable_tqdm:
        test_loader = tqdm(test_loader)
    with torch.no_grad():
        for _, test_input in enumerate(test_loader):
            x, _y, weight = convert_CTR_model_input(model, test_input, sample_voted_users=sample_voted_users)
            y_pred = model(x).cpu().data.numpy()  # .squeeze()
            pred_ans.append(y_pred)

    pred_ans =  np.concatenate(pred_ans).astype("float64")
    if return_prediction:
        return pred_ans

    if data_info is not None:
        test_filter = {"": (y >=-1).squeeze(), "_train_user_votes_num>=3": data_info["train_user_votes_num"] >= 3, "_train_submission_votes_num>=3": data_info["train_submission_votes_num"] >= 3, "_train_user_votes_num<=3": data_info["train_user_votes_num"] <= 3, "_train_submission_votes_num<=3": data_info["train_submission_votes_num"] <= 3} 
    else:
        test_filter = {"": (y >=-1).squeeze()}
    eval_result = OrderedDict()
    for wei in [None, weights]:
        for filter_name in test_filter:
            filter = test_filter[filter_name]
            eval_result[f"{filter_name}{'_with_weight' if wei is not None else ''}"] = 0
            for name, metric_func in model.metrics.items():
                if 0 not in y[filter].shape:
                    eval_result[f"{name}{filter_name}{'_with_weight' if wei is not None else ''}"] = apply_metric(metric_func, y[filter], pred_ans[filter], sample_weight=wei[filter] if wei is not None else None)
                if data is not None:
                    for vote in [0, 1]:
                        if 0 not in y[(data["VOTE"] == vote).to_numpy() * filter].shape:
                            eval_result[f"{name}_vote_{vote}{filter_name}{'_with_weight' if wei is not None else ''}"] = apply_metric(metric_func, y[(data["VOTE"] == vote).to_numpy() * filter], pred_ans[(data["VOTE"] == vote).to_numpy() * filter], sample_weight=wei[(data["VOTE"] == vote).to_numpy() * filter] if wei is not None else None)
    model = model.train()
    return eval_result


if __name__ == "__main__":
    config_path, config = parse_config()
    all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, max_voted_users, train_data, test_data, test_data_info = get_model_input(config)
    model = get_model(config, all_feature_columns, feature_names)
    train_weights = get_normalization_weights(train_data, config)
    next(train_model(config, model, x=train_model_input, y=train_data[target].values, weights = train_weights, batch_size=config['batch_size'], epochs=config['num_epochs'], verbose=2, validation_split=0.2, max_voted_users=max_voted_users))
    model, _, _, _, _ = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    test_weights = get_normalization_weights(test_data, config)
    if config["use_voted_users_feature"]:
        debug("Use all voted users as feature")
    eval_all_test_data = evaluate_model(model, test_model_input, test_model_input.get("input_ids", None), test_model_input.get("token_type_ids", None), test_model_input.get("attention_mask", None), test_data[target].values, data = test_data, weights = test_weights, batch_size=config['batch_size'], max_voted_users=max_voted_users,sample_voted_users=False, data_info = test_data_info)
    debug(eval_all_test_data=eval_all_test_data)

    if config["use_voted_users_feature"] and config["sample_part_voted_users"]:
        debug("Sample part voted users as feature")
        eval_all_test_data = evaluate_model(model, test_model_input, test_model_input.get("input_ids", None), test_model_input.get("token_type_ids", None), test_model_input.get("attention_mask", None), test_data[target].values, data = test_data, weights = test_weights, batch_size=config['batch_size'],max_voted_users=max_voted_users, sample_voted_users=True, data_info = test_data_info)
        debug(eval_all_test_data=eval_all_test_data)

    with open(config["log_path"], 'a') as log:
        log.write("eval_all_test_data:" + str(eval_all_test_data)+"\n")
    debug(config_path = config_path, log_path=config["log_path"])