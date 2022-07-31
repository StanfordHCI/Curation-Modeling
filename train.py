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
from dynamic_data import get_data_loader
from process_data import get_model_input
from model import TransformerVoter, LinearModel, _accuracy_score
from utils import get_config, load_model, save_model, to_device, parse_config
from deepctr_torch.callbacks import ModelCheckpoint
from deepctr_torch.layers.utils import slice_arrays
from deepctr_torch.models.basemodel import BaseModel
import torch.utils.data as Data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def get_normalization_weights(data:pd.DataFrame, config):
    upvote_downvote_weights = np.array(1 * (data["VOTE"] == 1) + config["downvote_weight"] * (data["VOTE"] != 1))
    user_votes_counter = Counter(data["USERNAME"])
    if config["user_normalization"] == "equal":
        debug(user_normalization = config["user_normalization"])
        user_weights = np.array([100/user_votes_counter[x] for x in data["USERNAME"]])
    else:
        user_weights = np.ones([len(data)])
    normalization_weights = upvote_downvote_weights * user_weights
    debug(normalization_weights=normalization_weights)
    return normalization_weights

def apply_metric(metric_func, y_true, y_pred, sample_weight = None):
    if metric_func != sklearn.metrics.accuracy_score and metric_func != _accuracy_score:
        try:
            val = metric_func(y_true, y_pred, labels = [0,1], sample_weight=sample_weight)
        except Exception as e:
            val = None
    else:
        val = metric_func(y_true, y_pred, sample_weight=sample_weight)
    return val
def train_model(config, model, data:pd.DataFrame, weights=None, batch_size=256, epochs=1, verbose=1, initial_epoch=0, validation_split=0., shuffle=True, step_generator = False, n_step_per_sample = 1):
    """
    :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.
    :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
    :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
    :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
    :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
    :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
    :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
    :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
    """
    if step_generator: assert epochs == 1 and batch_size == 1
    do_validation = False
    if validation_split and 0. < validation_split < 1.:
        do_validation = True
        split_at = int(len(data) * (1. - validation_split))
        data, val_data = data.iloc[:split_at], data.iloc[split_at:]
        weights, val_weights = weights[:split_at], weights[split_at:] # slice_arrays(weights, 0, split_at), slice_arrays(weights, split_at)
    else:
        val_data, val_weights = [], []
    # trainset, train_loader = get_data_loader(config, data, model.tokenizer, categorical_features, string_features, target, weight=weights, shuffle=shuffle, batch_size=batch_size) # TODO: change back
    trainset, train_loader = get_data_loader(config, val_data, model.tokenizer, categorical_features, string_features, target, weight=val_weights, shuffle=shuffle, batch_size=batch_size)

    model = model.train()
    optim = model.optim
    best_eval_acc = 0
    if config["load_pretrained_model"]:
        model, optim, initial_epoch, best_eval_acc, save_dict = load_model(config["save_model_dir"], model, optim, initial_epoch, best_eval_acc)
    if model.gpus:
        if not step_generator:
            print('parallel running on these gpus:', model.gpus)
        _model = torch.nn.DataParallel(model, device_ids=model.gpus)
    else:
        _model = model

    sample_num = len(trainset)

    # Train
    if not step_generator:
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(len(trainset), len(val_data), len(train_loader)))
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        loss_epoch = 0
        total_loss_epoch = 0
        train_result = {}
        if not step_generator:
            train_loader_tqdm = tqdm(train_loader, desc = "Training")
        for _, train_input in enumerate(train_loader_tqdm):
            # x, y, weight = convert_CTR_model_input(model, train_input, sample_voted_users=config["sample_part_voted_users"], add_target_user_ratio = config["add_target_user_ratio"])
            input_ids, token_type_ids, attention_mask, label, weight, df_index = train_input
            # x, _y, weight = convert_CTR_model_input(model, test_input, sample_voted_users=sample_voted_users)
            input_ids, token_type_ids, attention_mask, label, weight = to_device(model.device, False, input_ids, token_type_ids, attention_mask, label.float(), weight)
            for _step_i in range(n_step_per_sample):
                y_pred = _model(input_ids, token_type_ids, attention_mask)

                optim.zero_grad()
                loss = F.binary_cross_entropy(y_pred.reshape(label.shape), label, weight = weight.reshape(label.shape), reduction='sum')
                reg_loss = model.get_regularization_loss()
                total_loss = loss + reg_loss # + model.aux_loss

                loss_epoch += loss.item()
                total_loss_epoch += total_loss.item()
                total_loss.backward()
                optim.step()

            if verbose > 0 and not step_generator:
                for name, metric_func in model.metrics.items():
                    if name not in train_result:
                        train_result[name] = []
                    metric_result = apply_metric(metric_func, label.cpu().numpy(), y_pred.reshape(label.shape).cpu().data.numpy())
                    if metric_result is not None: train_result[name].append(metric_result)
            if step_generator:
                yield model

        if not step_generator:



        #     valset, val_loader = get_data_loader(config, val_data, model.tokenizer, categorical_features, string_features, target, weight=val_weights, shuffle=shuffle, batch_size=batch_size)
        #     accs = []
        #     with torch.no_grad():
        #         model.eval()
        #         for _, val_input in enumerate(tqdm(val_loader, desc = "Validating")):
        #             input_ids, token_type_ids, attention_mask, label, weight, df_index = val_input
        #             input_ids, token_type_ids, attention_mask, label, weight = to_device(model.device, False, input_ids, token_type_ids, attention_mask, label.float(), weight)
        #             y_pred = _model(input_ids, token_type_ids, attention_mask)
        #             acc = apply_metric(_accuracy_score, label.cpu().numpy(), y_pred.reshape(label.shape).cpu().data.numpy())
        #             accs.append(acc)
        #     avg_acc = np.sum(accs) / len(val_loader)
        #     debug(avg_acc=avg_acc)



            
            if do_validation:
                eval_result = evaluate_model(config, model, val_data, weights = val_weights, batch_size=batch_size)
                best_eval_acc = max(best_eval_acc, eval_result["acc_with_weight"])
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, total_loss_epoch / sample_num)
                for name, result in train_result.items():
                    eval_str += " - " + name + ": {0: .4f}".format(np.sum(result) / len(train_loader))
                if do_validation:
                    for name, result in eval_result.items():
                        eval_str += " - " + "val_" + name + (": {0: .4f}".format(result) if result is not None else ": N/A")
                print(eval_str)
                with open(config["log_path"], 'a') as log:
                    log.write(eval_str+"\n")
            save_model(model, epoch, eval_result["acc"], optim, config["save_model_dir"])
            if best_eval_acc == eval_result["acc_with_weight"]:
                save_model(model, epoch, eval_result["acc_with_weight"], optim, config["save_model_dir"], "best")
    yield None


def evaluate_model(config, model, data:pd.DataFrame, weights = None, batch_size=256, shuffle = False, return_prediction = False, sample_voted_users = False, data_info = None, disable_tqdm = False):
    model = model.eval()
    testset, test_loader = get_data_loader(config, data, model.tokenizer, categorical_features, string_features, target, weights, sample_voted_users=sample_voted_users, shuffle=False, batch_size=batch_size)
    pred_ans = []
    accs = []

    if not disable_tqdm:
        test_loader = tqdm(test_loader)
    with torch.no_grad():
        for _, test_input in enumerate(test_loader):
            input_ids, token_type_ids, attention_mask, label, weight, df_index = test_input
            # x, _y, weight = convert_CTR_model_input(model, test_input, sample_voted_users=sample_voted_users)
            input_ids, token_type_ids, attention_mask, label, weight = to_device(model.device, False, input_ids, token_type_ids, attention_mask, label.float(), weight)
            y_pred = model(input_ids, token_type_ids, attention_mask).cpu().data.numpy()  # .squeeze()
            pred_ans.append(y_pred)
            acc = apply_metric(_accuracy_score, label.cpu().numpy(), y_pred.reshape(label.shape))
            accs.append(acc)
    avg_acc = np.sum(accs) / len(test_loader)
    debug(avg_acc=avg_acc)

    pred_ans =  np.concatenate(pred_ans).astype("float64")
    if return_prediction:
        return pred_ans

    if data_info is not None:
        test_filter = {"": (data["VOTE"] >=-1).to_numpy(), "_train_user_votes_num>=3": data_info["train_user_votes_num"] >= 3, "_train_submission_votes_num>=3": data_info["train_submission_votes_num"] >= 3, "_train_user_votes_num<=3": data_info["train_user_votes_num"] <= 3, "_train_submission_votes_num<=3": data_info["train_submission_votes_num"] <= 3} 
    else:
        test_filter = {"": (data["VOTE"] >=-1).to_numpy()}
    ground_truth = data["VOTE"].to_numpy()
    eval_result = OrderedDict()
    debug(ground_truth=ground_truth, pred_ans=pred_ans)
    for wei in [None, weights]:
        for filter_name in test_filter:
            filter = test_filter[filter_name]
            eval_result[f"{filter_name}{'_with_weight' if wei is not None else ''}"] = 0
            for name, metric_func in model.metrics.items():
                if 0 not in ground_truth[filter].shape:
                    eval_result[f"{name}{filter_name}{'_with_weight' if wei is not None else ''}"] = apply_metric(metric_func, ground_truth[filter], pred_ans[filter], sample_weight=wei[filter] if wei is not None else None) # TODO:
                if data is not None:
                    for vote in [0, 1]:
                        if 0 not in ground_truth[(data["VOTE"] == vote).to_numpy() * filter].shape:
                            eval_result[f"{name}_vote_{vote}{filter_name}{'_with_weight' if wei is not None else ''}"] = apply_metric(metric_func, ground_truth[(data["VOTE"] == vote).to_numpy() * filter], pred_ans[(data["VOTE"] == vote).to_numpy() * filter], sample_weight=wei[(data["VOTE"] == vote).to_numpy() * filter] if wei is not None else None)
    model = model.train()
    return eval_result


if __name__ == "__main__":
    config_path, config = parse_config()
    target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info, num_all_users = get_model_input(config)
    if config["model_type"] == "Transformer":
        model = TransformerVoter(config, categorical_features, string_features, original_feature_map, num_all_users=num_all_users)
    elif config["model_type"] == "linear":
        model = LinearModel(config, categorical_features, string_features, original_feature_map, num_all_users=num_all_users)
        
    train_weights = get_normalization_weights(train_data, config)
    next(train_model(config, model, train_data, weights = train_weights, batch_size=config['batch_size'], epochs=config['num_epochs'], verbose=2, validation_split=0.2))

    model_types = ["latest", "best"]
    for model_type in model_types:
        model, _, _, _, _ = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
        test_weights = get_normalization_weights(test_data, config)
        if config["use_voted_users_feature"]:
            debug("Use all voted users as feature")
        eval_all_test_data = evaluate_model(config, model, data = test_data, weights = test_weights, batch_size=config['batch_size'], sample_voted_users=False, data_info = test_data_info)
        debug(eval_all_test_data=str(eval_all_test_data))
        with open(config["log_path"], 'a') as log:
            log.write(f"Evaluation result of the {model_type} model (use all voted users as feature):" + str(eval_all_test_data)+"\n")

        if config["use_voted_users_feature"] and config["sample_part_voted_users"]:
            debug("Sample part voted users as feature")
            eval_all_test_data = evaluate_model(config, model, data = test_data, weights = test_weights, batch_size=config['batch_size'], sample_voted_users=True, data_info = test_data_info)
            debug(eval_all_test_data=str(eval_all_test_data))
            with open(config["log_path"], 'a') as log:
                log.write(f"Evaluation result of the {model_type} model (sample part voted users as feature):" + str(eval_all_test_data)+"\n")
    debug(config_path = config_path, log_path=config["log_path"])