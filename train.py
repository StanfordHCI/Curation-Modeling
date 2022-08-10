import argparse
from collections import Counter, OrderedDict
import datetime
import os
import time
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
import torch
from superdebug import debug
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from dynamic_data import get_data_loader
from process_data import get_model_input, get_test_data_info
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
import wandb
import seaborn as sns


def get_normalization_weights(data:pd.DataFrame, train_submission_upvote_df:pd.DataFrame, config):
    # user_weights
    if not config["user_normalization"]:
        user_weights = np.ones([len(data)])
    else:
        if config["user_normalization"] == "equal_total":
            user_column = data["USERNAME"]
        elif config["user_normalization"] == "equal_upvote_downvote":
            user_column = data.apply(lambda row:f"{row['USERNAME']}-{row['VOTE']}", axis=1)
        user_column = user_column.to_list()
        user_votes_counter = Counter(user_column)
        user_weights = np.array([100/user_votes_counter[x] for x in user_column])
        
    # minority_weight, more same votes -> less weight
    if config["minority_vote_normalization"]:
        data = data.merge(train_submission_upvote_df[["SUBMISSION_ID", "Upvote rate"]], on = "SUBMISSION_ID", how = "left")
        data["same_vote_rate"] = data["Upvote rate"]
        data.loc[data["VOTE"] == 0, "same_vote_rate"] = 1 - data.loc[data["VOTE"] == 0, "same_vote_rate"]
        minority_weight = 1/data["same_vote_rate"].to_numpy()
        minority_weight[minority_weight > 10] = 10
    else:
        minority_weight = np.ones([len(data)])
        
    # upvote_downvote_weights
    upvote_downvote_weights = np.array(1 * (data["VOTE"] != 0) + config["downvote_weight"] * (data["VOTE"] == 0))
    
    normalization_weights = upvote_downvote_weights * user_weights * minority_weight
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

categorical_features, string_features, target = None, None, None
def train_model(config, model, data:pd.DataFrame, weights=None, batch_size=256, epochs=1, verbose=1, initial_epoch=0, validation_split=0., shuffle=True, step_generator = False, n_step_per_sample = 1, extra_input = None):
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
    trainset, train_loader = get_data_loader(config, data, model.tokenizer, (categorical_features if extra_input is None else extra_input[0]), (string_features if extra_input is None else extra_input[1]), (target if extra_input is None else extra_input[2]), weight=weights, sample_voted_users = config["sample_part_voted_users"], add_target_user_ratio = config["add_target_user_ratio"], shuffle=shuffle, batch_size=batch_size)

    model = model.train()
    optim = model.optim
    best_eval_acc, best_eval_acc_weight = 0, 0
    if config["load_pretrained_model"]:
        model, optim, initial_epoch, best_eval_acc_weight, save_dict = load_model(config["save_model_dir"], model, optim, initial_epoch, best_eval_acc_weight)
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
        else:
            train_loader_tqdm = train_loader
        for _, train_input in enumerate(train_loader_tqdm):
            # x, y, weight = convert_CTR_model_input(model, train_input, sample_voted_users=config["sample_part_voted_users"], add_target_user_ratio = config["add_target_user_ratio"])
            input_ids, token_type_ids, attention_mask, label, weight, df_index = train_input
            # x, _y, weight = convert_CTR_model_input(model, test_input, sample_voted_users=sample_voted_users)
            input_ids, token_type_ids, attention_mask, label, weight = to_device(model.device, False, input_ids, token_type_ids, attention_mask, label.float(), weight)
            for _step_i in range(n_step_per_sample):
                y_pred = _model(input_ids, token_type_ids, attention_mask)

                optim.zero_grad()
                loss = model.loss_func(y_pred.reshape(label.shape), label, weight = weight.reshape(label.shape), reduction='sum')
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
                    metric_result = apply_metric(metric_func, (label.cpu().numpy() > 0.5).astype(int), y_pred.reshape(label.shape).cpu().data.numpy())
                    if metric_result is not None: train_result[name].append(metric_result)
            if step_generator:
                yield model

        if not step_generator:
            if do_validation:
                eval_result = evaluate_model(config, model, val_data, weights = val_weights, batch_size=batch_size)
                best_eval_acc_weight = max(best_eval_acc_weight, eval_result["acc_with_weight"])
                best_eval_acc = max(best_eval_acc, eval_result["acc"])
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                train_loss = total_loss_epoch / sample_num
                train_acc = np.sum(train_result["acc"]) / len(train_loader)
                wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": eval_result["acc"], "val_acc_vote_0": eval_result["acc_vote_0"], "val_acc_vote_1": eval_result["acc_vote_1"], "val_acc_with_weight": eval_result["acc_with_weight"], "epoch": epoch + 1})
                eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, train_loss)
                for name, result in train_result.items():
                    eval_str += " - " + name + ": {0: .4f}".format(np.sum(result) / len(train_loader))
                if do_validation:
                    for name, result in eval_result.items():
                        eval_str += " - " + "val_" + name + (": {0: .4f}".format(result) if result is not None else ": N/A")
                print(eval_str)
                with open(config["log_path"], 'a') as log:
                    log.write(eval_str+"\n")
            save_model(model, epoch, eval_result["acc_with_weight"], optim, config["save_model_dir"], "latest")
            if best_eval_acc_weight == eval_result["acc_with_weight"]:
                save_model(model, epoch, eval_result["acc_with_weight"], optim, config["save_model_dir"], "best")
    if step_generator:
        yield None
    else:
        yield best_eval_acc, best_eval_acc_weight, eval_result["acc"], eval_result["acc_with_weight"], train_loss, train_acc


def evaluate_model(config, model, data:pd.DataFrame, weights = None, batch_size=256, ret = "eval_result", sample_voted_users = False, data_info:pd.DataFrame = None, disable_tqdm = False, extra_input = None, simple = True):
    model = model.eval()
    testset, test_loader = get_data_loader(config, data, model.tokenizer, (categorical_features if extra_input is None else extra_input[0]), (string_features if extra_input is None else extra_input[1]), (target if extra_input is None else extra_input[2]), weights, sample_voted_users=sample_voted_users, add_target_user_ratio = 0, shuffle=False, batch_size=batch_size)
    pred_ans = []

    if not disable_tqdm:
        test_loader = tqdm(test_loader)
    with torch.no_grad():
        for _, test_input in enumerate(test_loader):
            input_ids, token_type_ids, attention_mask, label, weight, df_index = test_input
            input_ids, token_type_ids, attention_mask, label, weight = to_device(model.device, False, input_ids, token_type_ids, attention_mask, label.float(), weight)
            y_pred = model(input_ids, token_type_ids, attention_mask).cpu().data.numpy()  # .squeeze()
            pred_ans.append(y_pred)

    pred_ans =  np.concatenate(pred_ans).astype("float64")
    if ret == "prediction":
        return pred_ans

    if simple or data_info is None:
        test_filter = {"": (data["VOTE"] >=-1).to_numpy()}
    else:
        if "same_vote_rate" not in data_info.columns:
            data_info, train_submission_upvote_df = get_test_data_info(train_data, test_data)
        test_filter = {"": (data["VOTE"] >=-1).to_numpy(), "_train_user_votes_num>=3": (data_info["train_user_votes_num"] >= 3).to_numpy(), "_train_submission_votes_num>=3": (data_info["train_submission_votes_num"] >= 3).to_numpy(), "_train_user_votes_num<=3": (data_info["train_user_votes_num"] <= 3).to_numpy(), "_train_submission_votes_num<=3": (data_info["train_submission_votes_num"] <= 3).to_numpy()} 

    # calculate all evaluation results
    ground_truth = (data["VOTE"].to_numpy() > 0.5).astype(float)
    eval_result = OrderedDict()
    for wei in [None, weights]:
        for filter_name in test_filter:
            filter = test_filter[filter_name]
            for name, metric_func in model.metrics.items():
                if 0 not in ground_truth[filter].shape:
                    eval_result[f"{name}{filter_name}{'_with_weight' if wei is not None else ''}"] = apply_metric(metric_func, ground_truth[filter], pred_ans[filter], sample_weight=wei[filter] if wei is not None else None)
                if data is not None:
                    for vote in [0, 1]:
                        if 0 not in ground_truth[(data["VOTE"] == vote).to_numpy() * filter].shape:
                            eval_result[f"{name}_vote_{vote}{filter_name}{'_with_weight' if wei is not None else ''}"] = apply_metric(metric_func, ground_truth[(data["VOTE"] == vote).to_numpy() * filter], pred_ans[(data["VOTE"] == vote).to_numpy() * filter], sample_weight=wei[(data["VOTE"] == vote).to_numpy() * filter] if wei is not None else None)

    # draw #votes for a user, #votes for a submission <-> acc, confidence curve
    if (not simple) and (data_info is not None):
        train_user_votes_nums = data_info["train_user_votes_num"].to_numpy()
        train_submission_votes_nums = data_info["train_submission_votes_num"].to_numpy()
        data_info["same_vote_rate"] = data_info["same_vote_rate"].fillna(-1)
        train_same_vote_rates = (data_info["same_vote_rate"].round(2).to_numpy() * 100).astype(int)
        train_user_votes_num_acc_df = pd.DataFrame(np.zeros((max(train_user_votes_nums) + 1,3)), columns=['Acc', 'Confidence', 'Total'])
        train_submission_votes_num_acc_df = pd.DataFrame(np.zeros((max(train_submission_votes_nums) + 1,3)), columns=['Acc', 'Confidence', 'Total'])
        train_same_vote_rate_acc_df = pd.DataFrame(np.zeros((101,3)), columns=['Acc', 'Confidence', 'Total'])

        
        for vote_i, pred_score in enumerate(pred_ans):
            pred_vote = float(pred_score > 0.5)
            gt_vote = ground_truth[vote_i]
            train_user_votes_num = train_user_votes_nums[vote_i]
            train_submission_votes_num = train_submission_votes_nums[vote_i] 
            train_same_vote_rate = train_same_vote_rates[vote_i]

            train_submission_votes_num_acc_df.at[train_submission_votes_num, "Acc"] += int(pred_vote == gt_vote)
            train_submission_votes_num_acc_df.at[train_submission_votes_num, "Confidence"] += abs(pred_score - 0.5)
            train_submission_votes_num_acc_df.at[train_submission_votes_num, "Total"] += 1

            train_user_votes_num_acc_df.at[train_user_votes_num, "Acc"] += int(pred_vote == gt_vote)
            train_user_votes_num_acc_df.at[train_user_votes_num, "Confidence"] += abs(pred_score - 0.5)
            train_user_votes_num_acc_df.at[train_user_votes_num, "Total"] += 1
            
            if train_same_vote_rate >= 0:
                train_same_vote_rate_acc_df.at[train_same_vote_rate, "Acc"] += int(pred_vote == gt_vote)
                train_same_vote_rate_acc_df.at[train_same_vote_rate, "Confidence"] += abs(pred_score - 0.5)
                train_same_vote_rate_acc_df.at[train_same_vote_rate, "Total"] += 1
        
        train_submission_votes_num_acc_df = train_submission_votes_num_acc_df[train_submission_votes_num_acc_df["Total"] > 0]
        train_submission_votes_num_acc_df["Acc rate"] = train_submission_votes_num_acc_df["Acc"]/train_submission_votes_num_acc_df["Total"]
        train_submission_votes_num_acc_df["Avg confidence"] = train_submission_votes_num_acc_df["Confidence"]/train_submission_votes_num_acc_df["Total"]
        train_submission_votes_num_acc_df["Total scaled"] = train_submission_votes_num_acc_df["Total"]/len(pred_ans)
        sns.set_theme()
        sns.lineplot(data=train_submission_votes_num_acc_df[["Acc rate", "Avg confidence", "Total scaled"]], legend = "auto").set(title='Accuracy & confidence given different #votes on this post')
        plt.show()

        train_user_votes_num_acc_df = train_user_votes_num_acc_df[train_user_votes_num_acc_df["Total"] > 0]
        train_user_votes_num_acc_df["Acc rate"] = train_user_votes_num_acc_df["Acc"]/train_user_votes_num_acc_df["Total"]
        train_user_votes_num_acc_df["Avg confidence"] = train_user_votes_num_acc_df["Confidence"]/train_user_votes_num_acc_df["Total"]
        train_user_votes_num_acc_df["Total scaled"] = train_user_votes_num_acc_df["Total"]/len(pred_ans)
        sns.set_theme()
        sns.lineplot(data=train_user_votes_num_acc_df[["Acc rate", "Avg confidence", "Total scaled"]], legend = "auto").set(title='Accuracy & confidence given different #votes from this user')
        plt.show()
        
        train_same_vote_rate_acc_df = train_same_vote_rate_acc_df[train_same_vote_rate_acc_df["Total"] > 0]
        train_same_vote_rate_acc_df["Acc rate"] = train_same_vote_rate_acc_df["Acc"]/train_same_vote_rate_acc_df["Total"]
        train_same_vote_rate_acc_df["Avg confidence"] = train_same_vote_rate_acc_df["Confidence"]/train_same_vote_rate_acc_df["Total"]
        train_same_vote_rate_acc_df["Total scaled"] = train_same_vote_rate_acc_df["Total"]/len(pred_ans)
        sns.set_theme()
        sns.lineplot(data=train_same_vote_rate_acc_df[["Acc rate", "Avg confidence", "Total scaled"]], legend = "auto").set(title='Accuracy & confidence given different %votes that is same as the target vote')
        plt.show()

    model = model.train()
    if ret == "eval_result":
        return eval_result
    elif ret == "eval_result_prediction":
        return eval_result, pred_ans


if __name__ == "__main__":
    args, config = parse_config(wandb)
    target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info, train_submission_upvote_df, num_all_users = get_model_input(config)

    if not args.test:
        if config["model_type"] == "Transformer":
            model = TransformerVoter(config, categorical_features, string_features, original_feature_map, num_all_users=num_all_users)
        elif config["model_type"] == "linear":
            model = LinearModel(config, categorical_features, string_features, original_feature_map, num_all_users=num_all_users)
            
        train_weights = get_normalization_weights(train_data, train_submission_upvote_df, config)
        best_eval_acc, best_eval_acc_weight, latest_eval_acc, latest_eval_acc_weight, train_loss, train_acc = next(train_model(config, model, train_data, weights = train_weights, batch_size=config['batch_size'], epochs=config['num_epochs'], verbose=2, validation_split=0.2))
        wandb.alert(
            title="Finished training!", 
            text=f"best_eval_acc: {best_eval_acc}, best_eval_acc_weight: {best_eval_acc_weight}, latest_eval_acc: {latest_eval_acc}, latest_eval_acc_weight: {latest_eval_acc_weight}, train_loss: {train_loss}, train_acc: {train_acc}"
        )

    model_type = "best"
    model, _, _, _, _ = load_model(config["save_model_dir"], model, model.optim, 0, 0, model_type)
    test_weights = get_normalization_weights(test_data, train_submission_upvote_df, config)
    if config["use_voted_users_feature"]: debug("Use all voted users as feature")
    eval_all_test_data = evaluate_model(config, model, data = test_data, weights = test_weights, batch_size=config['batch_size'], sample_voted_users=False, data_info = test_data_info)
    eval_result_str = "".join([f"- {key}: {value:.4f} " for key, value in eval_all_test_data.items()])
    debug(eval_all_test_data=str(eval_result_str))
    wandb_log = {"train_loss": train_loss, "train_acc": train_acc}
    wandb_log.update({"test_" + key: value for key, value in eval_all_test_data.items()})
    wandb.log(wandb_log)
    with open(config["log_path"], 'a') as log:
        log.write(f"Evaluation result of the {model_type} model (use all voted users as feature):" + eval_result_str +"\n")