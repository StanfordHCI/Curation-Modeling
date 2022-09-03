import argparse
from collections import Counter, defaultdict
import datetime
import os
import shutil
import yaml
import torch
from superdebug import debug
import wandb as wab
import pandas as pd
import sklearn.decomposition
from tqdm import tqdm
from scipy import sparse
use_debug = True

def merge_dict(main_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict):
            if key not in main_dict:
                main_dict[key] = {}
            merge_dict(main_dict[key], value)
        else:
            main_dict[key] = value
    return main_dict

def get_config(config_path, suffix="_train", wandb:wab = None, print_config = True):
    # load raw config
    default_config_path = 'default_config.yml'
    default_config = yaml.safe_load(open(default_config_path, 'r'))
    custom_config = yaml.safe_load(open(config_path, 'r'))
    if print_config:
        debug(custom_config=custom_config, config_path=config_path)
    config:dict = merge_dict(default_config, custom_config)

    experiment_name = os.path.basename(config_path).split(".")[0]
    now = datetime.datetime.now()

    # add to wandb
    if wandb is not None:
        wandb.init(project="curation_modeling", config = config, resume = config["load_pretrained_model"], save_code = True, name = experiment_name + f"-{now.year}/{now.month}/{now.day}_{now.hour}:{now.minute}")

    # process config
    config["save_model_dir"] = os.path.join("trained_models", experiment_name)
    os.makedirs(config["save_model_dir"], exist_ok=True)

    if suffix != "_train" and type(config["device"]) == list:
        config["device"] = -2
    if config["device"] != -1 and torch.cuda.is_available():
        print('GPU ready...')
        if config["device"] == -2:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            min_used_mem = 999999999999999
            config["device"] = 0
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if mem_info.used < min_used_mem:
                    min_used_mem = mem_info.used
                    config["device"] = i
            print(f'Smart using cuda:{config["device"]}')
        config["gpus"] = config["device"] if type(config["device"]) == list else [config["device"]]
        config["gpus"] = [f"cuda:{device}" for device in config["gpus"]]
        config["device"] = config["gpus"][0]
    else:
        if torch.cuda.is_available():
            print("GPU detected yet using CPU...")
        config["device"] = "cpu"
        config["gpus"] = None
    
    # write to log
    log_path = os.path.join(config["save_model_dir"], experiment_name+suffix+".log")
    config["log_path"] = log_path
    if os.path.exists(log_path): os.remove(log_path)
    shutil.copy(config_path, log_path)
    with open(log_path, 'a') as log:
        log.write("\n----------------- Below is default config -----------------\n")
        log.write(open(default_config_path, 'r').read())
    with open(log_path, 'a') as log:
        log.write("\n-----------------" + f"{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}" + "-----------------\n")
    
    return config
    
def parse_config(wandb = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to custom config file.")
    parser.add_argument("--test", default=False, action="store_true", help="Whether to only test")
    args = parser.parse_args()
    config = get_config(args.config, wandb=wandb)
    return args, config

def join_sets(sets):
    full_set = set()
    for a_set in sets:
        full_set.update(a_set)
    return full_set
    
def save_model(model, epoch, eval_acc, optim, save_dir, type = "latest"):
    save_path = os.path.join(save_dir, f"{type}.pt")
    save_dict = {'epoch': epoch, 
		'state_dict': model.state_dict(), 
		'eval_acc': eval_acc,
		'optimizer': optim.state_dict()}
    torch.save(save_dict, save_path)

def load_model_dict(save_dir, type = "latest", device = "cpu"):
    save_path = os.path.join(save_dir, f"{type}.pt")
    if os.path.exists(save_path):
        return torch.load(save_path, map_location = device)
    else:
        return None

def load_model(save_dir, model, optim, initial_epoch, best_eval_acc, type = "latest"):
    debug(f"Loading {type} model...")
    save_dict = load_model_dict(save_dir, type = type, device = model.device)
    if save_dict:
        model.load_state_dict(save_dict['state_dict'])
        # if lm_encoder is not None and 'lm_encoder' in save_dict:
        #     lm_encoder.load_state_dict(save_dict['lm_encoder'])
        optim.load_state_dict(save_dict['optimizer'])
        initial_epoch = save_dict['epoch']
        best_eval_acc = save_dict['eval_acc']
    return model, optim, initial_epoch, best_eval_acc, save_dict

def print_log(log_path, *strs, **strss):
    if use_debug:
        debug(*strs, **strss)
    else:
        print(*strs, **strss)
    strs = [str(x) for x in strs] + [f"{x}:{strss[x]};\t" for x in strss]
    with open(log_path, "a") as f:
        f.write(" ".join(strs) + "\n")

def batch_func(func, *args):
    return (func(arg) for arg in args)
def to_device(device, to_float, *params):
    if to_float:
        return batch_func(lambda x:x.to(device).float(), *params)
        # return (param.to(device).float() for param in params)
    else:
        return batch_func(lambda x:x.to(device) if x is not None else x, *params)
        # return (param.to(device) for param in params)


def get_bool_vec(selected_ids, vec_size):
    bool_vec = torch.zeros([vec_size], dtype = bool)
    for user in selected_ids:
        bool_vec[user] = True
    return bool_vec

def get_user_reps(selected_users, all_user_embedding, train_data:pd.DataFrame = None, selected_submissions = None, user_grouping_method = "neural", do_PCA = True):
    assert all_user_embedding is not None
    selected_users_bool_vec = get_bool_vec(selected_users, all_user_embedding.shape[0])
    # user_user_i_map = {}
    selected_user_i_user_map = {}
    selected_user_user_i_map = {}
    user_i = 0
    for user, is_selected in enumerate(selected_users_bool_vec):
        if is_selected:
            # user_user_i_map[user] = user_i
            selected_user_i_user_map[user_i] = user
            selected_user_user_i_map[user] = user_i
            user_i += 1
    # assert len(user_user_i_map) == len(user_i_user_map)
    selected_users_reps = None
    if "neural" in user_grouping_method:
        selected_users_reps = all_user_embedding[selected_users_bool_vec, :]
    elif "votes" in user_grouping_method:
        assert train_data is not None and selected_submissions is not None
        sub_sub_i_map = {sub: sub_i for sub_i, sub in enumerate(list(selected_submissions.keys()))}
        # selected_users_reps = torch.zeros([len(selected_user_user_i_map), len(selected_submissions)])
        train_users = train_data["USERNAME"].to_list()
        train_submission_ids = train_data["SUBMISSION_ID"].to_list()
        train_votes = train_data["VOTE"].to_list()
        i_list, j_list, vote_list = [], [], []
        for submission_id, username, vote in tqdm(zip(train_submission_ids, train_users, train_votes)):
            if username in selected_user_user_i_map and submission_id in sub_sub_i_map:
                vote = 1 if vote == 1 else -1
                i_list.append(selected_user_user_i_map[username])
                j_list.append(sub_sub_i_map[submission_id])
                vote_list.append(vote)
                # selected_users_reps[selected_user_user_i_map[username], sub_sub_i_map[submission_id]] = vote
        selected_users_reps = sparse.coo_matrix((vote_list, (i_list, j_list)), shape = [len(selected_user_user_i_map), len(selected_submissions)])
        # users_vote_sum = (selected_users_reps * selected_users_reps).sum(axis = -1, keepdim= True)
        # assert (users_vote_sum != 0).any()
        # selected_users_reps = selected_users_reps / users_vote_sum # average votes on each submission
        if do_PCA:
            selected_users_reps = selected_users_reps.todense()
            debug(selected_users_reps_before_PCA=selected_users_reps.shape)
            if selected_users_reps.shape[1] > 10000:
                pca_solver = sklearn.decomposition.PCA(n_components=1000)
            else:
                pca_solver = sklearn.decomposition.PCA(n_components=0.95)
            selected_users_reps = pca_solver.fit_transform(selected_users_reps)
            debug(selected_users_reps_after_PCA = selected_users_reps.shape)

    return selected_users_reps, selected_user_i_user_map

def record_existing_votes(train_data:pd.DataFrame):
    # collect existing votes
    existing_votes = {}
    existing_user_updown_votes = defaultdict(Counter)
    existing_user_votes = Counter()
    existing_submission_votes = defaultdict(Counter)
    existing_user_subreddits = defaultdict(set)
    usernames = train_data["USERNAME"].to_list()
    subreddits = train_data["SUBREDDIT"].to_list()
    sub_ids = train_data["SUBMISSION_ID"].to_list()
    votes = train_data["VOTE"].to_list()
    for row_i in range(len(train_data)):
        existing_votes[f'{usernames[row_i]}-{sub_ids[row_i]}'] = votes[row_i]
        existing_user_updown_votes[usernames[row_i]][votes[row_i]] += 1
        existing_user_votes[usernames[row_i]] += 1
        existing_submission_votes[sub_ids[row_i]][votes[row_i]] += 1
        existing_user_subreddits[usernames[row_i]].add(subreddits[row_i])
    return existing_votes, existing_user_votes, existing_user_updown_votes, existing_submission_votes, existing_user_subreddits