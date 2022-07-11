import datetime
import os
import shutil
import yaml
import torch
from superdebug import debug
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

def get_config(config_path, suffix="_train"):
    default_config = yaml.safe_load(open('default_config.yml', 'r'))
    custom_config = yaml.safe_load(open(config_path, 'r'))
    config = merge_dict(default_config, custom_config)
    config["save_model_dir"] = os.path.join("trained_models", os.path.basename(config_path).split(".")[0])
    os.makedirs(config["save_model_dir"], exist_ok=True)
    log_path = os.path.join(config["save_model_dir"], os.path.basename(config_path).split(".")[0]+suffix+".log")
    if os.path.exists(log_path): os.remove(log_path)
    shutil.copy(config_path, log_path)
    if config["device"] != -1 and torch.cuda.is_available():
        print('cuda ready...')
        config["gpus"] = config["device"] if type(config["device"]) == list else [config["device"]]
        config["gpus"] = [f"cuda:{device}" for device in config["gpus"]]
        config["device"] = config["gpus"][0]
    else:
        config["device"] = "cpu"
        config["gpus"] = None
    config["log_path"] = log_path
    now = datetime.datetime.now()
    with open(config["log_path"], 'a') as log:
        log.write("-----------------" + f"{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}" + "-----------------\n")
    
    return config
    
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
    # if lm_encoder is not None:
    #     save_dict['lm_encoder'] = lm_encoder.state_dict()
    if "lm_encoder" in " ".join(list(model.state_dict().keys())):
        debug("Model contains lm_encoder")
    else:
        debug("Model doesn't contain lm_encoder")
    torch.save(save_dict, save_path)

def load_model_dict(save_dir, type = "latest"):
    save_path = os.path.join(save_dir, f"{type}.pt")
    if os.path.exists(save_path):
        return torch.load(save_path)
    else:
        return None

def load_model(save_dir, model, optim, initial_epoch, best_eval_acc, type = "latest"):
    save_dict = load_model_dict(save_dir, type = type)
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
        return batch_func(lambda x:x.to(device), *params)
        # return (param.to(device) for param in params)
