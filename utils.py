import os
import shutil
import yaml
import torch
def merge_dict(main_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict):
            if key not in main_dict:
                main_dict[key] = {}
            merge_dict(main_dict[key], value)
        else:
            main_dict[key] = value
    return main_dict

def get_config(config_path = None):
    default_config = yaml.safe_load(open('default_config.yml', 'r'))
    custom_config = yaml.safe_load(open(config_path, 'r'))
    config = merge_dict(default_config, custom_config)
    config["save_model_dir"] = os.path.join("trained_models", os.path.basename(config_path).split(".")[0])
    os.makedirs(config["save_model_dir"], exist_ok=True)
    log_path = os.path.join(config["save_model_dir"], os.path.basename(config_path).split(".")[0]+".log")
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
    return config, log_path
    
def join_sets(sets):
    full_set = set()
    for a_set in sets:
        full_set.update(a_set)
    return full_set
    

def save_model(model, epoch, eval_acc, optim, save_dir, type = "latest"):
    save_path = os.path.join(save_dir, f"{type}.pt")
    torch.save({'epoch': epoch, 
		'state_dict': model.state_dict(), 
		'eval_acc': eval_acc,
		'optimizer': optim.state_dict()},
    save_path)

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
        optim.load_state_dict(save_dict['optimizer'])
        initial_epoch = save_dict['epoch']
        best_eval_acc = save_dict['eval_acc']
    return model, optim, initial_epoch, best_eval_acc, save_dict

def get_ctr_model(model_type):
    import deepctr_torch.models
    models = {
        "WDL": deepctr_torch.models.WDL,
        "DeepFM": deepctr_torch.models.DeepFM,
        "xDeepFM": deepctr_torch.models.xDeepFM,
        "AFM": deepctr_torch.models.AFM,
        "DIFM": deepctr_torch.models.DIFM,
        "IFM": deepctr_torch.models.IFM,
        "AutoInt": deepctr_torch.models.AutoInt,
        "DCN": deepctr_torch.models.DCN,
        "DCNMix": deepctr_torch.models.DCNMix,
        "FiBiNET": deepctr_torch.models.FiBiNET,
        "NFM": deepctr_torch.models.NFM,
        "MLR": deepctr_torch.models.MLR,
        "ONN": deepctr_torch.models.ONN,
        "PNN": deepctr_torch.models.PNN,
        "CCPM": deepctr_torch.models.CCPM,
        "DIEN": deepctr_torch.models.DIEN,
        "DIN": deepctr_torch.models.DIN,
        "AFN": deepctr_torch.models.AFN,
    }
    return models[model_type]