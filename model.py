import torch
from transformers import AutoModel, AutoTokenizer

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

def get_model(config, all_feature_columns):
    CTRModel = get_ctr_model(config["model_type"])
    model = CTRModel(all_feature_columns, all_feature_columns, task='binary', device=config["device"], gpus = config["gpus"])
    if config["use_language_model_encoder"]:
        model.lm_encoder = AutoModel.from_pretrained(config["language_model_encoder_name"])
        # tokenizer = get_tokenizer(config)
        model.lm_encoder.resize_token_embeddings(64000)
    else:
        model.lm_encoder = None
    model.compile(torch.optim.Adam(model.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])
    return model

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["language_model_encoder_name"], use_fast=False)
    return tokenizer

