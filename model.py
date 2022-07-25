import torch
from transformers import AutoModel, AutoTokenizer
from utils import load_model

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
        "Linear": deepctr_torch.models.dcn.LinearModel
    }
    return models[model_type]

def get_model(config, all_feature_columns, feature_names):
    CTRModel = get_ctr_model(config["model_type"])
    model = CTRModel(all_feature_columns, all_feature_columns, task='binary', device=config["device"], gpus = config["gpus"])
    if "UPVOTED_USERS" in feature_names:
        model.embedding_dict["UPVOTED_USERS"] = model.embedding_dict["USERNAME"]
    if "DOWNVOTED_USERS" in feature_names:
        model.embedding_dict["DOWNVOTED_USERS"] = model.embedding_dict["USERNAME"]
    if config["use_language_model_encoder"]:
        model.lm_encoder = AutoModel.from_pretrained(config["language_model_encoder_name"])
        # tokenizer = get_tokenizer(config)
        model.lm_encoder.resize_token_embeddings(64000)
    else:
        model.lm_encoder = None
    model.compile(torch.optim.Adam(model.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])
    return model

def get_best_model(config, all_feature_columns, feature_names):
    """
    CTRModel = get_ctr_model(config["model_type"])
    model = CTRModel(all_feature_columns, all_feature_columns, task='binary', device=config["device"], gpus = config["gpus"])
    model.compile(torch.optim.Adam(model.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])
    """
    model = get_model(config, all_feature_columns, feature_names)
    model, _, _, _, model_dict = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    assert model_dict is not None, "No trained model"
    state_dict = model_dict["state_dict"]
    if config["model_type"] == "MLR":
        user_embedding = torch.cat([state_dict[f"region_linear_model.{i}.embedding_dict.USERNAME.weight"] for i in range(20) if f"region_linear_model.{i}.embedding_dict.USERNAME.weight" in state_dict], dim = -1)
    else:
        user_embedding = state_dict[f"embedding_dict.USERNAME.weight"] #[num_users, user_embed_dim]
    return model, user_embedding.cpu()

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["language_model_encoder_name"], use_fast=False)
    return tokenizer

