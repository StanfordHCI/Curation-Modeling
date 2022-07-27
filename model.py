import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, ElectraTokenizerFast
from utils import load_model
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import *
from superdebug import debug

"""
class PredictionLayer(nn.Module):
    '''
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    '''

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output
"""

class TransformerVoter(nn.Module):
    def __init__(self, config, categorical_features, string_features, original_feature_map, num_all_users = 0, task = "binary"):
        super(TransformerVoter, self).__init__()
        self.device = config["device"]
        self.gpus = config["gpus"]
        if self.gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")
        self.lm_encoder = AutoModel.from_pretrained(config["language_model_encoder_name"])
        self.tokenizer = get_tokenizer(config, categorical_features, string_features, original_feature_map, config["use_voted_users_feature"], num_all_users)
        self.lm_encoder.resize_token_embeddings(len(self.tokenizer))
        self.prediction_head = nn.Linear(config["encoder_hidden_dim"], 1)
        self.to(self.device)
        self.compile(torch.optim.Adam(self.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoder_hidden = self.lm_encoder(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask).last_hidden_state
        # encoder_hidden_pooled = (attention_mask[:,:,None] * encoder_hidden).sum(axis=1) / attention_mask.sum(axis = 1, keepdim = True)
        target_user_hidden = encoder_hidden[:, 2, :] # [bsz, hidden_size], 2 is for the USER_i
        logits = self.prediction_head(target_user_hidden)
        return torch.sigmoid(logits) # [bsz, 1]
    def compile(self, optimizer, loss=None, metrics=None):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)


    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = _accuracy_score
                self.metrics_names.append(metric)
        return metrics_
def _accuracy_score(y_true, y_pred, sample_weight=None):
    return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0), sample_weight=sample_weight)

def get_best_model(config, all_feature_columns, feature_names):
    """
    CTRModel = get_ctr_model(config["model_type"])
    model = CTRModel(all_feature_columns, all_feature_columns, task='binary', device=config["device"], gpus = config["gpus"])
    model.compile(torch.optim.Adam(model.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])
    """
    model = TransformerVoter(config, all_feature_columns, feature_names)
    model, _, _, _, model_dict = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    assert model_dict is not None, "No trained model"
    state_dict = model_dict["state_dict"]
    if config["model_type"] == "MLR":
        user_embedding = torch.cat([state_dict[f"region_linear_model.{i}.embedding_dict.USERNAME.weight"] for i in range(20) if f"region_linear_model.{i}.embedding_dict.USERNAME.weight" in state_dict], dim = -1)
    else:
        user_embedding = state_dict[f"embedding_dict.USERNAME.weight"] #[num_users, user_embed_dim]
    return model, user_embedding.cpu()

def get_tokenizer(config, categorical_features, string_features, original_feature_map, use_voted_users_feature = False, num_all_users = 0):
    tokenizer = AutoTokenizer.from_pretrained(config["language_model_encoder_name"], use_fast=True)
    
    debug(original_token_num = len(tokenizer))
    new_tokens = []
    for feat in categorical_features + string_features:
        new_tokens.append(f"[{feat}]")
    for feat in original_feature_map:
        feature_space = len(original_feature_map[feat])
        for i in range(feature_space + 1):
            new_tokens.append(f"{feat}_{i}")
    if use_voted_users_feature:
        new_tokens.extend(["[UPVOTED_USERS]", "[DOWNVOTED_USERS]"])
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens[:25000]})
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens[25000:]})
    debug(latest_token_num = len(tokenizer))
    return tokenizer

