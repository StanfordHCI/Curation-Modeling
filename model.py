import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, ElectraTokenizerFast, BertConfig, BertModel
from utils import load_model
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import *
from superdebug import debug


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

class GeneralModel(nn.Module):
    def __init__(self, config, categorical_features, string_features, original_feature_map, num_all_users = 0, task = "binary"):
        super(GeneralModel, self).__init__()
        self.device = config["device"]
        self.gpus = config["gpus"]
        self.config = config
        if self.gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")
        self.tokenizer = get_tokenizer(config, categorical_features, string_features, original_feature_map, config["use_voted_users_feature"], num_all_users)
        self.out = PredictionLayer(task, )
        self.regularization_weight = []


    def post_init(self):
        self.add_regularization_weight(self.parameters(), l2=self.config["l2_normalization"])
        self.to(self.device)
        self.compile(torch.optim.Adam(self.parameters(), lr = self.config["learning_rate"]), self.config["loss_function"], metrics=self.config["eval_metrics"])

    
    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

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
                def weighted_mse_loss(input, target, weight, reduction = "mean"):
                    if reduction == "mean":
                        return torch.mean((weight * (input - target) ** 2))
                    elif reduction == "sum":
                        return torch.sum((weight * (input - target) ** 2))
                loss_func = weighted_mse_loss
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


class TransformerVoter(GeneralModel):
    def __init__(self, config, categorical_features, string_features, original_feature_map, num_all_users = 0, task = "binary"):
        super(TransformerVoter, self).__init__(config, categorical_features, string_features, original_feature_map, num_all_users, task)
        if "custom" not in config["language_model_encoder_name"]:
            self.lm_encoder = AutoModel.from_pretrained(config["language_model_encoder_name"])
        else:
            lm_config = BertConfig(num_attention_heads=2, intermediate_size = config["encoder_hidden_dim"], hidden_size=config["encoder_hidden_dim"], num_hidden_layers=2)
            self.lm_encoder = BertModel(lm_config)
        self.lm_encoder.resize_token_embeddings(len(self.tokenizer))
        self.prediction_head = nn.Linear(config["encoder_hidden_dim"], 1)
        if 'USERNAME' in categorical_features:
            self.special_token_pos = 2
        else:
            self.special_token_pos = 0
        self.post_init()
    def forward(self, input_ids, token_type_ids, attention_mask):
        encoder_out = self.lm_encoder(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask).last_hidden_state
        target_hidden = encoder_out[:, self.special_token_pos, :] # [bsz, hidden_size], 2 is for the USER_i
        # encoder_out_pooled = (attention_mask[:,:,None] * encoder_out).sum(axis=1) / attention_mask.sum(axis = 1, keepdim = True)
        logit = self.prediction_head(target_hidden)
        return self.out(logit) # [bsz, 1]

class LinearModel(GeneralModel):
    def __init__(self, config, categorical_features, string_features, original_feature_map, num_all_users = 0, task = "binary"):
        super(LinearModel, self).__init__(config, categorical_features, string_features, original_feature_map, num_all_users, task)
        assert string_features == []
        self.embedding = nn.Embedding(len(self.tokenizer), config["encoder_hidden_dim"])
        # self.fcn = nn.Linear(len(categorical_features)*config["encoder_hidden_dim"], config["encoder_hidden_dim"])
        hidden_units = [len(categorical_features)*config["encoder_hidden_dim"], config["encoder_hidden_dim"], config["encoder_hidden_dim"]]
        self.dropout = nn.Dropout(0)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.activation_layers = nn.ModuleList(
            [nn.ReLU(inplace=True) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)




        self.prediction_head = nn.Linear(config["encoder_hidden_dim"], 1)
        self.post_init()
    def forward(self, input_ids, token_type_ids, attention_mask):
        token_embeddings = self.embedding(input_ids[:, 2::2])




        dnn_input = token_embeddings.reshape(token_embeddings.shape[0], -1)
        logit = torch.zeros([dnn_input.shape[0], 1]).to(self.device)
        # logit += torch.sum(dnn_input, dim=-1, keepdim = True)

        # target_hidden = self.fcn(dnn_input)
        # target_hidden = F.gelu(target_hidden)
        # logit = self.prediction_head(target_hidden)



        # deep_out = self.dnn(dnn_input)
        for i in range(len(self.linears)):

            fc = self.linears[i](dnn_input)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            dnn_input = fc
        logit += self.prediction_head(dnn_input)
        return self.out(logit) # [bsz, 1]



def _accuracy_score(y_true, y_pred, sample_weight=None):
    return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0), sample_weight=sample_weight)

def get_best_model(config, categorical_features, string_features, original_feature_map):
    model_device, model_gpus = config["device"], config["gpus"]
    config["device"], config["gpus"] = "cpu", None
    model = TransformerVoter(config, categorical_features, string_features, original_feature_map)
    model, _, _, _, model_dict = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    model.device, config["device"], model.gpus, config["gpus"] = model_device, model_device, model_gpus, model_gpus
    model.to(model_device)
    assert model_dict is not None, "No trained model"
    # state_dict = model_dict["state_dict"]
    if config["model_type"] == "Transformer":
        token_embedding = model.lm_encoder.embeddings.word_embeddings # TODO:
    return model, token_embedding.cpu()

def get_tokenizer(config, categorical_features, string_features, original_feature_map, use_voted_users_feature = False, num_all_users = 0):
    if "custom" not in config["language_model_encoder_name"]:
        tokenizer = AutoTokenizer.from_pretrained(config["language_model_encoder_name"], use_fast=True)
    else:
        debug("Initializing the tokenizer from sbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    
    debug(original_token_num = len(tokenizer))
    new_tokens = ["[SEP]"]
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

