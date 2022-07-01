#%% import
import pandas as pd
import torch
from superdebug import debug
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models import DCN
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
CTRModel = DCN
#%% Import model
data = pd.read_csv('data/criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
debug(data)
target = ['label']

#%% Label Encoding: 
# discrete feature: map the features to integer value from 0 ~ len(#unique) - 1
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
debug(sparse_feat = data[sparse_features])
# dense numerical features -> [0,1]
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])
debug(dense_feat = data[dense_features])

#%% Generate feature columns
sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) for i,feat in enumerate(sparse_features)] # count #unique features for each sparse field, transform sparse features into dense vectors by embedding techniques
dense_feature_columns = [DenseFeat(feat, 1,) for feat in dense_features]
debug(sparse_feature_columns=sparse_feature_columns, dense_feature_columns=dense_feature_columns)
# fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,) for feat in dense_features]

# generate feature columns
dnn_feature_columns = sparse_feature_columns + dense_feature_columns # For dense numerical features, we concatenate them to the input tensors of fully connected layer.
linear_feature_columns = sparse_feature_columns + dense_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns) # record feature field name
debug(feature_names=feature_names)

#%% generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=2018)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}
debug(test_model_input=test_model_input)

#%% Define Model,train,predict and evaluate
if torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
model = CTRModel(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=20, verbose=2, validation_split=0.2)
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
# epoch: 40
# test LogLoss 0.8968
# test AUC 0.4271
# epoch: 30
# test LogLoss 0.7908
# test AUC 0.4375
# epoch: 20
# test LogLoss 0.6989
# test AUC 0.4349
# epoch: 10
# test LogLoss 0.6785
# test AUC 0.4219