from collections import Counter, defaultdict
import random
import warnings

from tqdm import tqdm 
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from superdebug import debug
import os
import pickle

from utils import get_config
def load_dataset(unit_test = False):
    vote_data = pd.read_csv('data/reddit/44_million_votes.txt', sep = '\t')
    vote_data['SUBMISSION_ID'] = vote_data['SUBMISSION_ID'].astype(str)
    debug(vote_data_num = len(vote_data))
    if unit_test: # sample 10% of the users, include all the voting data involving these users
        all_usernames = list(set(vote_data['USERNAME']))
        random.seed(42)
        selected_usernames = set(random.sample(all_usernames, k = int(0.2 * len(all_usernames))))
        vote_data = vote_data[vote_data['USERNAME'].isin(selected_usernames)] 
        debug(vote_data_num = len(vote_data))
    for vote in vote_data['VOTE']:
        assert vote == 'upvote' or vote == 'downvote', f'Vote {vote} is invalid'
    # SUBMISSION_ID SUBREDDIT    CREATED_TIME    USERNAME    VOTE
    # t3_e0i7l4	r/nagatoro		TeddehBear	upvote
    submission_data = pd.read_csv('data/reddit/submission_info.txt', sep = '\t')
    submission_data['SUBMISSION_ID'] = submission_data['SUBMISSION_ID'].astype(str)
    assert len(submission_data) == len(set(submission_data['SUBMISSION_ID'])), "submission ids should be unique" # each submission is a separate post and have a forest of comments
    submission_data['SUBREDDIT'] = ["r/" + subreddit for  subreddit in submission_data['SUBREDDIT']]
    submission_data['LINK'] = ["https://www.reddit.com" + link if link.startswith('/r/') else link for link in submission_data['LINK']]
    # SUBMISSION_ID	SUBREDDIT	TITLE	AUTHOR	#_COMMENTS	NSFW	SCORE	UPVOTED_%	LINK
    # t3_d8vv6s	japanpics	Gloomy day in Kyoto	DeanTheDoge	13		1303	0.98	https://www.reddit.com/r/japanpics/comments/d8vv6s/gloomy_day_in_kyoto/
    all_data = vote_data.merge(submission_data, on=['SUBMISSION_ID', 'SUBREDDIT'], how='inner')
    for vote in all_data['VOTE']:
        assert vote == 'upvote' or vote == 'downvote', f'Vote {vote} is invalid'
    return all_data
def get_selected_feature():
    sparse_features_embed_dims = {'USERNAME':64, 'SUBMISSION_ID':32, 'SUBREDDIT':8, 'AUTHOR':8, 'NSFW':4} # USERNAME is reader
    sparse_features = list(sparse_features_embed_dims.keys())
    dense_features = ['CREATED_TIME', '#_COMMENTS'] # , 'SCORE', 'UPVOTED_%' #! might should not use those
    target = ['VOTE']
    return sparse_features_embed_dims, sparse_features, dense_features, target

def clear_data(data, sparse_features, dense_features):
    data[sparse_features] = data[sparse_features].fillna('n/a')
    data[dense_features] = data[dense_features].fillna(0)

    return data

def transform_features(data, sparse_features, dense_features, target):
    original_feature_map = defaultdict(dict)
    for feature_name in sparse_features:
        lbe = LabelEncoder()
        original_features = data[feature_name]
        data[feature_name] = lbe.fit_transform(original_features)
        for i, transformed_feature in enumerate(data[feature_name]):
            if transformed_feature not in original_feature_map[feature_name]:
                original_feature_map[feature_name][transformed_feature] = original_features[i]

    lbe = LabelEncoder()
    data[target[0]] = lbe.fit_transform(data[target[0]])
    # dense numerical features -> [0,1]
    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    debug(data=data)
    return data, original_feature_map

def get_feature_columns(data, sparse_features, sparse_features_embed_dims, dense_features):
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=sparse_features_embed_dims[feat]) for i,feat in enumerate(sparse_features)] # count #unique features for each sparse field, transform sparse features into dense vectors by embedding techniques
    dense_feature_columns = [DenseFeat(feat, 1,) for feat in dense_features]
    debug(sparse_feature_columns=sparse_feature_columns, dense_feature_columns=dense_feature_columns)
    # fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,) for feat in dense_features]
    all_feature_columns = sparse_feature_columns + dense_feature_columns 
    # dnn_feature_columns = sparse_feature_columns + dense_feature_columns # For dense numerical features, we concatenate them to the input tensors of fully connected layer.
    # linear_feature_columns = sparse_feature_columns + dense_feature_columns

    feature_names = get_feature_names(all_feature_columns) # record feature field name
    debug(feature_names=feature_names)
    return all_feature_columns, feature_names

def generate_model_input(data:pd.DataFrame, feature_names, train_at_least_n_votes = 10):
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, test_data = [], []
    submission_votes = defaultdict(list)
    for row_i, row in tqdm(data.iterrows()):
        submission_votes[row["SUBMISSION_ID"]].append(row)
    random.seed(42)
    for submission_id in tqdm(list(submission_votes.keys())):
        votes_data = submission_votes[submission_id]
        if len(votes_data) > train_at_least_n_votes:
            train_i = random.sample(range(len(votes_data)), train_at_least_n_votes)
            for data_i, vote_data in enumerate(votes_data):
                if data_i in train_i:
                    train_data.append(vote_data)
                else:
                    test_data.append(vote_data)
    train_data = pd.DataFrame(train_data); test_data = pd.DataFrame(test_data)
    debug(train_vote_num = len(train_data), test_vote_num = len(test_data))
    train_model_input = {name:train_data[name] for name in feature_names}
    test_model_input = {name:test_data[name] for name in feature_names}
    return train_model_input, test_model_input, train_data, test_data

def get_model_input(config):
    prepared_data_path = config["prepared_data_path"]
    if not config["unit_test"]:
        assert "small" not in config["prepared_data_path"]
    if config["save_and_load_prepared_data"] and os.path.exists(prepared_data_path):
        debug("Loading prepared data...")
        with open(prepared_data_path, "rb") as f:
            all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, train_data, test_data = pickle.load(f)
    else:
        all_data = load_dataset(config["unit_test"])
        sparse_features_embed_dims, sparse_features, dense_features, target = get_selected_feature()
        cleared_data = clear_data(all_data, sparse_features, dense_features)
        featured_data, original_feature_map = transform_features(cleared_data, sparse_features, dense_features, target)
        all_feature_columns, feature_names = get_feature_columns(featured_data, sparse_features, sparse_features_embed_dims, dense_features)
        train_model_input, test_model_input, train_data, test_data = generate_model_input(featured_data, feature_names, train_at_least_n_votes = config["train_at_least_n_votes"])
        if config["save_and_load_prepared_data"]:
            with open(prepared_data_path, "wb") as f:
                pickle.dump((all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, train_data, test_data), f)
            debug(f"Prepared data saved to {prepared_data_path}")
    return all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, train_data, test_data

def analyze_data(train_data, test_data, original_feature_map):
    debug(train_upvote = sum(train_data["VOTE"] == 1),
        train_downvote = sum(train_data["VOTE"] == 0),
        test_upvote = sum(test_data["VOTE"] == 1),
        test_downvote = sum(test_data["VOTE"] == 0),
        )
    subreddit_votes = defaultdict(Counter)
    for row_i, row in train_data.iterrows():
        subreddit_votes[row["SUBREDDIT"]][row["VOTE"]] += 1
    for subreddit in subreddit_votes:
        subreddit_votes[subreddit]["downvote_rate"] = 100 * subreddit_votes[subreddit][0] / (subreddit_votes[subreddit][1] + subreddit_votes[subreddit][0])
        subreddit_votes[subreddit]["subreddit"] = original_feature_map["SUBREDDIT"][subreddit]
    subreddit_votes = pd.DataFrame(list(subreddit_votes.values())).set_index("subreddit")
    debug(subreddit_votes=subreddit_votes)
    subreddit_votes.to_csv("output/subreddit_votes.csv")

    
    """
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="day", y="total_bill", data=tips)
    ax.figure.savefig("data/reddit/output.png")
    """
if __name__ == '__main__':
    CONFIG_PATH = "configs/debug.yml"
    config, log_path = get_config(CONFIG_PATH) # default config
    all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, train_data, test_data = get_model_input(config)
    analyze_data(train_data, test_data, original_feature_map)


"""
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

dnn_feature_columns = sparse_feature_columns + dense_feature_columns # For dense numerical features, we concatenate them to the input tensors of fully connected layer.
linear_feature_columns = sparse_feature_columns + dense_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns) # record feature field name
debug(feature_names=feature_names)

#%% generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=2018)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}
debug(test_model_input=test_model_input)
"""

