import argparse
from collections import Counter, OrderedDict, defaultdict
import random
from typing import Union
import warnings
from matplotlib import pyplot as plt
import numpy as np

from tqdm import tqdm
from model import get_tokenizer

from reddit import get_batch_submission_text, get_single_submission_text 
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from superdebug import debug
import os
import pickle
from utils import get_config, parse_config

def find_correlated_user_pairs(vote_data, num_same_posts_thres):
    if os.path.exists("output/user_pair_agreement_level.pt"):
        user_pair_agreement_level = pickle.load(open("output/user_pair_agreement_level.pt", "rb"))
    else:
        usernames = vote_data["USERNAME"].to_numpy()
        user_votes_count = Counter(usernames)
        unique_users = {user for user in user_votes_count if user_votes_count[user] >= num_same_posts_thres}

        vote_data = vote_data[vote_data["USERNAME"].isin(unique_users)]
        submission_ids = vote_data["SUBMISSION_ID"].to_numpy()
        votes = vote_data["VOTE"].to_numpy()
        user_voted_posts = defaultdict(set)
        user_vote_on_posts = defaultdict(set)

        for row_i, username in enumerate(tqdm(usernames)):
            user_voted_posts[username].add(submission_ids[row_i])
            user_vote_on_posts[username].add(f"{submission_ids[row_i]}-{votes[row_i]}")

        user_pair_agreement_level = []
        unique_users = list(unique_users)
        for a_i, user_a in enumerate(tqdm(unique_users)):
            for user_b in unique_users[a_i + 1:]:
                if user_b != user_a:
                    num_same_posts = len(user_voted_posts[user_a] & user_voted_posts[user_b])
                    if num_same_posts >= num_same_posts_thres:
                        num_same_votes = len(user_vote_on_posts[user_a] & user_vote_on_posts[user_b])
                        user_pair_agreement_level.append(((user_a, user_b), abs(num_same_votes/num_same_posts - 0.5), num_same_posts))
        
        user_pair_agreement_level.sort(reverse=True, key=lambda x:x[1] * 10000000 + x[2])
        print(len(user_pair_agreement_level))
        pickle.dump(user_pair_agreement_level, open("output/user_pair_agreement_level.pt", "wb"))
    debug("Get correlated user pairs")
    
    correlated_user_pairs = []
    selected_users = set()
    for user_pair in user_pair_agreement_level:
        if user_pair[1] > 0.48:
            correlated_user_pairs.append(user_pair[0])
            selected_users.update(user_pair[0])
    return user_pair_agreement_level, correlated_user_pairs, selected_users

def sample_load_dataset(sample_ratio = 1, sample_method:Union[str, list] = 'USERNAME'):
    vote_data = pd.read_csv('data/reddit/44_million_votes.txt', sep = '\t')
    # SUBMISSION_ID SUBREDDIT    CREATED_TIME    USERNAME    VOTE
    # t3_e0i7l4	r/nagatoro		TeddehBear	upvote
    vote_data['SUBMISSION_ID'] = vote_data['SUBMISSION_ID'].astype(str)
    debug(vote_data_num = len(vote_data))

    if type(sample_method) == str:
        sample_method = [sample_method]
    selected_entries = None
    if sample_ratio < 1:
        if "most_votes" in sample_method:
            debug(f"Sampling {sample_ratio} of the most voted posts...")
            sample_column = "SUBMISSION_ID"
            submission_vote_num = Counter(vote_data["SUBMISSION_ID"])
            most_voted_submissions = list(submission_vote_num.keys())
            most_voted_submissions.sort(key = lambda x:submission_vote_num[x], reverse = True)
            selected_entries = set()
            total_vote_num = 0
            for submission_id in most_voted_submissions:
                selected_entries.add(submission_id)
                total_vote_num += submission_vote_num[submission_id]
                if total_vote_num >= sample_ratio * len(vote_data):
                    break
        elif "correlated_user_pairs" in sample_method:
            user_pair_agreement_level, correlated_user_pairs, selected_entries = find_correlated_user_pairs(vote_data, num_same_posts_thres=30)
            sample_column = "USERNAME"
        elif "USERNAME" in sample_method:
            sample_column = "USERNAME"
        elif "SUBMISSION_ID" in sample_method:
            sample_column = "SUBMISSION_ID"
        if "most_votes" not in sample_method:
            debug(f"Sampling {sample_ratio} of the {sample_column}s...")
        # sample x% of the users/submissions, include all the voting data involving these users/submissions
        if selected_entries is None:
            all_entries = set(vote_data[sample_column])
            random.seed(42)
            selected_entries = set(random.sample(list(all_entries), k = int(sample_ratio * len(all_entries))))
        vote_data = vote_data[vote_data[sample_column].isin(selected_entries)] 
        debug(vote_data_num = len(vote_data))

    # For each user, sample upvote:downvote = 1:1
    if "equal_up_down_votes" in sample_method:
        debug(f"Sampling so that upvote:downvote = 1:1")
        if sample_column == "USERNAME":
            selected_usernames = selected_entries
        else:
            selected_usernames = set(vote_data["USERNAME"])

        random.seed(42)
        user_votes_indices = defaultdict(list)
        usernames = vote_data["USERNAME"]
        votes = vote_data["VOTE"]
        for index in tqdm(vote_data.index): user_votes_indices[f'{usernames[index]}-{votes[index]}'].append(index)
        vote_data_indices = []
        for username in tqdm(selected_usernames):
            upvote_indices = user_votes_indices[f'{username}-upvote']
            downvote_indices = user_votes_indices[f'{username}-downvote']
            upvote_num = len(upvote_indices)
            downvote_num = len(downvote_indices)
            if downvote_num < upvote_num:
                upvote_indices = random.sample(upvote_indices, downvote_num)
            elif upvote_num < downvote_num:
                downvote_indices = random.sample(downvote_indices, upvote_num)
            vote_data_indices.extend(upvote_indices)
            vote_data_indices.extend(downvote_indices)
        vote_data_indices = set(vote_data_indices)
        vote_data = vote_data[vote_data.index.isin(vote_data_indices)]


    submission_data = pd.read_csv('data/reddit/submission_info.txt', sep = '\t') # each submission is a separate post and have a forest of comments
    # SUBMISSION_ID	SUBREDDIT	TITLE	AUTHOR	#_COMMENTS	NSFW	SCORE	UPVOTED_%	LINK
    # t3_d8vv6s	japanpics	Gloomy day in Kyoto	DeanTheDoge	13		1303	0.98	https://www.reddit.com/r/japanpics/comments/d8vv6s/gloomy_day_in_kyoto/

    # fix bugs in data
    submission_data['SUBMISSION_ID'] = submission_data['SUBMISSION_ID'].astype(str)
    assert len(submission_data) == len(set(submission_data['SUBMISSION_ID'])), "submission ids should be unique"
    submission_data['SUBREDDIT'] = ["r/" + subreddit for  subreddit in submission_data['SUBREDDIT']]
    submission_data['LINK'] = ["https://www.reddit.com" + link if link.startswith('/r/') else link for link in submission_data['LINK']]

    all_data = vote_data.merge(submission_data, on=['SUBMISSION_ID', 'SUBREDDIT'], how='inner')
    for vote in all_data['VOTE']:
        assert vote == 'upvote' or vote == 'downvote', f'Vote {vote} is invalid'
    return all_data

def get_selected_feature(config):
    #! do not use those: '#_COMMENTS', 'SCORE', 'UPVOTED_%' 
    categorical_features = config["categorical_features"]
    string_features = config["string_features"]
    target = ['VOTE']
    return categorical_features, string_features, target

def clean_data(data:pd.DataFrame, categorical_features, string_features):
    data["NSFW"] = data["NSFW"].map({"NSFW":"true", "": "false", np.nan: "false", None: "false"})
    categorical_features = [feat for feat in categorical_features if feat in data]
    string_features = [feat for feat in string_features if feat in data]
    data[categorical_features] = data[categorical_features].fillna('n/a')
    data[string_features] = data[string_features].fillna("")
    return data

def transform_features(data, categorical_features, string_features, target):
    original_feature_map = defaultdict(dict)
    for feature_name in categorical_features:
        if feature_name in data:
            lbe = LabelEncoder()
            original_features = data[feature_name]
            data[feature_name] = lbe.fit_transform(original_features)
            for i, transformed_feature in enumerate(data[feature_name]):
                if transformed_feature not in original_feature_map[feature_name]:
                    original_feature_map[feature_name][transformed_feature] = original_features[i]

    lbe = LabelEncoder()
    data[target[0]] = lbe.fit_transform(data[target[0]])
    # # dense numerical features -> [0,1]
    # mms = MinMaxScaler(feature_range=(0,1))
    # dense_features = [feat for feat in dense_features if feat in data]
    # data[dense_features] = mms.fit_transform(data[dense_features])
    return data, original_feature_map
"""
def get_feature_columns(data, sparse_features, sparse_features_embed_dims, varlen_sparse_features, varlen_sparse_features_embed_dims, dense_features):
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=(data[feat].nunique() if feat in data else 1),embedding_dim=sparse_features_embed_dims[feat]) for i,feat in enumerate(sparse_features)] # count #unique features for each sparse field, transform sparse features into dense vectors by embedding techniques
    max_voted_users = Counter(data["SUBMISSION_ID"]).most_common(1)[0][-1]
    varlen_sparse_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=data["USERNAME"].nunique()+1, embedding_dim=varlen_sparse_features_embed_dims[feat]), maxlen=max_voted_users, combiner='max') for i,feat in enumerate(varlen_sparse_features)]
    dense_feature_columns = [DenseFeat(feat, 1,) for feat in dense_features]
    debug(sparse_feature_columns=sparse_feature_columns, varlen_sparse_feature_columns=varlen_sparse_feature_columns, dense_feature_columns=dense_feature_columns)
    all_feature_columns = sparse_feature_columns + varlen_sparse_feature_columns + dense_feature_columns 
    feature_names = get_feature_names(all_feature_columns) # record feature field name
    debug(feature_names=feature_names)
    return all_feature_columns, feature_names, max_voted_users
"""
def divide_train_test_set(data:pd.DataFrame, train_at_least_n_votes = 0, train_test_different_submissions = False):
    random.seed(42)
    if train_at_least_n_votes == 0:
        data = data.sample(frac=1).reset_index(drop=True)
        if not train_test_different_submissions:
            # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            # debug(train_data=train_data, test_data=test_data)
            # assert len(set(test_data.index) & set(train_data.index)) == 0
            all_indices = list(data.index)
            test_indices = random.sample(all_indices, int(0.2 * len(all_indices)))
            test_data = data[data.index.isin(test_indices)]
            train_data = data[~data.index.isin(test_indices)]
        else:
            debug("Splitting train test set using different submissions")
            all_submissions = list(set(data["SUBMISSION_ID"]))
            test_submissions = random.sample(all_submissions, int(0.2 * len(all_submissions)))
            test_data = data[data["SUBMISSION_ID"].isin(test_submissions)]
            train_data = data[~data["SUBMISSION_ID"].isin(test_submissions)]
    else:
        train_data, test_data = [], []
        submission_votes = defaultdict(list)
        for row_i, row in tqdm(data.iterrows()):
            submission_votes[row["SUBMISSION_ID"]].append(row)
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
    if len(train_data) == 0:
        train_data = data.iloc[0:10]
    if len(test_data) == 0:
        test_data = data.iloc[-10:]
    debug(train_vote_num = len(train_data), test_vote_num = len(test_data))
    return train_data, test_data

def get_test_data_info(train_data:pd.DataFrame, test_data:pd.DataFrame):
    train_submission_votes_num = Counter(train_data["SUBMISSION_ID"])
    train_user_votes_num = Counter(train_data["USERNAME"])
    test_data_info = pd.DataFrame()
    test_data_info["train_submission_votes_num"] = test_data.apply(lambda row:train_submission_votes_num[row["SUBMISSION_ID"]], axis = 1)
    test_data_info["train_user_votes_num"] = test_data.apply(lambda row:train_user_votes_num[row["USERNAME"]], axis = 1)
    return test_data_info
def collect_users_votes_data(train_data:pd.DataFrame, test_data:pd.DataFrame):
    voted_users = defaultdict(set)
    for row_i, row in tqdm(train_data.iterrows()):
        voted_users[f'{row["SUBMISSION_ID"]}-{row["VOTE"]}'].add(row["USERNAME"])
    train_data["UPVOTED_USERS"] = train_data.apply(lambda r:list(voted_users[f'{r["SUBMISSION_ID"]}-1'] - {r["USERNAME"]}), axis=1)
    train_data["DOWNVOTED_USERS"] = train_data.apply(lambda r:list(voted_users[f'{r["SUBMISSION_ID"]}-0'] - {r["USERNAME"]}), axis=1)
    test_data["UPVOTED_USERS"] = test_data.apply(lambda r:list(voted_users[f'{r["SUBMISSION_ID"]}-1'] - {r["USERNAME"]}), axis=1)
    test_data["DOWNVOTED_USERS"] = test_data.apply(lambda r:list(voted_users[f'{r["SUBMISSION_ID"]}-0'] - {r["USERNAME"]}), axis=1)
    return train_data, test_data
    
"""
def tokenize_submission_text(train_data:pd.DataFrame, test_data:pd.DataFrame, train_model_input, test_model_input, config):
    if "SUBMISSION_TEXT" in train_data.columns:
        train_submission_text = list(train_data["SUBMISSION_TEXT"])
        test_submission_text = list(test_data["SUBMISSION_TEXT"])
        tokenizer = get_tokenizer(config)
        train_tokenized_submission_text = tokenizer(train_submission_text, padding=True, truncation=True, max_length=512, return_tensors="pt") # contains: input_ids, token_type_ids, attention_mask
        test_tokenized_submission_text = tokenizer(test_submission_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        train_model_input.update(train_tokenized_submission_text)
        test_model_input.update(test_tokenized_submission_text)
    return train_model_input, test_model_input
"""

def get_model_input(config):
    prepared_data_path = config["prepared_data_path"]
    if config["sample_ratio"] == 1:
        assert "small" not in config["prepared_data_path"]
    if config["save_and_load_prepared_data"] and os.path.exists(prepared_data_path):
        debug("Loading prepared data...")
        with open(prepared_data_path, "rb") as f:
            target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info = pickle.load(f)
    else:
        debug("Preparing data...")
        all_data = sample_load_dataset(config["sample_ratio"], config["sample_method"])
        all_data["SUBMISSION_TEXT"] = get_batch_submission_text(all_data['SUBMISSION_ID'])
        categorical_features, string_features, target = get_selected_feature(config)
        cleared_data = clean_data(all_data, categorical_features, string_features)
        featured_data, original_feature_map = transform_features(cleared_data, categorical_features, string_features, target)
        debug(featured_data=featured_data)
        # all_feature_columns, feature_names, max_voted_users = get_feature_columns(featured_data, sparse_features, sparse_features_embed_dims, varlen_sparse_features, varlen_sparse_features_embed_dims, dense_features)
        train_data, test_data = divide_train_test_set(featured_data, train_at_least_n_votes = config["train_at_least_n_votes"], train_test_different_submissions = config["train_test_different_submissions"])
        test_data_info = get_test_data_info(train_data, test_data)
        # train_model_input = {name:train_data[name] for name in feature_names if name in train_data}
        # test_model_input = {name:test_data[name] for name in feature_names if name in test_data}
        if config["use_voted_users_feature"]:
            train_data, test_data = collect_users_votes_data(train_data, test_data)
        # train_model_input, test_model_input = tokenize_submission_text(train_data, test_data, train_model_input, test_model_input, config)
        if config["save_and_load_prepared_data"]:
            with open(prepared_data_path, "wb") as f:
                pickle.dump((target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info), f)
            debug(f"Prepared data saved to {prepared_data_path}")
    return target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info

if __name__ == '__main__':
    config_path, config = parse_config()
    target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info = get_model_input(config)
    debug(config_path=config_path)