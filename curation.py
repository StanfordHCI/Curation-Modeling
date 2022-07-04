from collections import Counter, defaultdict
import random
import pandas as pd
from superdebug import debug
import torch
from data import get_model_input
from utils import get_config, get_ctr_model, load_model, save_model, load_model_dict
from sklearn import cluster

CONFIG_PATH = "configs/debug.yml"
config, log_path = get_config(CONFIG_PATH)

def get_best_model():
    from main import model
    model, _, _, _ = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    return model

def get_best_model_embedding():
    model_dict = load_model_dict(config["save_model_dir"], type = "best")
    assert model_dict is not None, "No trained model"
    state_dict = model_dict["state_dict"]
    user_embedding = state_dict[f"embedding_dict.USERNAME.weight"] #[num_users, user_embed_dim]
    return user_embedding.cpu()

def get_popular_subreddits(test_data):
    subreddit_votes_counter = Counter()
    subreddit_users = defaultdict(set)
    subreddit_unique_submissions = defaultdict(dict)
    for i, row in test_data.iterrows():
        subreddit_votes_counter[row["SUBREDDIT"]] += 1
        subreddit_users[row["SUBREDDIT"]].add(row["USERNAME"])
        if row["SUBMISSION_ID"] not in subreddit_unique_submissions[row["SUBREDDIT"]]:
            subreddit_unique_submissions[row["SUBREDDIT"]][row["SUBMISSION_ID"]] = row
    return subreddit_votes_counter, subreddit_users, subreddit_unique_submissions


def convert_cluster_users_subreddit_input(n_clusters, users_in_cluster:dict, a_subreddit_unique_submissions:dict):
    random.seed(42)
    cluster_a = random.randint(0, n_clusters - 1)
    cluster_b = random.randint(0, n_clusters - 1)
    cluster_users_subreddit_input = defaultdict(list)
    for cluster_x in [cluster_a, cluster_b]:
        cluster_x_users = users_in_cluster[cluster_x]
        for user in cluster_x_users:
            for _submission in a_subreddit_unique_submissions:
                submission:pd.DataFrame = _submission.copy(deep=True)
                submission["USERNAME"] = user
                cluster_users_subreddit_input[cluster_x].append(submission)
        cluster_users_subreddit_input[cluster_x] = pd.cat(cluster_users_subreddit_input[cluster_x], axis = 0)
    return cluster_users_subreddit_input



if __name__ == "__main__":
    all_feature_columns, target, train_model_input, test_model_input, feature_names, train_data, test_data = get_model_input(config)
    user_embedding = get_best_model_embedding()
    debug(user_embedding=user_embedding)
    subreddit_votes_counter, subreddit_users, subreddit_unique_submissions = get_popular_subreddits(test_data)
    common_subreddits_counts = subreddit_votes_counter.most_common(3)
    debug(common_subreddits_counts=common_subreddits_counts)

    a_subreddit = common_subreddits_counts[0][0]
    debug(f"Subreddit {a_subreddit} have {common_subreddits_counts[a_subreddit]} test votes")
    a_subreddit_unique_submissions = subreddit_unique_submissions[a_subreddit]
    a_subreddit_users:set = subreddit_users[a_subreddit]
    a_subreddit_users_bool_vec = torch.zeros([user_embedding.shape[0]], dtype = bool)
    for user in a_subreddit_users:
        a_subreddit_users_bool_vec[user] = True # in_subreddit
    user_user_i_map = {}
    user_i_user_map = {}
    user_i = 0
    for user, in_subreddit in enumerate(a_subreddit_users_bool_vec):
        if in_subreddit:
            user_user_i_map[user] = user_i
            user_i_user_map[user_i] = user
            user_i += 1
    a_subreddit_users_embedding = user_embedding[a_subreddit_users_bool_vec, :]
    from sklearn.cluster import KMeans
    n_clusters = int(len(user_user_i_map) / 20)
    debug(n_clusters=n_clusters)
    kmeans = KMeans(n_clusters = n_clusters, random_state = 42, verbose = 1).fit(a_subreddit_users_embedding)
    # debug(labels=kmeans.labels_) # val: [584 350 948 ... 813 938 152]
    users_in_cluster = defaultdict(set)
    for user_i, cluster_label in enumerate(kmeans.labels_):
        users_in_cluster[cluster_label].add(user_i_user_map[user_i])
    cluster_users_subreddit_input = convert_cluster_users_subreddit_input(n_clusters, users_in_cluster, a_subreddit_unique_submissions)
    
