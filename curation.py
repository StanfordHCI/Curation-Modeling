import argparse
from collections import Counter, defaultdict
import os
import random
import shutil
import pandas as pd
from superdebug import debug
import torch
from data import get_model_input
from utils import get_config, get_ctr_model, join_sets, load_model, print_log, save_model, load_model_dict
from venn import venn, pseudovenn

def get_best_model(config):
    all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, train_data, test_data = get_model_input(config)
    CTRModel = get_ctr_model(config["model_type"])
    model = CTRModel(all_feature_columns, all_feature_columns, task='binary', device=config["device"], gpus = config["gpus"])
    model.compile(torch.optim.Adam(model.parameters(), lr = config["learning_rate"]), "binary_crossentropy", metrics=['binary_crossentropy', "auc", "acc"])
    model, _, _, _, model_dict = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    assert model_dict is not None, "No trained model"
    state_dict = model_dict["state_dict"]
    if config["model_type"] == "MLR":
        user_embedding = torch.cat([state_dict[f"region_linear_model.{i}.embedding_dict.USERNAME.weight"] for i in range(20) if f"region_linear_model.{i}.embedding_dict.USERNAME.weight" in state_dict], dim = -1)
    else:
        user_embedding = state_dict[f"embedding_dict.USERNAME.weight"] #[num_users, user_embed_dim]
    return model, user_embedding.cpu()

def get_popular_subreddits(train_data, test_data, user_votes_thres = 0):
    subreddit_votes_counter = Counter()
    subreddit_active_users = defaultdict(Counter)
    subreddit_train_submissions = defaultdict(dict)
    subreddit_test_submissions = defaultdict(dict)
    for i, row in train_data.iterrows():
        subreddit_votes_counter[row["SUBREDDIT"]] += 1
        subreddit_active_users[row["SUBREDDIT"]][row["USERNAME"]] += 1
        if row["SUBMISSION_ID"] not in subreddit_train_submissions[row["SUBREDDIT"]]:
            subreddit_train_submissions[row["SUBREDDIT"]][row["SUBMISSION_ID"]] = row
    for subreddit in subreddit_active_users:
        users_vote_count = subreddit_active_users[subreddit]
        subreddit_active_users[subreddit] = {user for user in users_vote_count if users_vote_count[user] >= user_votes_thres}
    for i, row in test_data.iterrows():
        if row["SUBMISSION_ID"] not in subreddit_test_submissions[row["SUBREDDIT"]]:
            subreddit_test_submissions[row["SUBREDDIT"]][row["SUBMISSION_ID"]] = row
    return subreddit_votes_counter, subreddit_active_users, subreddit_train_submissions, subreddit_test_submissions


def convert_cluster_users_subreddit_submissions_data(cluster_x, users_in_cluster:dict, unique_submissions:dict):
    cluster_x_users_subreddit_submissions_data = []
    cluster_x_users = users_in_cluster[cluster_x]
    for user in cluster_x_users:
        for submission_id in unique_submissions:
            submission:pd.DataFrame = unique_submissions[submission_id].copy(deep=True)
            submission["USERNAME"] = user
            cluster_x_users_subreddit_submissions_data.append(submission)
    cluster_x_users_subreddit_submissions_data = pd.DataFrame(cluster_x_users_subreddit_submissions_data)
    return cluster_x_users_subreddit_submissions_data

def get_user_reps(selected_users, all_user_embedding, train_data:pd.DataFrame = None, selected_submissions = None, method = "neural"):
    assert all_user_embedding is not None
    selected_users_bool_vec = torch.zeros([all_user_embedding.shape[0]], dtype = bool)
    for user in selected_users:
        selected_users_bool_vec[user] = True # in_subreddit
    # user_user_i_map = {}
    user_i_user_map = {}
    user_i = 0
    for user, in_subreddit in enumerate(selected_users_bool_vec):
        if in_subreddit:
            # user_user_i_map[user] = user_i
            user_i_user_map[user_i] = user
            user_i += 1
    # assert len(user_user_i_map) == len(user_i_user_map)
    if method == "neural":
        selected_users_reps = all_user_embedding[selected_users_bool_vec, :]
    elif method == "votes":
        assert train_data is not None and selected_submissions is not None
        sub_sub_i_map = {sub: sub_i for sub_i, sub in enumerate(list(selected_submissions.keys()))}
        users_reps = torch.zeros([all_user_embedding.shape[0], len(selected_submissions)])
        for row_i, row in train_data.iterrows():
            if row["USERNAME"] in selected_users and row["SUBMISSION_ID"] in selected_submissions:
                vote = 1 if row["VOTE"] == 1 else -1
                users_reps[row["USERNAME"], sub_sub_i_map[row["SUBMISSION_ID"]]] = vote
        selected_users_reps = users_reps[selected_users_bool_vec, :]
        users_vote_sum = (selected_users_reps * selected_users_reps).sum(axis = -1, keepdim= True)
        assert (users_vote_sum != 0).all()
        selected_users_reps = selected_users_reps / users_vote_sum # average votes on each submission
        debug(selected_users_reps = selected_users_reps)

    return selected_users_reps, user_i_user_map
    
def get_user_clusters(selected_users_reps, user_i_user_map:dict, single_user_as_cluster = False):
    if single_user_as_cluster:
        users_in_clusters = {i: {user} for i,user in user_i_user_map.items()}
        cluster_centers = None
    else:
        n_clusters = int(len(user_i_user_map) / 100)
        debug(n_clusters=n_clusters) # n_clusters: 118
        debug("Begin clustering...")
        from sklearn.cluster import KMeans
        clustering = KMeans(n_clusters = n_clusters, random_state = 42, verbose = 0).fit(selected_users_reps)
        cluster_centers = clustering.cluster_centers_
        debug(cluster_centers=cluster_centers)
        """
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(linkage = "complete").fit(selected_users_reps)
        """
        """
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters, random_state = 42, verbose = 0).fit(selected_users_reps)
        """
        labels = clustering.labels_ # clustering.labels_: [584 350 948 ... 813 938 152]
        """
        from sklearn.mixture import GaussianMixture
        labels = GaussianMixture(n_clusters, random_state = 42, verbose = 0).fit_predict(selected_users_reps)
        """
        users_in_clusters = defaultdict(set)
        usernames_in_clusters = defaultdict(set)
        for user_i, cluster_x in enumerate(labels): 
            users_in_clusters[cluster_x].add(user_i_user_map[user_i])
            usernames_in_clusters[cluster_x].add(original_feature_map["USERNAME"][user_i_user_map[user_i]])
        assert len(join_sets(users_in_clusters.values())) == sum([len(users) for users in users_in_clusters.values()])
        debug(cluster_user_num=str({cluster_x: len(users_in_clusters[cluster_x]) for cluster_x in users_in_clusters}))
        debug(usernames_in_clusters=str(usernames_in_clusters))
    return users_in_clusters, cluster_centers

def predict_cluster_users_preferred_submissions(model, cluster_x_users_subreddit_submissions_data:pd.DataFrame, train_data:pd.DataFrame, feature_names, thres = 10):
    # collect existing votes
    existing_votes = {}
    for row_i, row in train_data.iterrows():
        existing_votes[f'{row["USERNAME"]}-{row["SUBMISSION_ID"]}'] = row["VOTE"]

    # predict unseen votes
    cluster_x_users_subreddit_submissions_input = {name:cluster_x_users_subreddit_submissions_data[name] for name in feature_names}
    predict_cluster_x_users_subreddit_submissions_votes = model.predict(cluster_x_users_subreddit_submissions_input, batch_size=config['batch_size'])
    debug(predict_cluster_x_users_subreddit_submissions_votes=predict_cluster_x_users_subreddit_submissions_votes)
    submission_votes = {}
    for row_i, vote in enumerate(predict_cluster_x_users_subreddit_submissions_votes[:, 0]):
        row = cluster_x_users_subreddit_submissions_data.iloc[row_i]
        submission_id = row["SUBMISSION_ID"]
        if submission_id not in submission_votes:
            submission_votes[submission_id] = [0, 0]
        if f'{row["USERNAME"]}-{row["SUBMISSION_ID"]}' not in existing_votes:
            vote = int(vote >= 0.5)
        else: # use existing votes if available
            vote = existing_votes[f'{row["USERNAME"]}-{row["SUBMISSION_ID"]}']
        submission_votes[submission_id][vote] += 1

    # include submissions to preferred_submissions where %upvotes is higher than threshold
    preferred_submissions = set()
    for submission_id in submission_votes:
        submission_votes[submission_id].append(submission_votes[submission_id][1] / (submission_votes[submission_id][0] + submission_votes[submission_id][1])) # %upvotes
        if submission_votes[submission_id][-1] >= thres:
            preferred_submissions.add(submission_id)

    # sort submissions using %upvotes
    submissions_ranking = list(submission_votes.keys())
    submissions_ranking.sort(reverse=True, key=lambda id: submission_votes[id][-1])
    return preferred_submissions, submissions_ranking, submission_votes

def predict_clusters_preferences(users_in_clusters, unique_submissions, train_data, feature_names, cluster_centers=None, single_user_as_cluster = False):
    clusters_users_preferred_submissions = {}
    used_cluster_centers = []
    if os.path.exists(config["preferred_submissions_venn_figure_dir"]):
        shutil.rmtree(config["preferred_submissions_venn_figure_dir"])
    os.makedirs(config["preferred_submissions_venn_figure_dir"], exist_ok=True)
    for cluster_x in users_in_clusters:
        if (not single_user_as_cluster) and (len(users_in_clusters[cluster_x]) <= config["cluster_user_num_lower_thres"] or len(users_in_clusters[cluster_x]) > config["cluster_user_num_upper_thres"]): # keep middle sized centers
            continue
        if cluster_centers is not None: # only keep not similar centers
            cluster_x_center = cluster_centers[cluster_x]
            similar_center = False
            for center in used_cluster_centers:
                if torch.dot(cluster_x_center, center) > 0:
                    similar_center = True
                    break
            if similar_center:
                continue
        if len(clusters_users_preferred_submissions) >= 6:
            break
        cluster_x_users_subreddit_submissions_data = convert_cluster_users_subreddit_submissions_data(cluster_x, users_in_clusters, unique_submissions)
        cluster_x_users_preferred_submissions, cluster_x_preferred_submissions_ranking, cluster_x_users_submission_votes = predict_cluster_users_preferred_submissions(model, cluster_x_users_subreddit_submissions_data, train_data, feature_names, thres = config["upvote_downvote_ratio_thres"])
        clusters_users_preferred_submissions[f"Cluster {cluster_x}"] = cluster_x_users_preferred_submissions
        print_log(config["log_path"], f"Users in cluster {cluster_x} prefers {len(cluster_x_users_preferred_submissions)}/{len(unique_submissions)} submissions (sorted using %upvotes): {cluster_x_preferred_submissions_ranking[:len(cluster_x_users_preferred_submissions)]}")
        if len(clusters_users_preferred_submissions) > 1:
            ax = venn(clusters_users_preferred_submissions) if len(clusters_users_preferred_submissions) <=5 else pseudovenn(clusters_users_preferred_submissions)
            ax.figure.savefig(f"{config['preferred_submissions_venn_figure_dir']}/{len(clusters_users_preferred_submissions)}_clusters.png")

def curation_a_subreddit(a_subreddit, subreddit_active_users, subreddit_votes_counter, user_votes_thres, subreddit_train_submissions, subreddit_test_submissions, user_embedding, train_data, test_data, feature_names, single_user_as_cluster = False):
    a_subreddit_active_users:set = subreddit_active_users[a_subreddit]
    print_log(config["log_path"], f"In train data, subreddit {a_subreddit} have {len(a_subreddit_active_users)} active users (who votes >= {user_votes_thres} times), {subreddit_votes_counter[a_subreddit]} votes and {len(subreddit_train_submissions[a_subreddit])} unique submissions. In test data, subreddit {a_subreddit} have {len(subreddit_test_submissions[a_subreddit])} unique submissions.") 

    # a_subreddit_users_reps, user_i_user_map = get_user_reps(a_subreddit_users, all_user_embedding=user_embedding, method = "neural")
    a_subreddit_users_reps, user_i_user_map = get_user_reps(a_subreddit_active_users, all_user_embedding=user_embedding, train_data=train_data, selected_submissions = subreddit_train_submissions[a_subreddit], method = config["user_clustering_method"])
    users_in_clusters, cluster_centers = get_user_clusters(a_subreddit_users_reps, user_i_user_map, single_user_as_cluster=single_user_as_cluster)
    predict_clusters_preferences(users_in_clusters, subreddit_test_submissions[a_subreddit], train_data, feature_names, cluster_centers, single_user_as_cluster=single_user_as_cluster)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to custom config file.")
    args = parser.parse_args()
    config = get_config(args.config, "_curation")
    return config

if __name__ == "__main__":
    config = parse_config()
    all_feature_columns, target, train_model_input, test_model_input, feature_names, original_feature_map, train_data, test_data = get_model_input(config)
    model, user_embedding = get_best_model(config)
    debug(user_embedding=user_embedding)
    active_user_votes_thres = config["active_user_votes_thres"]
    subreddit_votes_counter, subreddit_active_users, subreddit_train_submissions, subreddit_test_submissions = get_popular_subreddits(train_data, test_data, user_votes_thres = active_user_votes_thres) # subreddit_votes_counter, subreddit_users, subreddit_train_submissions are based on train_data, subreddit_test_submissions are based on test_data
    common_subreddits_counts = subreddit_votes_counter.most_common(20)
    for subreddit_id, vote_counts in common_subreddits_counts:
        print(f"Subreddit {subreddit_id}: {original_feature_map['SUBREDDIT'][subreddit_id]}, {vote_counts} votes")

    a_subreddit = int(input("Select a subreddit: ")) # common_subreddits_counts[0][0]
    print_log(config["log_path"], f"Selected subreddit: {a_subreddit} ({original_feature_map['SUBREDDIT'][a_subreddit]})")
    single_user_as_cluster = config["single_user_as_cluster"]
    curation_a_subreddit(a_subreddit, subreddit_active_users, subreddit_votes_counter, active_user_votes_thres, subreddit_train_submissions, subreddit_test_submissions, user_embedding, train_data, test_data, feature_names, single_user_as_cluster=single_user_as_cluster)
