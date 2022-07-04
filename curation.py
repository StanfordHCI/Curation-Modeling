from collections import Counter, defaultdict
import random
import pandas as pd
from superdebug import debug
import torch
from data import get_model_input
from utils import get_config, get_ctr_model, join_sets, load_model, save_model, load_model_dict
from venn import venn, pseudovenn
CONFIG_PATH = "configs/debug.yml"
config, log_path = get_config(CONFIG_PATH)

def get_best_model():
    from main import model
    model, _, _, _, model_dict = load_model(config["save_model_dir"], model, model.optim, 0, 0, "best")
    assert model_dict is not None, "No trained model"
    state_dict = model_dict["state_dict"]
    user_embedding = state_dict[f"embedding_dict.USERNAME.weight"] #[num_users, user_embed_dim]
    return model, user_embedding.cpu()

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


def convert_cluster_users_subreddit_submissions_data(cluster_x, users_in_cluster:dict, a_subreddit_unique_submissions:dict):
    cluster_x_users_subreddit_submissions_data = []
    cluster_x_users = users_in_cluster[cluster_x]
    for user in cluster_x_users:
        for submission_id in a_subreddit_unique_submissions:
            submission:pd.DataFrame = a_subreddit_unique_submissions[submission_id].copy(deep=True)
            submission["USERNAME"] = user
            cluster_x_users_subreddit_submissions_data.append(submission)
    cluster_x_users_subreddit_submissions_data = pd.DataFrame(cluster_x_users_subreddit_submissions_data)
    return cluster_x_users_subreddit_submissions_data

def predict_cluster_users_preferred_submissions(model, cluster_x_users_subreddit_submissions_data:pd.DataFrame, feature_names, thres = 10):
    cluster_x_users_subreddit_submissions_input = {name:cluster_x_users_subreddit_submissions_data[name] for name in feature_names}
    predict_cluster_x_users_subreddit_submissions_votes = model.predict(cluster_x_users_subreddit_submissions_input, batch_size=config['batch_size'])
    debug(predict_cluster_x_users_subreddit_submissions_votes=predict_cluster_x_users_subreddit_submissions_votes)
    submission_votes = {}
    for row_i, vote in enumerate(predict_cluster_x_users_subreddit_submissions_votes[:, 0]):
        row = cluster_x_users_subreddit_submissions_data.iloc[row_i]
        submission_id = row["SUBMISSION_ID"]
        vote = int(vote >= 0.5)
        if submission_id not in submission_votes:
            submission_votes[submission_id] = [0, 0]
        submission_votes[submission_id][vote] += 1
    preferred_submissions = set()
    for submission_id in submission_votes:
        if (submission_votes[submission_id][0] == 0 and submission_votes[submission_id][1] > 0) or submission_votes[submission_id][1] / submission_votes[submission_id][0] >= thres:
            preferred_submissions.add(submission_id)
    return preferred_submissions, submission_votes

if __name__ == "__main__":
    all_feature_columns, target, train_model_input, test_model_input, feature_names, train_data, test_data = get_model_input(config)
    model, user_embedding = get_best_model()
    debug(user_embedding=user_embedding)
    subreddit_votes_counter, subreddit_users, subreddit_unique_submissions = get_popular_subreddits(test_data)
    common_subreddits_counts = subreddit_votes_counter.most_common(3)
    debug(common_subreddits_counts=common_subreddits_counts)

    a_subreddit = common_subreddits_counts[0][0]
    debug(f"Subreddit {a_subreddit} have {subreddit_votes_counter[a_subreddit]} test votes, {len(subreddit_unique_submissions[a_subreddit])} unique submissions") # Subreddit 26646 have 7027 test votes
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
    assert len(user_user_i_map) == len(user_i_user_map)
    a_subreddit_users_embedding = user_embedding[a_subreddit_users_bool_vec, :]
    n_clusters = int(len(user_user_i_map) / 100)
    debug(n_clusters=n_clusters) # n_clusters: 118
    """
    from sklearn.cluster import KMeans
    clustering = KMeans(n_clusters = n_clusters, random_state = 42, verbose = 0).fit(a_subreddit_users_embedding)
    """
    """
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(linkage = "complete").fit(a_subreddit_users_embedding)
    """
    """
    from sklearn.cluster import SpectralClustering
    clustering = SpectralClustering(n_clusters, random_state = 42, verbose = 0).fit(a_subreddit_users_embedding)
    """
    # labels = clustering.labels_ # clustering.labels_: [584 350 948 ... 813 938 152]
    from sklearn.mixture import GaussianMixture
    labels = GaussianMixture(n_clusters, random_state = 42, verbose = 0).fit_predict(a_subreddit_users_embedding)


    users_in_cluster = defaultdict(set)
    for user_i, cluster_x in enumerate(labels): 
        users_in_cluster[cluster_x].add(user_i_user_map[user_i])
    assert len(join_sets(users_in_cluster.values())) == sum([len(users) for users in users_in_cluster.values()])
    debug(cluster_user_num=str({cluster_x: len(users_in_cluster[cluster_x]) for cluster_x in users_in_cluster}))
    clusters_users_preferred_submissions = {}
    for cluster_x in users_in_cluster:
        if len(users_in_cluster[cluster_x]) <= 50:
            continue
        if len(clusters_users_preferred_submissions) >= 6:
            break
        cluster_x_users_subreddit_submissions_data = convert_cluster_users_subreddit_submissions_data(cluster_x, users_in_cluster, a_subreddit_unique_submissions)
        cluster_x_users_preferred_submissions, cluster_x_users_submission_votes = predict_cluster_users_preferred_submissions(model, cluster_x_users_subreddit_submissions_data, feature_names, thres = 10)
        clusters_users_preferred_submissions[f"Cluster {cluster_x}"] = cluster_x_users_preferred_submissions
        print(f"Users in cluster {cluster_x} prefers {len(cluster_x_users_preferred_submissions)}/{len(subreddit_unique_submissions[a_subreddit])} submissions {cluster_x_users_preferred_submissions}")
        if len(clusters_users_preferred_submissions) > 1:
            ax = venn(clusters_users_preferred_submissions) if len(clusters_users_preferred_submissions) <=5 else pseudovenn(clusters_users_preferred_submissions)
            ax.figure.savefig(f"output/figures/preferred_subs/{len(clusters_users_preferred_submissions)}clusters.png")
    # debug(f"Cluster {clusters_users_preferred_submissions[0][0]} preferred submissions: {clusters_users_preferred_submissions[0][1] - clusters_users_preferred_submissions[1][1]}")
    # debug(f"Cluster {clusters_users_preferred_submissions[1][0]} preferred submissions: {clusters_users_preferred_submissions[1][1] - clusters_users_preferred_submissions[0][1]}")
    # debug(f"Both preferred submissions: {clusters_users_preferred_submissions[1][1] & clusters_users_preferred_submissions[0][1]}")
        


    
