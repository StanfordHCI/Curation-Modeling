# uvicorn curation_backend:app --reload --port 5000
import copy
import json
from utils import get_config, join_sets, load_model, print_log, save_model, load_model_dict, record_existing_votes
from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd
from superdebug import debug
from process_data import get_model_input
from model import get_best_model
import time
import re
from train import evaluate_model, train_model
from fastapi import FastAPI

app = FastAPI()

CONFIG_PATH = "configs/deploy_CURIO_full_data.yml"
print(CONFIG_PATH)
config = get_config(CONFIG_PATH, "_curation", print_config = False)
batch_size = config["batch_size"]
upvote_confidence_thres = config["upvote_confidence_thres"] # 0.5

target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info, train_submission_upvote_df, num_all_users = get_model_input(config)
model, token_embedding = get_best_model(config, categorical_features, string_features, original_feature_map)
model = model.to(model.device); model.eval()
all_users = list(range(max(max(train_data["USERNAME"]), max(test_data["USERNAME"])) + 1))
existing_votes, _, _, _, _ = record_existing_votes(pd.concat([train_data, test_data], axis=0))
for user_submission_id in list(existing_votes.keys()):
    if existing_votes[user_submission_id] == 0:
        del existing_votes[user_submission_id]
# existing_votes: {"user-submission_id": 1}

inverse_username_map = {original_feature_map["USERNAME"][username]:username for username in original_feature_map["USERNAME"].keys()} 
community_curators = json.loads(open("data/Curio/community_curators.json", "r").read())


def construct_submission_info(submission_id, community, author, title, content, username = "", vote = 1.0):
    return pd.Series({
        'SUBMISSION_ID': submission_id,
        'SUBREDDIT': f"r/{community.replace(' ', '|')}",
        'CREATED_TIME': re.sub("[0-9][0-9]:[0-9][0-9]:[0-9][0-9] ", "", time.ctime(time.time())),
        'USERNAME': 0 if username == "" else inverse_username_map[username],
        'VOTE': vote,
        'TITLE': title,
        'AUTHOR': inverse_username_map[author] if type(author) == str else author,
        '#_COMMENTS': 0,
        'NSFW': 'false',
        'SCORE': 0,
        'UPVOTED_%': 0.5,
        'LINK': '',
        'SUBMISSION_TEXT': (title + " [SEP] " + content) if content != "" else title,
        'UPVOTED_USERS': [],
        'DOWNVOTED_USERS': []}
    )

def predict_curator_upvote_rate_with_model(community: str, author: str, title: str, content: str, submission_id: str, model, new_existing_votes = None):
    # Curation threshold: if we set it as one, then the posts posted by a curator will always be curated since they will automatically receive one curator upvote. And if we set it as two, then the posts posted by a curator and echoed by another curator will also always get curated.
    if new_existing_votes is None:
        new_existing_votes = existing_votes
    submission_info = construct_submission_info(submission_id, community, author, title, content)
    if community.replace(" ", "|") in community_curators and "TEST" not in community and "test" not in community:
        curator_usernames = community_curators[community.replace(" ", "|")]
    else:
        curator_usernames = []
    curator_users = {inverse_username_map[username] for username in curator_usernames}
    if len(curator_users) > 0:
        print(f"Predicting {len(curator_users)} curators")
        # get group_users_submissions_data
        group_users_submissions_data = []
        unique_submissions = pd.DataFrame([submission_info])
        for user in curator_users:
            submissions = unique_submissions.copy(deep=True)
            submissions["USERNAME"] = [user] * len(submissions) # it doesn't matter whether the user itself is in UPVOTED_USERS / DOWNVOTED_USERS: we will substitute it with real votes
            group_users_submissions_data.append(submissions)
        group_users_submissions_data = pd.concat(group_users_submissions_data,axis=0)
        
        # predict unseen votes
        predicted_group_users_submissions_votes = evaluate_model(config, model, data=group_users_submissions_data, weights = None, batch_size=config["batch_size"], sample_voted_users=False, extra_input = (categorical_features, string_features, target), ret = "prediction") # ndarray size: (3423664, 1)
        
        pred_user_submission_vote_vec = - np.ones([max(curator_users) + 1], dtype = int) # use ground truth vote if available, -1 for not in data

        curator_votes = [0, 0]
        usernames = group_users_submissions_data["USERNAME"].to_numpy()
        for row_i in range(len(group_users_submissions_data)):
            username = usernames[row_i]
            vote_score = predicted_group_users_submissions_votes[row_i, 0]
            if f'{username}-{submission_id}' not in new_existing_votes:
                vote = int(vote_score >= upvote_confidence_thres)
            else: # use existing votes if available
                vote = int(new_existing_votes[f'{username}-{submission_id}'] >= 0.5)
            pred_user_submission_vote_vec[username] = vote
            curator_votes[vote] += 1
        assert (pred_user_submission_vote_vec != -1).sum() > 0
        
        # calculate %upvotes
        curator_votes.append(curator_votes[1] / (curator_votes[0] + curator_votes[1])) # %upvotes
        curator_upvote_rate = (pred_user_submission_vote_vec == 1).sum(axis=0).astype(float)/(pred_user_submission_vote_vec != -1).sum(axis=0).astype(float) # scalar
        assert curator_votes[-1] == curator_upvote_rate
        # curate_submission = curator_votes[-1] >= upvote_ratio_thres
    
    else:
        curator_upvote_rate = 1.0
    return curator_upvote_rate

def predict_curator_upvote_rate(community: str, author: str, title: str, content: str, submission_id: str = "custom"):
    return predict_curator_upvote_rate_with_model(community, author, title, content, submission_id, model)

def finetune_model_1step(model, submission_id, community, author, title, content, username, vote = 1.0):
    # NOTE: input copy.deepcopy(model) to avoid modifying the original model
    if "TEST" in community or "test" in community:
        print("Do not finetune model on test community")
        return model
    submission_info = construct_submission_info(submission_id, community, author, title, content, username, vote)
    new_model = next(train_model(config, model, data=pd.DataFrame([submission_info]), weights = None, batch_size=1, epochs=1, verbose=2, validation_split=False, step_generator=True, n_step_per_sample=1, extra_input = (categorical_features, string_features, target)))
    print(f"Model finetuned on {username} {'upvote' if vote == 1.0 else 'downvote'} on {submission_id} in {community} ({title})")
    return new_model

@app.get("/predict_curator_upvote_rate_with_new_upvote")
def predict_curator_upvote_rate_with_new_upvote(community: str, username: str, author: str, title: str, content: str, submission_id: str):
    # Usage: when a user posted a post (and upvoted it, so username = author), or when a user reacted to a post
    # Finetune the model one step further and then predict the curator upvote rate
    global model
    if f'{inverse_username_map[username]}-{submission_id}' not in existing_votes or existing_votes[f'{inverse_username_map[username]}-{submission_id}'] != 1:
        existing_votes[f'{inverse_username_map[username]}-{submission_id}'] = 1
        # train model
        model = finetune_model_1step(model, submission_id, community, author, title, content, username)
    # predict
    return predict_curator_upvote_rate_with_model(community, author, title, content, submission_id, model)

@app.get("/predict_curator_upvote_rate_with_new_downvote")
def predict_curator_upvote_rate_with_new_downvote(community: str, username: str, author: str, title: str, content: str, submission_id: str):
    # Usage: when a user removed their reaction to a post
    # Finetune the model one step further and then predict the curator upvote rate
    global model
    if f'{inverse_username_map[username]}-{submission_id}' in existing_votes and existing_votes[f'{inverse_username_map[username]}-{submission_id}'] == 1:
        del existing_votes[f'{inverse_username_map[username]}-{submission_id}']
        # train model
        model = finetune_model_1step(model, submission_id, community, author, title, content, username, vote = 0.0)
    # predict
    return predict_curator_upvote_rate_with_model(community, author, title, content, submission_id, model)

# def train_model_predict_curator_upvote_rate_with_new_echo(orig_community: str, new_community: str, username: str, author: str, title: str, content: str, model, orig_submission_id: str = "custom", echoed_submission_id: str = "custom", new_existing_votes = None):
#     # when a user is attempting to echo a post or when the user has already echoed the post, we need to finetune the model further (the user (echoauthor) will upvote on this post in the original community, the original author will also upvote on the post in the new community it is echoed to) and then predict the curator upvote rate
#     # train model
#     new_model = finetune_model_1step(copy.deepcopy(model), orig_submission_id, orig_community, author, title, content, username) # username is echoauthor
#     new_model = finetune_model_1step(new_model, echoed_submission_id, new_community, author, title, content, author) # author is original post author
#     # predict
#     curator_upvote_rate = predict_curator_upvote_rate_with_model(new_community, author, title, content, echoed_submission_id, new_model, new_existing_votes)
#     return new_model, curator_upvote_rate

@app.get("/predict_curator_upvote_rate_with_echo_attempt")
def predict_curator_upvote_rate_with_echo_attempt(orig_community: str, new_community: str, username: str, author: str, title: str, content: str, orig_submission_id: str, echoed_submission_id: str = "custom"):
    # Usage: when a user is attempting to echo a post
    new_existing_votes = copy.deepcopy(existing_votes)
    new_existing_votes[f'{inverse_username_map[username]}-{orig_submission_id}'] = 1
    new_existing_votes[f'{inverse_username_map[username]}-{echoed_submission_id}'] = 1
    new_existing_votes[f'{inverse_username_map[author]}-{echoed_submission_id}'] = 1
    # train model
    new_model = finetune_model_1step(copy.deepcopy(model), orig_submission_id, orig_community, author, title, content, username) # username is echoauthor
    new_model = finetune_model_1step(new_model, echoed_submission_id, new_community, author, title, content, author) # author is original post author
    # predict
    curator_upvote_rate = predict_curator_upvote_rate_with_model(new_community, author, title, content, echoed_submission_id, new_model, new_existing_votes)
    return curator_upvote_rate

@app.get("/predict_curator_upvote_rate_with_echo_proposal")
def predict_curator_upvote_rate_with_echo_proposal(orig_community: str, username: str, author: str, title: str, content: str, orig_submission_id: str):
    # Usage: when a user has proposed an echo (to an ancestor community)
    global model, existing_votes
    existing_votes[f'{inverse_username_map[username]}-{orig_submission_id}'] = 1
    # train model
    model = finetune_model_1step(model, orig_submission_id, orig_community, author, title, content, username) # username is echoauthor
    # predict
    curator_upvote_rate = predict_curator_upvote_rate_with_model(orig_community, author, title, content, orig_submission_id, model)
    return curator_upvote_rate

@app.get("/predict_curator_upvote_rate_with_echo_consent")
def predict_curator_upvote_rate_with_new_echo(orig_community: str, new_community: str, username: str, author: str, title: str, content: str, orig_submission_id: str, echoed_submission_id: str):
    # Usage: when the author has consented to the echo (to an ancestor community)
    global model, existing_votes
    existing_votes[f'{inverse_username_map[username]}-{echoed_submission_id}'] = 1
    existing_votes[f'{inverse_username_map[author]}-{echoed_submission_id}'] = 1
    # train model
    model = finetune_model_1step(model, echoed_submission_id, new_community, author, title, content, author) # author is original post author
    # predict
    curator_upvote_rate_orig_community = predict_curator_upvote_rate_with_model(orig_community, author, title, content, orig_submission_id, model)
    curator_upvote_rate_new_community = predict_curator_upvote_rate_with_model(new_community, author, title, content, echoed_submission_id, model)
    return {"orig_community": curator_upvote_rate_orig_community, "new_community": curator_upvote_rate_new_community}

@app.get("/predict_curator_upvote_rate_with_new_echo")
def predict_curator_upvote_rate_with_new_echo(orig_community: str, new_community: str, username: str, author: str, title: str, content: str, orig_submission_id: str, echoed_submission_id: str):
    # Usage: when a user has already echoed the post (to a descendant community)
    global model, existing_votes
    existing_votes[f'{inverse_username_map[username]}-{orig_submission_id}'] = 1
    existing_votes[f'{inverse_username_map[username]}-{echoed_submission_id}'] = 1
    existing_votes[f'{inverse_username_map[author]}-{echoed_submission_id}'] = 1
    # train model
    model = finetune_model_1step(model, orig_submission_id, orig_community, author, title, content, username) # username is echoauthor
    model = finetune_model_1step(model, echoed_submission_id, new_community, author, title, content, author) # author is original post author
    # predict
    curator_upvote_rate_orig_community = predict_curator_upvote_rate_with_model(orig_community, author, title, content, orig_submission_id, model)
    curator_upvote_rate_new_community = predict_curator_upvote_rate_with_model(new_community, author, title, content, echoed_submission_id, model)
    return {"orig_community": curator_upvote_rate_orig_community, "new_community": curator_upvote_rate_new_community}

if __name__ == "__main__":
    custom_submission_id = "custom_" + str(random.randint(1000000000000000, 9999999999999999))
    custom_author = "wanrong" # input("Input author's username: ")
    custom_title = 'Employee of the year' # input("Input post title: ")
    custom_content = 'This is so funny!!!' # input("Input post content: ")
    custom_community = 'The Positive Corner' # input("Input community: ")
    curator_upvote_rate = predict_curator_upvote_rate(custom_community, custom_author, custom_title, custom_content, custom_submission_id)
    
    debug(curator_upvote_rate=curator_upvote_rate)
    upvote_ratio_thres = config["upvote_ratio_thres"] # 0.5
    if curator_upvote_rate > upvote_ratio_thres:
        print(f"You can post immediately")
    else:
        print("your post will stay in the background first.")