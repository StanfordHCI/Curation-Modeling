import pickle
import nbimporter
import streamlit as st
from curation import *
import os
import json
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-service-account-file.json"
from google.cloud import language_v1
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import grpc, google
client = language_v1.LanguageServiceClient()

@st.experimental_singleton
def get_db():
    submission_analysis_path = "data/reddit/submission_analysis.db"
    engine = create_engine(f"sqlite:///{submission_analysis_path}", connect_args={'timeout': 10})
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    return session
    

session = get_db()

Base = declarative_base()
class Analysis(Base):
    __tablename__ = 'analysis'
    id = Column(String, primary_key=True, autoincrement=True)
    sentiment_score = Column(Integer)
    content_classes = Column(String)
    entities = Column(String)


CONFIG_PATH = "configs/subreddit_minority_no_peer.yml"
config = get_config(CONFIG_PATH, "_curation", print_config = False)
selected_subreddit = config["selected_subreddit"]
selected_subreddit = "r/politics_r/Conservative_r/Liberal_r/Republican_r/democrats_r/VoteBlue"
user_grouping_method = config["user_grouping_method"] # 
user_grouping_method = "interest_r/Conservative_r/Liberal_r/Republican_r/democrats_r/VoteBlue random_user_as_group"


if 'cached_predictions' not in st.session_state:
    
    active_user_votes_thres = config["active_user_votes_thres"]
    batch_size = config["batch_size"]
    


    submission_sentiment_map = {}
    submission_class_map = {}
    submission_entity_map = {}


    target, original_feature_map, categorical_features, string_features, train_data, test_data, test_data_info, train_submission_upvote_df, num_all_users = get_model_input(config)
    extra_input = (categorical_features, string_features, target)
    model, token_embedding = get_best_model(config, categorical_features, string_features, original_feature_map)
    model.eval()
    all_users = list(range(max(max(train_data["USERNAME"]), max(test_data["USERNAME"])) + 1))

    subreddit_votes_counter, subreddit_active_users, subreddit_user_vote_count, subreddit_train_submissions, subreddit_test_submissions, all_submissions = get_subreddits_submissions(train_data, test_data, user_votes_thres = active_user_votes_thres, max_test_submissions_per_subreddit=config["max_test_submissions_per_subreddit"]) # subreddit_votes_counter, subreddit_users, subreddit_train_submissions are based on train_data, subreddit_test_submissions are based on test_data


    existing_votes, existing_user_votes, existing_user_updown_votes, existing_submission_votes, existing_user_subreddits = record_existing_votes(train_data)


    selected_subreddit_active_users, subreddit_active_users, subreddit_votes_counter, subreddit_train_submissions, subreddit_test_submissions = get_selected_subreddit_info(config, selected_subreddit, subreddit_active_users, subreddit_votes_counter, subreddit_train_submissions, subreddit_test_submissions, original_feature_map, active_user_votes_thres)

    analyze_post = False

    if analyze_post and config["submission_source"] == "test_data":
        subreddit_submissions_ids = list(subreddit_test_submissions[selected_subreddit].keys())
        subreddit_submissions_ids, subreddit_submissions_text = get_submissions_text(subreddit_submissions_ids, subreddit_test_submissions, selected_subreddit, pass_analyzed = True, submission_sentiment_map = submission_sentiment_map, submission_class_map = submission_class_map, submission_entity_map = submission_entity_map)
        # subreddit_submissions_ids, subreddit_submissions_text
        post_analysis_batch({id: text for id, text in zip(subreddit_submissions_ids, subreddit_submissions_text)}, session, Analysis, submission_sentiment_map, submission_class_map, submission_entity_map, language_v1, client, google)
        
        
    
    manual_user_groups = config["manual_user_groups"]
    # manual_user_groups = {"Conservative": {66, 39, 10, 44, 16, 60}, "Democratic":{0, 65, 64, 37, 49, 52, 20, 22, 23, 26, 29}}
    debug(user_grouping_method=user_grouping_method)



    debug(max_user = max(all_users), max_selected_subreddit_active_users = max([int(_) for _ in selected_subreddit_active_users]))
    all_username_tokens = [f"USERNAME_{user_i}" for user_i in all_users]
    all_username_token_ids = torch.tensor(model.tokenizer.convert_tokens_to_ids(all_username_tokens))
    all_username_token_ids = all_username_token_ids.to(model.device); model = model.to(model.device)
    with torch.no_grad():
        user_embedding = model.lm_encoder.embeddings.word_embeddings(all_username_token_ids)
        debug(user_embedding=user_embedding.shape)
    # debug(all_username_tokens=all_username_tokens, all_username_token_ids=all_username_token_ids, user_embedding=user_embedding)
    selected_subreddit_active_users_reps, selected_subreddit_active_user_i_user_map = get_user_reps(selected_subreddit_active_users, all_user_embedding=user_embedding, train_data=train_data, selected_submissions = subreddit_train_submissions[selected_subreddit], user_grouping_method = user_grouping_method)
    debug(selected_subreddit_active_users_reps=selected_subreddit_active_users_reps) # NOTE: selected_subreddit_active_users_reps is not None only if user_grouping_method == "neural" or "votes"


    reliability_bias_df, media_url_re = get_url_reliability_bias()



    if type(selected_subreddit_active_users_reps) == torch.tensor: selected_subreddit_active_users_reps = selected_subreddit_active_users_reps.cpu()
    users_in_groups, group_centers = get_user_groups(selected_subreddit, selected_subreddit_active_users, selected_subreddit_active_users_reps, selected_subreddit_active_user_i_user_map, user_grouping_method=user_grouping_method, existing_user_votes=existing_user_votes, manual_user_groups=manual_user_groups, train_data = train_data, original_feature_map=original_feature_map, selected_submissions = subreddit_train_submissions[selected_subreddit], model = model, subreddit_active_users=subreddit_active_users, selected_subreddit_active_users=selected_subreddit_active_users, subreddit_user_vote_count = subreddit_user_vote_count, reliability_bias_df=reliability_bias_df, media_url_re=media_url_re, extra_input=extra_input)
    debug(user_num_in_group = {x: len(y) for x, y in users_in_groups.items()})


    submissions_before_curation:dict = subreddit_test_submissions[selected_subreddit]
    # # TODO: only select a small set of submissions
    # one_key = list(submissions_before_curation.keys())[0]
    # submissions_before_curation = {one_key: submissions_before_curation[one_key]}
    
    
    pred_group_votes_info = {}
    
    
    model = model.to(model.device); model.eval()
    groups_preferred_submissions, groups_preferred_submissions_text, groups_submission_upvote_count_matrix = predict_groups_preferences(config, model, users_in_groups, submissions_before_curation, subreddit_test_submissions, selected_subreddit, group_centers=group_centers, user_grouping_method=user_grouping_method, existing_votes=existing_votes, existing_user_updown_votes=existing_user_updown_votes, pred_group_votes_info = pred_group_votes_info, upvote_ratio_thres = 0.5, upvote_confidence_thres=0.0, selected_subreddit_active_user_i_user_map=selected_subreddit_active_user_i_user_map, extra_input=extra_input, display = False)
    
    
    st.session_state['cached_predictions'] = (config, model, test_data, users_in_groups, submissions_before_curation, subreddit_test_submissions, selected_subreddit, group_centers, user_grouping_method, existing_votes, existing_user_updown_votes, selected_subreddit_active_user_i_user_map, extra_input, submission_sentiment_map, submission_class_map, submission_entity_map, reliability_bias_df, media_url_re, pred_group_votes_info, original_feature_map, all_submissions)
else:
    config, model, test_data, users_in_groups, submissions_before_curation, subreddit_test_submissions, selected_subreddit, group_centers, user_grouping_method, existing_votes, existing_user_updown_votes, selected_subreddit_active_user_i_user_map, extra_input, submission_sentiment_map, submission_class_map, submission_entity_map, reliability_bias_df, media_url_re, pred_group_votes_info, original_feature_map, all_submissions = st.session_state['cached_predictions']

batch_size = 1024
upvote_ratio_thres = st.sidebar.slider("Set the threshold for curator upvote rate", 0.0, 1.0,  config["upvote_ratio_thres"], step = 0.01) 
if type(config["upvote_confidence_thres"]) == list:
    config["upvote_confidence_thres"] = config["upvote_confidence_thres"][0]
config["upvote_confidence_thres"] = float(config["upvote_confidence_thres"])
upvote_confidence_thres = st.sidebar.slider("Set the threshold for upvote confidence", 0.0, 1.0, config["upvote_confidence_thres"], step = 0.01)


group_names = list(users_in_groups.keys())
if "manual" not in group_names:
    group_names.append("manual")
group_x = st.selectbox("Select a group of curators from our recommendations", group_names)
if group_x == "manual":
    available_usernames = original_feature_map["USERNAME"].values()
    inverse_username_map = {v:k for k,v in original_feature_map["USERNAME"].items()}
    chosen_users = st.multiselect("Manually select curators", available_usernames)
    users_in_groups["manual"] = {inverse_username_map[x] for x in chosen_users}

if st.button("Get curate posts!"):
    groups_preferred_submissions, groups_preferred_submissions_text, groups_submission_upvote_count_matrix = predict_groups_preferences(config, model, users_in_groups, submissions_before_curation, subreddit_test_submissions, selected_subreddit, group_centers=group_centers, user_grouping_method=user_grouping_method, existing_votes=existing_votes, existing_user_updown_votes=existing_user_updown_votes, pred_group_votes_info = pred_group_votes_info, upvote_ratio_thres = upvote_ratio_thres, upvote_confidence_thres=upvote_confidence_thres, selected_subreddit_active_user_i_user_map=selected_subreddit_active_user_i_user_map, extra_input=extra_input)



    max_show_posts = 30
    all_preferred_submissions_text = set.intersection(*[set(groups_preferred_submissions_text[group_x]) for group_x in groups_preferred_submissions_text])
    # groups_preferred_submissions_text


        

    top_preferred_submission_text = groups_preferred_submissions_text[group_x][:max_show_posts]
    top_preferred_submission_ids = groups_preferred_submissions[group_x][:max_show_posts]
    top_distinct_preferred_submission_text = [_ for i, _ in enumerate(groups_preferred_submissions_text[group_x]) if _ in set(groups_preferred_submissions_text[group_x]) - all_preferred_submissions_text][:max_show_posts]
    top_distinct_preferred_submission_ids = [groups_preferred_submissions[group_x][i] for i, _ in enumerate(groups_preferred_submissions_text[group_x]) if _ in set(groups_preferred_submissions_text[group_x]) - all_preferred_submissions_text][:max_show_posts]
    # TODO:
    show_text = top_preferred_submission_text  
    show_ids = top_preferred_submission_ids
    for i, submission_text in enumerate(show_text):
        user_bar, text_bar = st.columns([3, 10])
        id = show_ids[i]
        user_bar.write(f'##### {all_submissions[id]["USERNAME"] if "USERNAME" not in original_feature_map else original_feature_map["USERNAME"][all_submissions[id]["USERNAME"]]}')
        user_bar.write(all_submissions[id]["CREATED_TIME"])
        text_bar.warning(submission_text)

    print(f"Users in group {group_x} prefers {show_text}") # 



    st.write("Pearson ranking items:", groups_submission_upvote_count_matrix.index.to_list())
    groups_submission_upvote_count_matrix_nonzero = groups_submission_upvote_count_matrix[groups_submission_upvote_count_matrix.sum(axis = 1) != 0]
    group_preference_pearson_corr = np.corrcoef(groups_submission_upvote_count_matrix_nonzero) # (697, 697)
    st.write(group_preference_pearson_corr)



    visualize_group_preferences(groups_preferred_submissions, test_data, user_grouping_method, submission_sentiment_map = submission_sentiment_map, submission_class_map=submission_class_map, submission_entity_map=submission_entity_map, reliability_bias_df=reliability_bias_df, media_url_re=media_url_re)