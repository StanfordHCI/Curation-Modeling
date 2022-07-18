from collections import defaultdict
from email.policy import default
import os
import pickle
import time
import pandas as pd
import praw
from superdebug import debug
from tqdm import tqdm
from pmaw import PushshiftAPI
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

submission_text_path = "data/reddit/submission_text.db"
engine = create_engine(f"sqlite:///{submission_text_path}")
DBSession = sessionmaker(bind=engine)
session = DBSession()

Base = declarative_base()
class Submissions(Base):
    __tablename__ = 'submissions'
    id = Column(String, primary_key=True, autoincrement=True)
    text = Column(String)

def get_single_submission_text(submission_id):
    submission_id = submission_id.split("_")[-1]
    # submission=reddit.submission(url = "https://www.reddit.com/r/houston/comments/e7gje4/downtown_houston_at_sunset/")
    submission_text = ""
    success = False
    try_time = 0
    while not success and try_time < 5:
        submission=reddit.submission(submission_id)
        try:
            submission_text = submission.title
            selftext = submission.selftext
            if selftext != "":
                submission_text += " [SEP] " + selftext
            success = True
        except Exception as e:
            print(f"Error: {e}")
            try_time += 1
            time.sleep(5)
    return submission_text
    debug(submission_title = submission.title, submission_texts=submission.selftext)
    first_comment = submission.comments._comments[0]
    debug(first_comment_text = first_comment.body)
def build_submission_text_dict(submission_ids, submission_texts, submission_text_dict_path):
    submission_text_dict = {}
    for i, submission_text in enumerate(submission_texts):
        submission_text_dict[submission_ids[i]] = submission_text
    pickle.dump(submission_text_dict, open(submission_text_dict_path, "wb"))
"""
def get_batch_submission_text(submission_ids, submission_text_dict_path):
    submission_ids = list(submission_ids)
    submission_texts = []
    if os.path.exists(submission_text_dict_path):
        submission_text_dict = pickle.load(submission_text_dict_path)
        for submission_id in submission_ids:
            submission_texts.append(submission_text_dict[submission_id])
        return submission_texts
    else:
        for start in tqdm(range(0, len(submission_ids), 100)):
            part_submission_ids = submission_ids[start: start + 100]
            i = 0
            try:
                for submission in reddit.info(fullnames=part_submission_ids):
                    while part_submission_ids[i].split("_")[-1] != submission.id:
                        submission_texts.append("")
                        i += 1
                    submission_text = submission.title
                    selftext = submission.selftext
                    if selftext != "":
                        submission_text += " [SEP] " + selftext
                    submission_texts.append(submission_text)
                    i += 1
            except:
                for i in range(len(submission_texts), start + 100):
                    submission_text = get_single_submission_text(submission_ids[i])
                    submission_texts.append(submission_text)
        assert len(submission_texts) == len(submission_ids)
        build_submission_text_dict(submission_ids, submission_texts, submission_text_dict_path)
        return submission_texts
"""
def store_batch_submission_text_praw(submission_ids):
    submission_texts = []
    i = 0
    try:
        for submission in reddit.info(fullnames=submission_ids):
            while submission_ids[i].split("_")[-1] != submission.id:
                submission_texts.append("")
                i += 1
            submission_text = submission.title
            selftext = submission.selftext
            if selftext != "":
                submission_text += " [SEP] " + selftext
            submission_texts.append(submission_text)
            i += 1
        submission_texts.extend([""] * (len(submission_ids) - len(submission_texts)))
    except:
        for submission_id in submission_ids:
            submission_text = get_single_submission_text(submission_id)
            submission_texts.append(submission_text)
    # submission_id_text_map = {}
    for i in range(len(submission_ids)):
        # submission_id_text_map[submission_ids[i]] = submission_texts[i]
        session.add(Submissions(id=submission_ids[i], text=submission_texts[i]))
    # return submission_id_text_map

def store_batch_submission_text_pmaw(submission_ids):
    short_id_map = {id.split("_")[-1]:id for id in submission_ids}
    remain_ids = set(submission_ids)
    existing_ids = session.query(Submissions.id).filter(Submissions.id.in_(remain_ids)).all()
    remain_ids = remain_ids - {_[0] for _ in existing_ids}
    # submission_id_text_map = {}
    posts = pmaw_api.search_submissions(ids=list(remain_ids))
    for post in tqdm(posts):
        submission_id = post["id"]
        submission_text = post["title"]
        if "selftext" in post and post["selftext"] != "":
            submission_text += " [SEP] " + post["selftext"]
        # submission_id_text_map[short_id_map[submission_id]] = submission_text
        if short_id_map[submission_id] in remain_ids:
            session.add(Submissions(id=short_id_map[submission_id], text=submission_text))
            remain_ids.remove(short_id_map[submission_id])
    session.commit()
    return remain_ids # submission_id_text_map, 

def store_all_submission_text(submission_ids, batch_size = 10000, ret = "list"):
    submission_ids = list(submission_ids) # [id.split("_")[-1] for id in list(submission_ids)]
    remain_ids = set()
    """
    for start_i in tqdm(range(0, len(submission_ids), batch_size)):
        batch_remains = store_batch_submission_text_pmaw(submission_ids[start_i : start_i + batch_size])
        remain_ids.update(batch_remains)
    """
    remain_ids = set(submission_ids)
    for start_i in tqdm(range(0, len(submission_ids), batch_size)):
        existing_ids = session.query(Submissions.id).filter(Submissions.id.in_(set(submission_ids[start_i : start_i + batch_size]))).all()
        remain_ids = remain_ids - {_[0] for _ in existing_ids}
    
    if len(remain_ids) > 0:
        remain_ids = list(remain_ids)
        for start_i in tqdm(range(0, len(remain_ids), batch_size)):
            store_batch_submission_text_praw(remain_ids[start_i : start_i + batch_size])
    return
    pickle.dump(remain_ids, open("data/reddit/tmp_remain_ids.pt", "wb"))
def get_batch_submission_text(submission_ids):
    submission_id_text_map = {}
    existing_items = session.query(Submissions).filter(Submissions.id.in_(submission_ids)).all()
    for item in existing_items:
        submission_id_text_map[item.id] = item.text
    return [submission_id_text_map.get(id, "") for id in submission_ids]

if __name__ == "__main__":
    pmaw_api = PushshiftAPI()
    reddit = praw.Reddit(
        client_id="kmyK_5R42klZo-WpcwX1xA",
        client_secret="L7Wj40LXF7d8StPvESo9vpzxo3wAzw",
        password="Azvrdtlibera4",
        user_agent="testscript by u/Azure-Vision",
        username="Azure-Vision",
    )
    print(reddit.user.me())
    vote_data = pd.read_csv('data/reddit/submission_info.txt', sep = '\t')
    debug("Read vote data!")
    store_all_submission_text(vote_data['SUBMISSION_ID'], ret = "dict")


    