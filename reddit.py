import praw
from superdebug import debug
from tqdm import tqdm
reddit = praw.Reddit(
    client_id="kmyK_5R42klZo-WpcwX1xA",
    client_secret="L7Wj40LXF7d8StPvESo9vpzxo3wAzw",
    password="Azvrdtlibera4",
    user_agent="testscript by u/Azure-Vision",
    username="Azure-Vision",
)
print(reddit.user.me())
def get_single_submission_text(submission_id):
    submission_id = submission_id.split("_")[-1]
    submission=reddit.submission(submission_id)
    # submission=reddit.submission(url = "https://www.reddit.com/r/houston/comments/e7gje4/downtown_houston_at_sunset/")
    submission_text = submission.title
    selftext = submission.selftext
    if selftext != "":
        submission_text += " [SEP] " + selftext
    return submission_text
    debug(submission_title = submission.title, submission_texts=submission.selftext)
    first_comment = submission.comments._comments[0]
    debug(first_comment_text = first_comment.body)
def get_batch_submission_text(submission_ids):
    submission_ids = list(submission_ids)
    submission_texts = []
    for start in range(0, len(submission_ids), 100):
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
    return submission_texts
if __name__ == "__main__":
    debug(get_single_submission_text("t3_e0i7l4"))
    debug(get_single_submission_text("t3_d8vv6s"))
    debug(get_batch_submission_text(["t3_e0i7l4", "t3_d8vv6s"]))
    