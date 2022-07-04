import praw
from superdebug import debug
reddit = praw.Reddit(
    client_id="kmyK_5R42klZo-WpcwX1xA",
    client_secret="L7Wj40LXF7d8StPvESo9vpzxo3wAzw",
    password="Azvrdtlibera4",
    user_agent="testscript by u/Azure-Vision",
    username="Azure-Vision",
)
print(reddit.user.me())
submission=reddit.submission(url = "https://www.reddit.com/r/houston/comments/e7gje4/downtown_houston_at_sunset/")
first_comment = submission.comments._comments[0]
debug(first_comment_text = first_comment.body)