from api import praw_id, praw_secret
from data_helper import get_subreddit_posts

import praw
import pandas as pd
from data_cleaning import data_cleaning

def data_generation():
    reddit = praw.Reddit(client_id=praw_id, client_secret=praw_secret, user_agent='test scraper app')

    gamestop_posts = get_subreddit_posts(reddit, 'GameSgit chtop', 100)
    tesla_posts = get_subreddit_posts(reddit, 'teslainvestorsclub', 100)

    col_names = ['title', 'score','id','subreddit','url','num_comments', 'body', 'created']
    gamestop_frame = pd.DataFrame(gamestop_posts, columns=col_names)
    tesla_frame = pd.DataFrame(tesla_posts, columns=col_names)

    gs = gamestop_frame.to_csv("gamestop.csv")
    tesla = tesla_frame.to_csv("tesla.csv")

    gs_tokens, gs = data_cleaning(gs)
    tesla_tokens, tesla = data_cleaning(tesla)

    frames = [tesla, gs]
    final_df = pd.concat(frames)

    # set labels

    final_df['labels'] = np.where(final_df['subreddit'] == 'teslainvestorsclub', 1, 0)
    final_df['text'] = final_df['text_string_lem']

    return final_df