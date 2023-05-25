from ADAboost import adaboost_model
from knn import k_nearest_neighbors
from xgboost import xgboost_model
from bert import bert_model
from sgd import sgd_model


def main():
    reddit = praw.Reddit(client_id=praw_id, client_secret=praw_secret, user_agent='test scraper app')

    gamestop_posts = get_subreddit_posts(reddit, 'GameStop', 100)
    tesla_posts = get_subreddit_posts(reddit, 'teslainvestorsclub', 100)

    col_names = ['title', 'score','id','subreddit','url','num_comments', 'body', 'created']
    gamestop_frame = pd.DataFrame(gamestop_posts, columns=col_names)
    tesla_frame = pd.DataFrame(tesla_posts, columns=col_names)

    gamestop_frame.to_csv("gamestop.csv")
    tesla_frame.to_csv("tesla.csv")

if __name__ == "__main__":
    main()