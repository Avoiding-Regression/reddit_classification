from tqdm import tqdm

def get_subreddit_posts(reddit_client, subreddit_name, post_limit):
    posts_list = []
    subreddit_posts = reddit_client.subreddit(subreddit_name).hot(limit=post_limit)
    for post in tqdm(subreddit_posts, total=post_limit):
        posts_list.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    return posts_list
