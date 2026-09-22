from feeds import fetch,run
from content_model import recommend

def acquire():
    return {'sources':[fetch('https://api.github.com/search/repositories?q=topic%3Amachine-learning&sort=updated&per_page=50')]}


def analyze(snapshot):
    repos=snapshot['sources'][0]['payload']['items']
    interests='python machine learning data engineering pipelines'
    return {'project':'CatalogRecommender','real_repository_candidates':len(repos),'interests':interests,
            'recommendations':recommend(repos,interests,10),
            'note':'Live content-based recommendations from current public repository metadata. Offline collaborative filtering and chronological evaluation are separate in engine.py and demo.py. No live relevance labels exist, so no online accuracy metric is claimed.'}


if __name__=='__main__': run(acquire,analyze)
