"""Implicit item similarity; all model statistics are fit on training events."""
import math
from collections import defaultdict

def temporal_split(events):
    by_user = defaultdict(list)
    for user,item,timestamp in events:
        by_user[user].append((user,item,timestamp))
    train, test = [], []
    for user,rows in sorted(by_user.items()):
        rows.sort(key=lambda x:(x[2],x[1]))
        # Keep only each item's first interaction; repeated clicks are not new targets.
        seen, unique = set(), []
        for row in rows:
            if row[1] not in seen: unique.append(row); seen.add(row[1])
        if len(unique) >= 2:
            cutoff = unique[-1][2]
            earlier = [r for r in unique if r[2] < cutoff]
            if earlier:
                train.extend(earlier)
                test.extend(r for r in unique if r[2] == cutoff)
            else: train.extend(unique)
        else: train.extend(unique)
    return train,test

class Recommender:
    def __init__(self, events):
        self.history, self.users = defaultdict(set), defaultdict(set)
        for user,item,_ in events:
            self.history[user].add(item)
            self.users[item].add(user)
        self.similarities = {}
        items = sorted(self.users)
        for i,item in enumerate(items):
            for other in items[i+1:]:
                common = len(self.users[item] & self.users[other])
                if common:
                    similarity = common / math.sqrt(len(self.users[item])*len(self.users[other]))
                    self.similarities[item,other] = self.similarities[other,item] = similarity
    def recommend(self,user,k=5,popularity_only=False):
        if k < 1: raise ValueError('k must be positive')
        seen = self.history.get(user,set())
        results = []
        for item in sorted(set(self.users)-seen):
            contributions = [(self.similarities.get((source,item),0),source) for source in sorted(seen)]
            score = 0 if popularity_only else sum(x[0] for x in contributions)
            strongest = max(contributions,default=(0,None))
            results.append({'item':item,'score':score,'popularity':len(self.users[item]),
                'because':strongest[1] if score>0 else 'popularity fallback'})
        return sorted(results,key=lambda x:(-x['score'],-x['popularity'],x['item']))[:k]

def evaluate(model, holdout, k=5):
    targets = defaultdict(set)
    for user,item,_ in holdout: targets[user].add(item)
    if not targets: raise ValueError('holdout required')
    result = {'users':len(targets),'k':k,'unseen_catalog_targets':sum(item not in model.users for items in targets.values() for item in items)}
    for name,popularity in [('item_cosine',False),('popularity_baseline',True)]:
        hits,rr = [],[]
        for user,relevant in sorted(targets.items()):
            ranking = [x['item'] for x in model.recommend(user,k,popularity)]
            hits.append(bool(set(ranking)&relevant))
            rr.append(next((1/i for i,item in enumerate(ranking,1) if item in relevant),0))
        result[name] = {'hit_rate':sum(hits)/len(hits),'mrr':sum(rr)/len(rr)}
    return result
