"""Content-based cold-start recommendations from real repository metadata."""
import math
import re
from collections import Counter

def terms(text):return re.findall(r'[a-z0-9]+',text.lower())

def recommend(repos,interests,k=10):
    if k<1:raise ValueError('k must be positive')
    bags=[Counter(terms(' '.join([r['full_name'],r.get('description') or '', ' '.join(r.get('topics') or [])]))) for r in repos]
    df=Counter(t for bag in bags for t in bag)
    idf={t:math.log((1+len(bags))/(1+n))+1 for t,n in df.items()}
    query=Counter(terms(interests))
    q={t:count*idf[t] for t,count in query.items() if t in idf}
    qnorm=math.sqrt(sum(v*v for v in q.values()))
    results=[]
    for repo,bag in zip(repos,bags):
        vector={t:(1+math.log(count))*idf[t] for t,count in bag.items()}
        norm=math.sqrt(sum(v*v for v in vector.values()))
        score=sum(v*vector.get(t,0) for t,v in q.items())/(qnorm*norm) if qnorm and norm else 0
        results.append({'repository':repo['full_name'],'url':repo['html_url'],'score':round(score,6),
                        'matching_terms':sorted(set(q)&set(bag)),'stars':repo.get('stargazers_count',0),
                        'updated_at':repo.get('updated_at')})
    return sorted(results,key=lambda x:(-x['score'],-x['stars'],x['repository']))[:k]
