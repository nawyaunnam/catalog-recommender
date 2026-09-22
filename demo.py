import json
from engine import Recommender, temporal_split, evaluate

events = [('u1','python',1),('u1','sql',2),('u1','pipelines',3),
          ('u2','python',1),('u2','sql',2),('u2','pipelines',4),
          ('u3','sql',1),('u3','pipelines',2),('u3','python',5),
          ('u4','python',1),('u4','testing',2),('u4','sql',6),
          ('u5','testing',1),('u5','python',2),('u5','pipelines',7)]
train, test = temporal_split(events)
model = Recommender(train)
print(json.dumps({'recommendations':model.recommend('u1',3),
                  'cold_start':model.recommend('new-user',3),
                  'holdout_evaluation':evaluate(model,test,k=2)}, indent=2))
