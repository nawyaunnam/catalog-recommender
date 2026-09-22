import unittest
from engine import Recommender, temporal_split, evaluate

class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.model=Recommender([('u','a',1),('v','a',1),('v','b',2),('w','c',1)])
    def test_exclude_seen(self):
        rows=self.model.recommend('u')
        self.assertNotIn('a',[r['item'] for r in rows])
        self.assertEqual(rows[0]['item'],'b')
        self.assertEqual(rows[0]['because'],'a')
    def test_cold_start(self):
        self.assertEqual(self.model.recommend('new')[0]['item'],'a')
    def test_holdout_leakage(self):
        train,test=temporal_split([('u','a',1),('u','a',2),('u','b',3)])
        self.assertEqual([x[1] for x in train],['a'])
        self.assertEqual([x[1] for x in test],['b'])
    def test_same_timestamp_not_leaked(self):
        train,test=temporal_split([('u','a',1),('u','b',2),('u','c',2)])
        self.assertTrue(all(x[2]<min(y[2] for y in test) for x in train))
    def test_unknown_catalog_targets_count_as_misses(self):
        result=evaluate(self.model,[('u','unknown',3)],1)
        self.assertEqual(result['item_cosine']['hit_rate'],0)
        self.assertEqual(result['unseen_catalog_targets'],1)
    def test_duplicate_interaction_does_not_change_model(self):
        a=Recommender([('u','a',1),('v','a',1),('v','b',2)])
        b=Recommender([('u','a',1),('v','a',1),('v','b',2),('v','b',3)])
        self.assertEqual(a.recommend('u'),b.recommend('u'))
