import unittest
from content_model import recommend


class ContentModelTests(unittest.TestCase):
    def setUp(self):
        self.repos = [
            {'full_name': 'a/pipeline', 'description': 'Python data pipelines',
             'html_url': 'https://github.com/a/pipeline', 'stargazers_count': 1},
            {'full_name': 'b/game', 'description': 'Graphics game engine',
             'html_url': 'https://github.com/b/game', 'stargazers_count': 100}]

    def test_relevance_before_popularity(self):
        result = recommend(self.repos, 'data python')
        self.assertEqual(result[0]['repository'], 'a/pipeline')
        self.assertEqual(result[0]['matching_terms'], ['data', 'python'])

    def test_unknown_interests_fall_back_to_popularity(self):
        result = recommend(self.repos, 'unseenword')
        self.assertEqual(result[0]['repository'], 'b/game')
        self.assertEqual(result[0]['score'], 0)

    def test_empty_catalog(self):
        self.assertEqual(recommend([], 'python'), [])
