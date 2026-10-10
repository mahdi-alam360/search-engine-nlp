import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
from app import SearchIndex, make_handler

class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = SearchIndex(json.loads(Path('corpus.json').read_text()))
        cls.server = ThreadingHTTPServer(('127.0.0.1',0),make_handler(cls.index))
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
    def test_relevance(self):
        for query, expected in [('BM25 document retrieval','retrieval'),('image registration','registration'),('curriculum learning','curriculum'),('colour clustering','clustering')]:
            self.assertEqual(self.index.search(query)[0]['id'],expected)
    def test_no_evidence(self):
        self.assertEqual(self.index.answer('quasar')['mode'],'no_evidence')
    def test_empty_corpus(self):
        self.assertEqual(SearchIndex([]).search('images'),[])
    def test_validation(self):
        for query,k in [('',3),('x'*1001,3),('query',0),('query',True)]:
            with self.assertRaises(ValueError): self.index.search(query,k)
    def test_search_api(self):
        with urlopen(self.base+'/search?q=BM25&k=1') as response:
            self.assertEqual(json.load(response)['results'][0]['id'],'retrieval')
    def test_answer_api(self):
        req = Request(self.base+'/answer',data=json.dumps({'query':'BM25'}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req) as response:
            result=json.load(response)
        self.assertEqual(result['mode'],'extractive')
        self.assertIn('[retrieval]',result['answer'])
    def test_invalid_json(self):
        req = Request(self.base+'/answer',data=b'{broken',headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as caught: urlopen(req)
        self.assertEqual(caught.exception.code,400)
    def test_model_failure(self):
        from unittest.mock import patch
        with patch('app.urlopen',side_effect=OSError('offline')):
            with self.assertRaises(OSError): self.index.answer('BM25',model='example')

if __name__ == '__main__': unittest.main()
