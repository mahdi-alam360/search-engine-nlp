import json
from pathlib import Path
from app import SearchIndex
index = SearchIndex(json.loads(Path('corpus.json').read_text()))
cases = [('BM25 document retrieval','retrieval'),('image registration','registration'),('curriculum learning','curriculum'),('colour clustering','clustering')]
hits = sum(index.search(query)[0]['id']==expected for query,expected in cases)
print(json.dumps({'toy_corpus_documents':len(index.documents),'handwritten_queries':len(cases),'top1_hits':hits,'top1_accuracy':hits/len(cases),'unknown_query_abstains':index.answer('quasar')['mode']=='no_evidence','scope':'Functional smoke evaluation; not an independent benchmark'},indent=2))
