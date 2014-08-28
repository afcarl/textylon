from gensim.corpora.wikicorpus import WikiCorpus


wiki = WikiCorpus('', processes=None, lemmatize=False, dictionary=None)
texts = wiki.get_texts()
with open('wikitext.txt', 'w') as wikitext:
    for text in texts:
        wikitext.write(' '.join(text) + "\n")
    
