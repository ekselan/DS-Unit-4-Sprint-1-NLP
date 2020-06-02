# DS-Unit-4-Sprint-1-NLP

## Setting Up Conda Environment

From Command Line inside folder where requirements.txt file is located:
```sh
conda create -n conda-env-name python==3.8
```
- Activate environment:
```sh
conda activate conda-env-name
```
- Add packages for this sprint:
```sh
pip install -r requirements.txt
```
- Add Ipython Kernel reference for use from JupyterLab:
```sh
python -m ipykernel install --user --name conda-env-name --display-name "Desired Display Name"
```
- Install spacy models:
```sh
python -m spacy download en_core_web_md
```
```sh
python -m spacy download en_core_web_lg
```
- Deactivate conda env and the run:
```sh
jupyter lab
```

## Using Spacy
- Import Statements:
```py
from collections import Counter

import squarify
import matplotlib.pyplot as plt

import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_lg")
```
- Count Function:
```py
def count(docs):
    """
    Function which takes a corpus of document and returns a dataframe
    of word counts to analyze.
    """

    word_counts = Counter()
    appears_in = Counter()

    total_docs = len(docs)

    for doc in docs:
        word_counts.update(doc)
        appears_in.update(set(doc))

    temp = zip(word_counts.keys(), word_counts.values())

    wc = pd.DataFrame(temp, columns = ['word', 'count'])

    wc['rank'] = wc['count'].rank(method='first', ascending=False)
    total = wc['count'].sum()

    wc['pct_total'] = wc['count'].apply(lambda x: x / total)

    wc = wc.sort_values(by='rank')
    wc['cul_pct_total'] = wc['pct_total'].cumsum()

    t2 = zip(appears_in.keys(), appears_in.values())
    ac = pd.DataFrame(t2, columns=['word', 'appears_in'])
    wc = ac.merge(wc, on='word')

    wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)

    return wc.sort_values(by='rank')
```
- Tokenizer Pipeline:
```py
# Tokenizer
tokenizer = Tokenizer(nlp.vocab)

# Tokenizer Pipe

tokens = []

""" Make them tokens """
for doc in tokenizer.pipe(df['texts'], batch_size=500):
    doc_tokens = [token.text for token in doc]
    tokens.append(doc_tokens)

df['tokens'] = tokens
```
- Create Word Counts DF:
```py
wc = count(df['tokens'])
```
- Visualize Top N Words:
```py
wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()
```
---

## Dealing with Stop Words:

### Custom Stop Word Technique:
- Update tokens w/o Stop Words:
```py
tokens = []

""" Update those tokens w/o stopwords"""
for doc in tokenizer.pipe(df['texts'], batch_size=500):
    
    doc_tokens = []
    
    for token in doc:
        if (token.is_stop == False) & (token.is_punct == False):
            doc_tokens.append(token.text.lower())

    tokens.append(doc_tokens)

df['tokens'] = tokens
```
- Extend Stop Words:
```py
STOP_WORDS = nlp.Defaults.stop_words.union(
    [" ", 
     "1", 
     "check-in", 
     "austin", 
     "i'm", 
     "i've",
     "it's",
     "-"])
```
- Update tokens:
```py
tokens = []

for doc in tokenizer.pipe(df['texts'], batch_size=500):
    
    doc_tokens = []
    
    for token in doc: 
        if token.text.lower() not in STOP_WORDS:
            doc_tokens.append(token.text.lower())
   
    tokens.append(doc_tokens)
    
df['tokens'] = tokens
```
### Lemmatization Technique
- Get Lemmas Function:
```py
def get_lemmas(text):

    lemmas = []
    
    doc = nlp(text)
    
    # Something goes here :P
    for token in doc: 
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON'):
            lemmas.append(token.lemma_)
    
    return lemmas
df['lemmas'] = df['texts'].apply(get_lemmas) #> Cannot apply to tokens, must be applied to strings, not list.
```










