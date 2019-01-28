import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import os
import string
import gensim
from gensim import corpora

# TODO add a frequency filter

r_path = 'Speeches/Republican'
speech_list = []
for speech in os.listdir(r_path):
    file = open(r_path + '/' + speech, 'r')
    speech_list.append(file.read())

d_path = 'Speeches/Democrat'
for speech in os.listdir(d_path):
    file = open(d_path + '/' + speech, 'r')
    speech_list.append(file.read())


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in speech_list]

# Assign every unique term an index
dictionary = corpora.Dictionary(doc_clean)
# Create term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Create the LDA model object
Lda = gensim.models.ldamodel.LdaModel

# Train model on term matrix
ldamodel = Lda(doc_term_matrix, num_topics=4, id2word=dictionary, passes=50)
print(ldamodel.print_topics(num_topics=4, num_words=3))
