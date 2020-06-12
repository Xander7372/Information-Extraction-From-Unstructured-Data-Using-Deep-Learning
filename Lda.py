from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.models import CoherenceModel
import gensim
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import pyLDAvis.gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
f1 = open("/content/drive/My Drive/Colab Notebooks/tweet_desc2.txt","r")
f2= open("/content/drive/My Drive/Colab Notebooks/tweet_img2.txt","r")
doc_b = f1.read()
doc_a = f2.read()
# compile sample documents into a list
doc_set = [doc_a, doc_b]

# list for tokenized documents in loop
texts_lem = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts_lem.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary1 = corpora.Dictionary(texts_lem)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary1.doc2bow(text) for text in texts_lem]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary1, passes=1000)
print(ldamodel.print_topics(num_topics=2, num_words=4))

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stopped_tokens,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=5,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = ldamodel.show_topics(formatted=False)

fig, axes = plt.subplots(1, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary=ldamodel.id2word, mds='mmds')
vis 

coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts_lem, dictionary=dictionary1, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)