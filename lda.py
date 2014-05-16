## Code for predicting topics using the LDA implementation provided by GenSim
from gensim import corpora, models, similarities
import csv
import numpy as np

# Loads in the documents from our CSV file of training data
data = np.array(list( csv.reader( open(  'train.csv', 'rU' ) ) ))

# Convert the body we want to predict the topic for into a list for GenSim
texts_matrix = data[:,3]
texts = texts_matrix.tolist()
# StopWords to ignore, important for the perfromance of LDA
stoplist = """a,able,about,across,after,all,almost,also,am
        ,among,an,and,any,are,as,at,be,because
        ,been,but,by,can,cannot,could,dear
        ,did,do,does,either,else,ever,every
        ,for,from,get,got,had,has,have,he,her
        ,hers,him,his,how,however,i,if,in,into
        ,is,it,its,just,least,let,like,likely,
        may,me,might,most,must,my,neither,no,nor,
        not,of,off,often,on,only,or,other,our,own,
        rather,said,say,says,she,should,since,so,
        some,than,that,the,their,them,then,there,
        these,they,this,tis,to,too,twas,us,wants,
        was,we,were,what,when,where,which,while,
        who,whom,why,will,with,would,yet,you,your""".split(",")
# Remove stopwords from the docs
texts = [[word for word in document.lower().split() if word not in stoplist]
    for document in texts]
texts = [[word for word in text ] for text in texts]
# Create a dictionary for GenSim using our data
dictionary = corpora.Dictionary(texts)
# Convert the data into a BOW representation
corpus = [dictionary.doc2bow(text) for text in texts]
# Train the LDA model
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=150, update_every=0, chunksize=1, passes=50)

# Now get the predictions for topics from the model for each document, this output is redirect on the terminal to a CSV file and then manually appended to the training data
for index, text in enumerate(corpus):
    topics = []
    topic_added = False
    for topic in lda[text]:
        topics.append(topic[0])
        topic_added = True
    if not topic_added:
        topics.append(-1)
    print topics
