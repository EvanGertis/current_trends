# Topic Modeling

This project is for Current Trends in computing. The goal of this research is to develop a text mining tool that can extract topics from reddits. The hypothesis behind this study is that using topic modeling we can identify philosophical arguments within reddits.

### Literature Review

[Literature Review](https://www.researchgate.net/publication/354854228_Review_of_Topic_Modeling_Journals_I_READING_TEA_LEAVES_HOW_HUMANS_INTERPRET_TOPIC_MODELS)


### Contribution to the field
As the field of philosophy grows it is imperative that we develop an understanding of the social implications of philosophical arugments playing out in online communities. a practical application of topic modeling is managing, organizing, and annotating   large archives of texts. This project aims to extract topics that describe philosophical positions for a given dataset taken from reddit.com. 

### How does LDA work?
LDA is a generative probablistic model that assumes that each topic is a mixture over an underlying set of words, and each document is a mixture of a set of probabilities. For example, if we take M documents consisting of N words and K topics then the model uses these parameters to train the output.

K: number of topics
N: number of words in the document
M: the number of documents to analyze
alpha: the Dirchlet-prior concentration parameter of the per-document topic distribution
beta: the same parameter of the per-topic distribution
phi(k): word distribution for topic k
theta(i): the opic distribtuion for document i
z(i,j): the topic assignment for word w(i,j)
w(i,j): the j word in the ith document 
phi and theta re the dirchlet distributions
z and w are the multinomials

The alpha parameter is known as the dirchlet prior concentration parameter. It represents document-topic density. With a high alpha, documents are assumed to be made up of more topiics and result in more specific topic distributioon per document.

The beta parameter is a prior concentration parameter that represents topic-word distribution. With a high beta, topics are assumed to be made up of most of the words and result in a more specific word distribution per topic. 
### Implementation
In this example we load the data, clean the data, explore the data, prepare for LDA analysis, perform LDA model training and analyze the LDA model results. 