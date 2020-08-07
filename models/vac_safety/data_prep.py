import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim.models.keyedvectors as word2vec
import gc

MAX_FEATURES = 20000

def loadEmbeddingMatrix(typeToLoad, tokenizer):
    if(typeToLoad=="word2vec"):
        path = "./models/vac_safety/embedding_data/GoogleNews-vectors-negative300.bin"
        word2vecDict = word2vec.KeyedVectors.load_word2vec_format(\
            path, binary=True)
        embed_size = 300
        
    embeddings_index = dict()
    for word in word2vecDict.wv.vocab:
        embeddings_index[word] = word2vecDict.word_vec(word)
    print('Loaded %s word vectors.' % len(embeddings_index))

    gc.collect()
    #We get the mean and standard deviation of the embedding weights so that we could maintain the 
    #same statistics for the rest of our own random generated weights. 
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    
    nb_words = len(tokenizer.word_index)
    #We are going to set the embedding size to the pretrained dimension as we are replicating it.
    #the size will be Number of Words in Vocab X Embedding Size
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    gc.collect()

    #With the newly created embedding matrix, we'll fill it up with the words that we have in both 
    #our own dictionary and loaded pretrained embedding. 
    embeddedCount = 0
    for word, i in tokenizer.word_index.items():
        i-=1
        #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
        embedding_vector = embeddings_index.get(word)
        #and store inside the embedding matrix that we will train later on.
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
            embeddedCount+=1
    print('total embedded:',embeddedCount,'common words')
    
    del(embeddings_index)
    gc.collect()
    
    #finally, return the embedding matrix
    return embedding_matrix

def prepare_data(train, test):
    for fp in train:
        train_pd = pd.read_csv(fp)
    
    for fp in test:
        test_pd = pd.read_csv(fp)

    list_classes = \
        ["toxic", "severe_toxic", "obscene", "threat", \
        "insult", "identity_hate"]
    y    = train_pd[list_classes].values
    # y_te = test_pd[list_classes].values

    list_sentences_train = train_pd["comment_text"]
    list_sentences_test = test_pd["comment_text"]
    
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(\
        list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(\
        list_sentences_test)
    
    maxlen = 200
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

    embedding_matrix = loadEmbeddingMatrix('word2vec',\
        tokenizer=tokenizer)

    return (X_t, y, X_te, list_sentences_test, list_classes, \
        embedding_matrix, tokenizer)

    