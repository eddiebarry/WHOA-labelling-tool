import os
import keras
import numpy as np
from data_prep import prepare_data
from keras.layers import Dense, Input, LSTM, \
    Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, \
    optimizers, layers



class VacSafetyModel:
    def __init__(self, batch_size=16, weight_dir=None):
        self.train_files = None
        self.test_files  = None
        self.train_data  = None
        self.test_data   = None
        self.maxlen      = None
        self.tokenizer   = None
        self.batch_size  = batch_size
        self.model       = None
        self.labels      = None
        self.embedding_matrix = None
        

    
    def set_data(self, train, test, batch_size=None, maxlen=200, \
        weight_dir = None):
        self.train_files = train
        self.test_files  = test
        if batch_size:
            self.batch_size = batch_size
        
        self.train_data, self.train_labels, self.test_data, \
            self.test_sentences, \
            self.labels, \
            self.embedding_matrix, \
            self.tokenizer = prepare_data( \
            train = self.train_files, \
            test  = self.test_files)
        
        if weight_dir:
            self.weight_dir = weight_dir
        
        self.model = self.define(weight_dir)
    
    def train(self, lr=1e-5, optim="Adam", loss="BCE", epochs=1):
        if loss == "BCE":
            loss ='binary_crossentropy'

        if optim == "Adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'])

        hist = self.model.fit(self.train_data, self.train_labels, \
            batch_size=self.batch_size, \
            epochs=epochs, validation_split=0.1)

        return hist
        

    def test(self,save_dir):
        self.result_dir = os.path.join(save_dir,'results.csv')
        preds = self.model.predict(self.test_data[:100])
        print("predictions done")

        # save to numpy
        text = np.array(self.test_sentences,dtype='str')
        text = np.expand_dims(text, axis=1)
        text_and_confs = np.concatenate((text,preds),axis=1)
        
        # save in dir
        names = ['text']
        names = names + self.train_labels

        # save confidences and text
        test_df = pd.DataFrame(text_and_confs,columns=names)
        test_df.to_csv(self.result_dir)
        

    def define(self, weight_dir=None):
        #maxlen=200 as defined earlier
        inp = Input(shape=(self.maxlen, )) 
        
        x = Embedding(len(self.tokenizer.word_index), \
            self.embedding_matrix.shape[1], \
            weights=[self.embedding_matrix], \
            trainable=False)(inp)
        
        x = Bidirectional(LSTM(60, \
            return_sequences=True, \
            name='lstm_layer', \
            dropout=0.1, \
            recurrent_dropout=0.1))(x)
        
        x = GlobalMaxPool1D()(x)

        x = Dropout(0.1)(x)

        x = Dense(50, activation="relu")(x)

        x = Dropout(0.1)(x)

        x = Dense(6, activation="sigmoid")(x)

        model = Model(inputs=inp, outputs=x)

        if weight_dir:
            # load weights
            model.load_weights(weight_dir)
            print("weights loaded")
        
        return model