from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from collections import Counter
from pipeline import ContractPipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import pandas as pd
from directoryassistor import DirectoryAssistor
import matplotlib.pyplot as plt
import time
import pickle
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

plt.style.use('tableau-colorblind10')


class NaiveBayes():

    def __init__(self, train_dir, test_dir, stop_words, lemmatize=False, 
                 stem=False):
        """
        Instantiates a RandomForest object and instantiates ContractPipelines
        for each directory
        
        Params:
        train_dir: str directory where the training set is held must 
        end with '/'
        test_dir:  str directory where the testing set is held must
        end with '/'
        houldout_dir:  str directory where the holdout set is held must 
        end with '/'
        """

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.stop_words = stop_words
        self.train_pipe = ContractPipeline(self.train_dir, self.stop_words)
        self.test_pipe = ContractPipeline(self.test_dir, self.stop_words)
        self.model = None
        self.lemmatize = lemmatize
        self.stem = stem

    def convert_txt_to_lists(self):
        """
        If the documents are in .txt format use this to call the 
        get_list_of_txts on each of the directorys
        """
        
        self.train_pipe.get_list_of_txts()
        self.test_pipe.get_list_of_txts()
    
    def convert_doc_to_lists(self):
        """
        If the documents are in .doc format use this to call the 
        get_list_of_txts on each of the directorys
        """
        
        self.train_pipe.get_list_of_docs()
        self.test_pipe.get_list_of_docs()

    def remove_stop_words(self):
        """
        Runs the remove_stop_words method on each of the piplines
        """
        
        self.train_pipe.remove_stop_words()
        self.test_pipe.remove_stop_words()
    
    def condense(self):
        """
        Runs the word_condenser method on each of the pipelines
        """
        
        self.train_pipe.word_condenser()
        self.test_pipe.word_condenser()
    
    def tfidf(self, stem=False, lemmatize=False, max_features=None, 
              ngram_range=(1,1)):
        """
        Calls the tf_vect method on each of the piplelines. It uses the original
        string with stop words removed.  If stem == True then it uses a stemmed
        version of the string and if lemmatize == True then it uses a lemmatized
        version fo the string.

        Params
        stem: bool default is false
        lemmatize: bool default is false
        """
        
        if not stem and not lemmatize:
            self.train_pipe.tf_vect(self.train_pipe.stops_removed_str,
                                    max_features=max_features, 
                                    ngram_range=ngram_range)
            self.test_pipe.tf_vect(self.test_pipe.stops_removed_str,
                                   max_features=max_features, 
                                   ngram_range=ngram_range)
            
        elif self.stem:
            self.condense()
            self.train_pipe.tf_vect(self.train_pipe.porter_str,
                                    max_features=max_features, 
                                    ngram_range=ngram_range)
            self.test_pipe.tf_vect(self.test_pipe.porter_str,
                                   max_features=max_features, 
                                   ngram_range=ngram_range)
            
        elif self.lemmatize:
            self.condense()
            self.train_pipe.tf_vect(self.train_pipe.wordnet_str,
                                    max_features=max_features, 
                                    ngram_range=ngram_range)
            self.test_pipe.tf_vect(self.test_pipe.wordnet_str,
                                   max_features=max_features, 
                                   ngram_range=ngram_range)
        else:
            print('Cannot lemmatize and stem')
        
    def vectorizer(self, test=False, ngram_range=(1,1)):
        self.train_pipe.count_vectorizer(ngram_range=ngram_range)
        
        if test:
            self.test_pipe.count_vectorizer(ngram_range=ngram_range)

    def naive_bayes_model(self):
        """
        Sets up a naive bayes model for the documents in the directory

        Params
        directory: directory for the documents
        stop_words: list of stop_words for word filtration
        technique: technique: str choose from ['porter', 'snowball','wordnet']

        Returns
        nb_model: a naive bayes model for the documents in the directory
        cv: CountVectorizer object for the documents
        """
        self.nb_model = MultinomialNB()
        self.nb_model.fit(self.train_pipe.tfidf, self.train_pipe.target_lst)
        

    def return_top_n_words(self,  n=7):
        """
        Prints out the top n words for each document in the categories for the 
        documents in the directory

        Params
        directory: directory for the documents
        stop_words: list of stop_words for word filtration
        documents: a list of the categories (folders) in the directory
        technique: technique: str choose from ['porter', 'snowball','wordnet']

        """
        feature_words = self.train_pipe.cv.get_feature_names()
        categories = self.nb_model.classes_
        self.top_words_dic = {}
        for cat in range(len(categories)):
            print(f"\n Target: {cat}, name: {categories[cat]}")
            log_prob = self.nb_model.feature_log_prob_[cat]
            i_topn = np.argsort(log_prob)[::-1][:n]
            features_topn = [feature_words[i] for i in i_topn]
            self.top_words_dic[categories[cat]] = features_topn
            print(f"Top {n} tokens: ", features_topn)
            
        

    def get_accuracy_classification_report(self, ngram_range=(1,1)):
        """
        Prints out and returns the accuracy score from the prediction vs the actuals
        for the test set

        Params
        train_docs: list of strs used to train on
        test_docs: list of strs used to test
        test_targes: list of strs for the test target values

        Returns
        Accuracy score for the model
        """
        self.nb_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=ngram_range)),
                            ('tfidf', TfidfTransformer()),
                            ('model', MultinomialNB()),
                            ])
        
        if not self.stem and not self.lemmatize:
            X_train = self.train_pipe.stops_removed_str
            X_test = self.test_pipe.stops_removed_str
        
        elif self.lemmatize:
            self.condense()
            X_train = self.train_pipe.wordnet_str
            X_test = self.test_pipe.wordnet_str
        
        elif self.stem:
            self.condense()
            X_train = self.train_pipe.porter_str
            X_test = self.test_pipe.porter_str
        
        else:
            print('Cannot lemmatize and stem')
        
        
        self.nb_pipeline.fit(X_train, self.train_pipe.target_lst)
        print(len(X_test))
        self.predicted = self.nb_pipeline.predict(X_test)
        print(self.predicted.shape)
        print(len(self.test_pipe.target_lst))
        self.accuracy = np.mean(self.predicted == self.test_pipe.target_lst)
        print("\nThe accuracy on the test set is {0:0.3f}.".format(self.accuracy))

        self.class_report = classification_report(self.test_pipe.target_lst, 
                                                  self.predicted,
                                                  digits=3, output_dict=True)
    
    def pickle_model(self):
        """
        Pickles the model
        """

        with open('nb_pickle.pkl', 'wb') as f:
            pickle.dump(self.nb_pipeline, f)

    def confusion_matrix_plot(self, ax):
        """
        Generates a confusion matrix

        Params
        ax: axes to be used for the plot
        """
        if not self.stem and not self.lemmatize:
            X_test = self.test_pipe.stops_removed_str
        
        elif self.lemmatize:
            X_test = self.test_pipe.wordnet_str
        
        elif self.stem:
            X_test = self.test_pipe.porter_str
        
        else:
            print('Cannot lemmatize and stem')
        
        plot_confusion_matrix(self.nb_pipeline, X_test, self.test_pipe.target_lst, 
                              xticks_rotation='vertical',
                              cmap=plt.cm.Blues, ax=ax)


if __name__ == "__main__":

    train_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TrainTest/TrainDocs/'
    test_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TestDocs/'
    
    start = time.time()

    stop_words = stopwords.words('english')
    extra_stops = ['city', 'contractor', 'contract', 'wbe', 'chicago', 'must',
                   'number', 'may']
    stop_words.extend(extra_stops)

    nb = NaiveBayes(train_dir, test_dir, stop_words)
    nb.convert_txt_to_lists()
    nb.remove_stop_words()
    # nb.tfidf(ngram_range=(4,4))
    # nb.vectorizer(ngram_range=(4,4))
    # nb.naive_bayes_model()
    # nb.return_top_n_words()
    nb.get_accuracy_classification_report(ngram_range=(4,4))
    nb.pickle_model()
    # fig, ax = plt.subplots(figsize=(12,12))
    # plt.rcParams.update({'font.size': 20})
    # nb.confusion_matrix_plot(ax)
    # ax.set_xlabel('Predicted Label', fontsize=20, weight='bold')
    # ax.set_ylabel('True Label', fontsize=20, weight='bold')
    # ax.set_title('Confusion Matrix', fontsize=30, weight='bold')
    # plt.xticks(fontsize=16, rotation=70, weight='bold')
    # plt.yticks(fontsize=16, weight='bold')
    # plt.tight_layout()
    # plt.show()

    # ngrams = []
    # max_grams = 7
    # y = []
    # x = np.arange(1, max_grams+1, step=1)
    
    # for i in range(1, max_grams+1):
    #     ngrams.append((i,i))
    
    # for ngram in ngrams:
    #     print(f'Generating nb model with {ngram} range')
    #     nb.get_accuracy_classification_report(ngram_range=ngram)     
    #     y.append(nb.accuracy)
    
    d = nb.class_report
    df = pd.DataFrame(d)
    print(df.to_markdown())
    end = time.time()
    print(f'This took {end-start:.2f}')





    
