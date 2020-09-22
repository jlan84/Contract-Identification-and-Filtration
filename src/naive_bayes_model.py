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

    def __init__(self, train_targets, tfidf_matrix, count_vec_object):
        """
        Params
        tfidf_matrix: matrix with tfidf performed on it
        cv: count vectorizer object
        """
        self.train_targets = train_targets
        self.tfidf_matrix = tfidf_matrix
        self.cv = count_vec_object
        


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
        self.nb_model.fit(self.tfidf_matrix, self.train_targets)
        

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
        feature_words = self.cv.get_feature_names()
        categories = self.nb_model.classes_
        self.top_words_dic = {}
        for cat in range(len(categories)):
            print(f"\n Target: {cat}, name: {categories[cat]}")
            log_prob = self.nb_model.feature_log_prob_[cat]
            i_topn = np.argsort(log_prob)[::-1][:n]
            features_topn = [feature_words[i] for i in i_topn]
            self.top_words_dic[categories[cat]] = features_topn
            print(f"Top {n} tokens: ", features_topn)
            
        

    def get_accuracy_classification_report(self, train_docs, test_docs, 
                                           test_targets, ngram_range=(1,1)):
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

        self.nb_pipeline.fit(train_docs, self.train_targets)
        self.predicted = self.nb_pipeline.predict(test_docs)
        self.accuracy = np.mean(self.predicted == test_targets)
        print("\nThe accuracy on the test set is {0:0.3f}.".format(self.accuracy))
        self.class_report = classification_report(test_targets, self.predicted, digits=3,output_dict=True)
    
    def pickle_model(self):
        with open('nb_pickle.pkl', 'wb') as f:
            pickle.dump(self.nb_pipeline, f)

    def confusion_matrix_plot(self, test_docs, test_targets, ax):
        """
        Generates a confusion matrix

        Params
        test_docs: list of strings from the test set
        test_targets: list test target strings associated with the test_docs
        ax: axes to be used for the plot
        """
        plot_confusion_matrix(self.nb_pipeline, test_docs, test_targets, xticks_rotation='vertical',
                              cmap=plt.cm.Blues, ax=ax)




if __name__ == "__main__":

    train_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TrainTest/TrainDocs/'
    test_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TestDocs/'
    holdout_dir = ''
    start = time.time()

    stop_words = stopwords.words('english')
    extra_stops = ['city', 'contractor', 'contract', 'wbe', 'chicago', 'must',
                   'number', 'may']
    stop_words.extend(extra_stops)

    train_pipe = ContractPipeline(train_dir, stop_words)
    test_pipe = ContractPipeline(test_dir, stop_words)
    
    train_pipe.get_list_of_txts()
    test_pipe.get_list_of_txts()

    train_pipe.remove_stop_words()
    test_pipe.remove_stop_words()

    train_pipe.ngram_tf_vect(train_pipe.stops_removed_str, max_features=2000, ngram_range=(3,3))

    count_matrix, tfidf_matrix, cv = (train_pipe.tf_idf_matrix(train_pipe.
                                                               stops_removed_str,
                                                               max_features=2000))
    
    nb = NaiveBayes(train_pipe.target_lst, train_pipe.tfidf, cv)
    nb.naive_bayes_model()
    nb.return_top_n_words(n=4)

    nb.get_accuracy_classification_report(train_pipe.stops_removed_str,
                                          test_pipe.stops_removed_str,
                                          test_pipe.target_lst, ngram_range=(4,4))
    
    
    d = {'True Label': test_pipe.target_lst, 
        'Predicted Label': nb.nb_pipeline.predict(test_pipe.stops_removed_str)}
    
    df = pd.DataFrame(d)
    df.to_csv('../data/optimized_predictions.csv')




    # ngrams = []
    # max_grams = 7
    # y = []
    # x = np.arange(1, max_grams+1, step=1)
    
    # for i in range(1, max_grams+1):
    #     ngrams.append((i,i))
    
    # for ngram in ngrams:
    #     print(f'Generating nb model with {ngram} range')
    #     nb.get_accuracy_classification_report(train_pipe.stops_removed_str,
    #                                       test_pipe.stops_removed_str,
    #                                       test_pipe.target_lst, ngram_range=ngram)     
    #     y.append(nb.accuracy)
    



    # fig, ax = plt.subplots(figsize=(12,12))
    # plt.rcParams.update({'font.size': 20})
    # nb.confusion_matrix_plot(test_pipe.stops_removed_str, test_pipe.target_lst, ax)
    # ax.set_xlabel('Predicted Label', fontsize=20, weight='bold')
    # ax.set_ylabel('True Label', fontsize=20, weight='bold')
    # ax.set_title('Confusion Matrix', fontsize=30, weight='bold')
    # plt.xticks(fontsize=16, rotation=70, weight='bold')
    # plt.yticks(fontsize=16, weight='bold')
    # plt.tight_layout()
    # plt.show()
    
    print(nb.class_report)
    end = time.time()
    print(f'This took {end-start:.2f}')





    
