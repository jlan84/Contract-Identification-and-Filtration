from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pipeline import ContractPipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from directoryassistor import DirectoryAssistor
from nltk.corpus import stopwords
import time
import pickle


class RandomForest():
    def __init__(self, train_dir, test_dir, holdout_dir, stop_words):
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
        self.holdout_dir = holdout_dir
        self.stop_words = stop_words
        self.train_pipe = ContractPipeline(self.train_dir, self.stop_words)
        self.test_pipe = ContractPipeline(self.test_dir, self.stop_words)
        self.holdout_pipe = ContractPipeline(self.holdout_dir, self.stop_words)
        self.model = None

    def convert_txt_to_lists(self):
        """
        If the documents are in .txt format use this to call the 
        get_list_of_txts on each of the directorys
        """
        
        self.train_pipe.get_list_of_txts()
        self.test_pipe.get_list_of_txts()
        self.holdout_pipe.get_list_of_txts()
    
    def convert_doc_to_lists(self):
        """
        If the documents are in .doc format use this to call the 
        get_list_of_txts on each of the directorys
        """
        
        self.train_pipe.get_list_of_docs()
        self.test_pipe.get_list_of_docs()
        self.holdout_pipe.get_list_of_docs()

    def remove_stop_words(self):
        """
        Runs the remove_stop_words method on each of the piplines
        """
        
        self.train_pipe.remove_stop_words()
        self.test_pipe.remove_stop_words()
        self.holdout_pipe.remove_stop_words()
    
    def condense(self):
        """
        Runs the word_condenser method on each of the pipelines
        """
        
        self.train_pipe.word_condenser()
        self.test_pipe.word_condenser()
        self.holdout_pipe.word_condenser()
    
    def tfidf(self, stem=False, lemmatize=False, max_features=None):
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
                                    max_features=max_features)
            self.test_pipe.tf_vect(self.test_pipe.stops_removed_str,
                                   max_features=max_features)
            self.holdout_pipe.tf_vect(self.holdout_pipe.stops_removed_str,
                                      max_features=max_features)
        elif stem:
            self.condense()
            self.train_pipe.tf_vect(self.train_pipe.porter_str,
                                    max_features=max_features)
            self.test_pipe.tf_vect(self.test_pipe.porter_str,
                                   max_features=max_features)
            self.holdout_pipe.tf_vect(self.holdout_pipe.porter_str,
                                      max_features=max_features)
        elif lemmatize:
            self.condense()
            self.train_pipe.tf_vect(self.train_pipe.wordnet_str,
                                    max_features=max_features)
            self.test_pipe.tf_vect(self.test_pipe.wordnet_str,
                                   max_features=max_features)
            self.holdout_pipe.tf_vect(self.holdout_pipe.wordnet_str,
                                      max_features=max_features)
        else:
            print('Cannot lemmatize and stem')
    
    def add_nb_predictions(self):
        
        print('Generating Predictions from NB Model')

        with open('nb_pickle.pkl', 'rb') as f:
            self.nb_model = pickle.load(f)
        
        self.X_train_probs = (
            self.nb_model.predict_proba(self.train_pipe.stops_removed_str)
            )
        self.X_test_probs = (
            self.nb_model.predict_proba(self.test_pipe.stops_removed_str)
            )
        self.X_holdout_probs = (
            self.nb_model.predict_proba(self.holdout_pipe.stops_removed_str)
            )
        self.train_pipe.tfidf = (
            np.append(self.train_pipe.tfidf.toarray(), self.X_train_probs, 1)
            )
        self.test_pipe.tfidf = (
            np.hstack((self.test_pipe.tfidf.toarray(), self.X_test_probs))
            )
        self.holdout_pipe.tfidf = (
            np.hstack((self.holdout_pipe.tfidf.toarray(), self.X_holdout_probs))
            )

    def create_random_forest_model(self, n_estimators=100, criterion='gini',
                                   max_depth=None, norm='l2', use_idf=True,
                                   smooth_idf=True, verbose=True,
                                   random_state=None):
        X_train = self.train_pipe.tfidf
        y_train = self.train_pipe.target_lst

        X_test = self.test_pipe.tfidf
        y_test = self.test_pipe.target_lst
        
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                    criterion=criterion, max_depth=max_depth,
                                    verbose=True, n_jobs=-1, 
                                    random_state=random_state)
        print('Fitting the model')
        self.model.fit(X_train, y_train)
        print('Predicting')
        self.prediction = self.model.predict(X_test)
        print('Calculating the model score')
        self.score = self.model.score(X_test, y_test)
                                        
    def grid_search(self):
        self.model = RandomForestClassifier(random_state=123)
        params = {'n_estimators': [100,200],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': np.arange(5,20,2),
                  'criterion': ['gini', 'entropy']}
        gs = GridSearchCV(estimator=self.model, param_grid=params, cv=5,
                          verbose=True, n_jobs=-1)
        X_train = self.train_pipe.tfidf
        y_train = self.train_pipe.target_lst

        X_test = self.test_pipe.tfidf
        y_test = self.test_pipe.target_lst

        print('Fitting the model')
        gs.fit(X_train, y_train)

        self.best_params = gs.best_params_
        
        




if __name__ == '__main__':
    
    train_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TrainTest/TrainDocs/'
    test_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TrainTest/TestDocs/'
    holdout_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TestDocs/'

    start_time = time.time()
    stop_words = stopwords.words('english')
    extra_stops = ['city', 'contractor', 'contract', 'wbe', 'chicago', 'must']
    stop_words.extend(extra_stops)

    rf = RandomForest(train_dir, test_dir, holdout_dir, stop_words)
    rf.convert_txt_to_lists()
    rf.remove_stop_words()
    rf.tfidf(max_features=2000)
    rf.add_nb_predictions()
    # rf.grid_search()
    # print(rf.best_params)
    rf.create_random_forest_model(n_estimators=200, max_depth=15, random_state=123)
    # # print(rf.prediction[:10])
    # # print(rf.test_pipe.target_lst[:10])
    print(rf.score)
    end_time = time.time()
    print(f'This took {end_time-start_time:.2f} seconds')
    




    
