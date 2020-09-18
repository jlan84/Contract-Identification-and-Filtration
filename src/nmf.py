from sklearn.decomposition import NMF
import numpy as np
from pipeline import ContractPipeline
from directoryassistor import DirectoryAssistor
from nltk.corpus import stopwords
import time
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('fivethirtyeight')

class NMFModel():

    def __init__(self, directory, stop_words):
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

        self.directory = directory
        self.stop_words = stop_words
        self.pipe = ContractPipeline(self.directory, self.stop_words)
        self.model = None

    def convert_txt_to_lists(self):
        """
        If the documents are in .txt format use this to call the 
        get_list_of_txts on each of the directorys
        """
        
        self.pipe.get_list_of_txts()
    
    def convert_doc_to_lists(self):
        """
        If the documents are in .doc format use this to call the 
        get_list_of_txts on each of the directorys
        """
        
        self.pipe.get_list_of_docs()

    def remove_stop_words(self):
        """
        Runs the remove_stop_words method on each of the piplines
        """
        
        self.pipe.remove_stop_words()
        
    def condense(self):
        """
        Runs the word_condenser method on each of the pipelines
        """
        
        self.pipe.word_condenser()
        
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
            self.pipe.tf_vect(self.pipe.stops_removed_str,
                                    max_features=max_features)
        
            
        elif stem:
            self.condense()
            self.pipe.tf_vect(self.pipe.porter_str,
                                    max_features=max_features)
            
        elif lemmatize:
            self.condense()
            self.pipe.tf_vect(self.pipe.wordnet_str,
                                    max_features=max_features)
        
        else:
            print('Cannot lemmatize and stem')



    def generate_model(self, n_components=None, alpha=0.1, l1_ratio=0.5):
        
        X = self.pipe.tfidf
        self.model = NMF(n_components=n_components, random_state=123,
                         alpha=0.1, l1_ratio=0.5)
        print('Fitting model')
        self.W = self.model.fit_transform(X)
        self.H = self.model.components_
    
    def error_optimization(self, n_components_lst, ax, alpha=0.1, l1_ratio=0.5):
        
        X = self.pipe.tfidf
        error_lst = []
        for components in n_components_lst:
            model = NMF(n_components=components, random_state=123, alpha=alpha,
                        l1_ratio=l1_ratio)
            model.fit_transform(X)
            error_lst.append(model.reconstruction_err_)
        ax.plot(n_components_lst, error_lst)
        ax.set_title('NMF Reconstruction Error vs Num Components', fontsize=20)
        ax.set_ylabel('Reconstruction Error', fontsize=16)
        ax.set_xlabel('Num Components', fontsize=16)
        plt.tight_layout()
        plt.show()

    def print_top_words(self, n_top_words=20):
        for topic_idx, topic in enumerate(self.model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([self.pipe.vect.get_feature_names()[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

if __name__ == '__main__':

    directory = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
EC2Data/txtFiles/'

    start_time = time.time()
    k = 18
    stop_words = stopwords.words('english')
    nmf = NMFModel(directory, stop_words=stop_words)
    nmf.convert_txt_to_lists()
    nmf.remove_stop_words()
    nmf.tfidf(max_features=3000)
    nmf.generate_model(n_components=k)
    # nmf.print_top_words()
    # fig, ax = plt.subplots(figsize=(10,8))
    # n_components_lst = [4, 8, 10, 12, 14, 16, 18]
    # nmf.error_optimization(n_components_lst, ax)
    
    # print(nmf.model.reconstruction_err_)
    
    # topics = ['Latent_Topic_{}'.format(i) for i in range(18)]
    # contracts = ['Arch_Engineering', 'Commodities', 'Comptroller', 'Construction',
    #              'Delegate_Agency', 'Prof_Services']
    # df = pd.DataFrame(nmf.W, index=contracts, columns=topics)
    # df.to_csv('../data/')
    
    print(nmf.W[:10])
    
    end_time = time.time()
    print(f'This took {end_time-start_time:.2f} seconds')
    