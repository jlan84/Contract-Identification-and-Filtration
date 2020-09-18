import textract
from directoryassistor import DirectoryAssistor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import time
import pandas as pd


class ContractPipeline():
    
    def __init__ (self, directory, stop_words):
        """
        Instantiates a ContractPipeline Object

        Params
        directory: str for main directory where the folders for the documntes are stored
        stop_words: list of words that will be removed from the documents
        """
        self.directory = directory
        self.stop_words = stop_words
        self.ds = DirectoryAssistor()
        self.individual_bag_o_words = {}


    def get_list_of_docs(self):
        """
        Creates
        target_lst: list that has all of the types for each document in the
        corresponding index of doc_text_lst
        doc_text_lst: list of lowercased cleaned strings for the text in each document
        """
        print('Converting to txt lists')
        start_time = time.time()

        folder_lst = self.ds.create_content_list(self.directory)
        doc_lst = []
        self.target_lst = []
        self.doc_text_lst = []
        
        for i in range(len(folder_lst)):
            doc_lst.append(self.ds.create_content_list(self.directory+folder_lst[i]))
            self.individual_bag_o_words[folder_lst[i]] = []
            for j in range(len(doc_lst[i])):
                text = textract.process(self.directory+folder_lst[i]+'/'+doc_lst[i][j])
                # convert to str
                text = text.decode('utf-8')
                # lowercase all text
                text = text.lower()
                # remove all punctuation
                text = re.sub(r'\W+', ' ', text)
                # remove underscores
                text = text.replace("_","")
                self.doc_text_lst.append(text)
                self.target_lst.append(folder_lst[i])
                self.individual_bag_o_words[folder_lst[i]].append(text.split())
                lst = []
        for val in self.target_lst:
            lst.append(val.replace('_', ' '))
        self.target_lst = lst
        
        end_time = time.time()
        print(f'This took {end_time-start_time:.2f} seconds')
    
    def get_list_of_txts(self):
        """
        Creates
        target_lst: list that has all of the types for each document in the
        corresponding index of doc_text_lst
        doc_text_lst: list of lowercased cleaned strings for the text in each document
        """
        print('Converting to txt lists')
        start_time = time.time()

        folder_lst = self.ds.create_content_list(self.directory)
        doc_lst = []
        self.target_lst = []
        self.doc_text_lst = []
        
        for i in range(len(folder_lst)):
            doc_lst.append(self.ds.create_content_list(self.directory+folder_lst[i]))
            self.individual_bag_o_words[folder_lst[i]] = []
            for j in range(len(doc_lst[i])):
                # read in file as str
                try:
                    with open(self.directory+folder_lst[i]+'/'+doc_lst[i][j], 'r') as f:
                        text = f.read().replace('\n', '')
                except:
                    continue
                
                # lowercase all text
                text = text.lower()
                
                # remove all punctuation
                text = re.sub(r'\W+', ' ', text)
                
                # remove underscores
                text = text.replace("_","")
                
                self.doc_text_lst.append(text)
                self.target_lst.append(folder_lst[i])
                self.individual_bag_o_words[folder_lst[i]].append(text.split())
        lst = []
        for val in self.target_lst:
            lst.append(val.replace('_', ' '))
        self.target_lst = lst

        end_time = time.time()
        print(f'This took {end_time-start_time:.2f} seconds')
    
    def bag_o_words(self):
        print('Creating bag o words')
        start_time = time.time()

        for key in self.individual_bag_o_words.keys():
            lst = []
            for val in self.individual_bag_o_words[key]:
                for word in val:
                    lst.append(word)
            self.individual_bag_o_words[key] = Counter(lst)
        total_word_lst = []
        for i in self.doc_text_lst:
            lst = i.split()
            for j in lst:
                total_word_lst.append(j)
        self.total_bag_o_words = Counter(total_word_lst)
        
        end_time = time.time()
        print(f'This took {end_time-start_time:.2f} seconds')

    def join_list_of_strings(self, lst):
        """
        Joins the list into a string
        
        Params
        lst: list of words
        """
        return [" ".join(x) for x in lst]

    def remove_stop_words(self):
        """
        Returns a new list of strings with stop words removed

        stops_removed_str: list of strings with stop words removed
        stops_removed_lst: list of lists containing words with stops removed
        """
        print('Removing stop words')
        start_time = time.time()

        split_lst = [txt.split() for txt in self.doc_text_lst]
        self.stops_removed_lst = []

        for split in split_lst:
            stops = [w for w in split if w not in self.stop_words]
            stop_num = [w for w in stops if not (w.isdigit() 
                        or w[0] == '-' and w[1:].isdigit())]
            self.stops_removed_lst.append(stop_num)

        self.stops_removed_str = self.join_list_of_strings(self.stops_removed_lst)

        end_time = time.time()
        print(f'This took {end_time-start_time:.2f} seconds')

    def word_condenser(self):
        """
        Takes in a list of strings and lemmatizes or stems them depending on the
        technique chosen

        self.porter_str: list of strings with porter stem technique used
        self.snowball_str: list of strings with snowball stem technique used
        self.wordnet_str: list of strings with wordnet lemmatize technique used
        """
        print('Condensing')
        start_time = time.time()

        porter = PorterStemmer()
        snowball = SnowballStemmer('english')
        wordnet = WordNetLemmatizer()
        
        porter_lst= [[porter.stem(w) for w in words] for words in self.stops_removed_lst]
        snowball_lst= [[snowball.stem(w) for w in words] for words in self.stops_removed_lst]
        wordnet_lst= [[wordnet.lemmatize(w) for w in words] for words in self.stops_removed_lst]
        
        self.porter_str = self.join_list_of_strings(porter_lst)
        self.snowball_str = self.join_list_of_strings(snowball_lst)
        self.wordnet_str = self.join_list_of_strings(wordnet_lst)
        
        end_time = time.time()
        print(f'This took {end_time-start_time:.2f} seconds')

    def tf_idf_matrix(self, documents, max_features=None):
        """
        Sets up a word count matrix, a tfidf matrix, and a CountVectorizer for
        the documents in the directory

        Params
        documents: list of strings to be vectorized

        Returns
        count_matrix: matrix with word counts
        tfidf_matrix: a tfidf matrix of the documents
        cv: CountVectorizer object for the documents
        """ 
        print('Generating tfidf and count matrix')
        start_time = time.time()

        cv = CountVectorizer(max_features=max_features)
        count_matrix = cv.fit_transform(documents)
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
        
        end_time = time.time()
        print(f'This took {end_time-start_time:.2f} seconds')
        return count_matrix, tfidf_matrix, cv

    def tf_vect(self, documents, max_features=None):
        """
        Returns tf-idf matrix from documents
        
        Prams
        documents: list of strings
        """
        print('Generating tfidf')
        start_time = time.time()

        self.vect = TfidfVectorizer(max_features=max_features)
        self.tfidf = self.vect.fit_transform(documents)
        
        end_time = time.time()
        print(f'This took {end_time-start_time:.2f} seconds')


if __name__ == "__main__":

    train_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TrainDocs/'
    directory = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/TestDocs/Commodities/'
    
    
    start = time.time()
    stop_words = stopwords.words('english')
    train_pipe = ContractPipeline(train_dir, stop_words)
    train_pipe.get_list_of_txts()
    train_pipe.remove_stop_words()
    # count_matrix, tfidf_matrix, cv = train_pipe.tf_idf_matrix(train_pipe.stops_removed_str, max_features=20)
    train_pipe.tf_vect(train_pipe.stops_removed_str, max_features=20)
    print(train_pipe.target_lst[:10])
    
    end = time.time()
    print(f'Time to run: {end - start:.2f}') 
    
    