import textract
from directoryassistor import DirectoryAssistor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class ContractSifter():

    def __init__(self, directory, stop_words):
        self.directory = directory
        self.ds = DirectoryAssistor()
        self.stop_words = stop_words
        self.porter_dic = {}
        self.snowball_dic = {}
        self.wordnet_dic = {}
        self.combined_dic = {}

    def create_dic(self):
        """
        Returns a dictionary with folder names as the keys and an empty lst 
        as values

        params

        folder_names: list of folder names in the directory

        Returns 
        Dictionary
        """
        lst = self.ds.create_content_list(self.directory)
        word_dic = {key: [] for key in lst}
        return word_dic
    
    def remove_stop_words(self, lst):
        return [w for w in lst if w not in self.stop_words]
    
    def add_words(self):
        """
        Adds words from the files in the directories that are associated with
        the keys in the self.word_dic

        Returns
        self.word_dic with a list of words with the following removed from each 
        file in the folder for that key:
            1. Stop words
            2. Punctuation
            3. Underscores
        """
        self.word_dic = self.create_dic()
        for key in self.word_dic.keys():
            lst = self.ds.create_content_list(self.directory+key)
            for file in lst:
                full_text = textract.process(self.directory+key+'/'+file)
                str_full_text = full_text.decode('utf-8')
                lower_full_text = str_full_text.lower()
                edited_text = re.sub(r'\W+', ' ', lower_full_text)
                edited_text = edited_text.replace("_","")
                tokens = word_tokenize(edited_text)
                stop_lst = self.remove_stop_words(tokens)
                self.word_dic[key].append(stop_lst)
    
    def combine_word_lists(self):
        """
        Combine all of the lists for a key into one list from the Pipeline
        word_dic attribute
        """
        for key in self.word_dic.keys():
            result = []
            for lst in self.word_dic[key]:
                result.extend(lst)
            self.combined_dic[key] = result
    
    def word_condenser(self):
        
        porter = PorterStemmer()
        snowball = SnowballStemmer('english')
        wordnet = WordNetLemmatizer()
        for key in self.combined_dic.keys():
            porter_lst = []
            snowball_lst = []
            wordnet_lst = []
            for word in self.combined_dic[key]:
                porter_lst.append(porter.stem(word))
                snowball_lst.append(snowball.stem(word))
                wordnet_lst.append(wordnet.lemmatize(word))
            self.porter_dic[key] = porter_lst
            self.snowball_dic[key] = snowball_lst
            self.wordnet_dic[key] = wordnet_lst
    
    def word_count(self, dic):
        """
        Returns the count of the words in each key of the dictionary

        Params

        dic = dict for which the words will be counted

        Returns

        new_dic: dict with word count for each key
        """
        word_count_dic = {}
        for key, val in dic.items():
            word_count_dic[key] = Counter(val)
        new_dic = dict(word_count_dic)
        return new_dic

    def word_cloud(self, dic):
        """
        Generates a word cloud for each key in the dic

        Params

        dic: dict for which the word cloud will be generated

        Returns

        Plot with word cloud for each key in dic
        """

        word_cloud_dic = {}
        for key, val in dic.items():
            word_cloud_dic[key] = ' '.join(val)
        wc_lst = []
        for val in word_cloud_dic.values():
            wc = WordCloud(width=1000, height=1000, background_color='white', 
                            min_font_size=9)
            wc_lst.append(wc.generate(val))
        fig, axs = plt.subplots(3,3, figsize=(15,12))
        titles = list(dic.keys())
        for cloud, title, ax in zip(wc_lst, titles, axs.flatten()):
            chartBox = ax.get_position()
            ax.set_position(pos=[chartBox.x0,chartBox.y0,chartBox.width*1.05,
                                        chartBox.height*1.05])
            ax.imshow(cloud)
            ax.set_title(title, fontsize=16, weight='bold')
            ax.axis("off")
        axs[2,1].set_axis_off()
        axs[2,2].set_axis_off()
        chartBox = axs[2,0].get_position()
        axs[2,0].set_position(pos=[chartBox.x0*2.8,chartBox.y0*.9,chartBox.width*1.05,
                                    chartBox.height*1.05])
        plt.show()

def convert_to_string(dic):
    """
    Converts the lists in the dic for each key to a string

    Params

    dic: dict to be converted with a list of words for values

    Returns

    string_dic with the values converted to string instead of list of words
    """
    string_dic = {}
    for key in dic.keys():
        for val in dic[key]:
            string = " ".join(val)
            string_dic[key].append(string)
    return string_dic

def convert_dic_to_list(dic):
    """
    Creates a list of all the values in dic

    Params
    dic: dict with a list of strings for keys

    Returns

    dic_list that is a list of all the values in dic
    """
    dic_list = []
    for key in dic.keys():
        for val in dic[key]:
            dic_list.append(val)
    return dic_list



if __name__ == "__main__" and execute:
    pass