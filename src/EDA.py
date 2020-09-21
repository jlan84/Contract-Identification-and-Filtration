from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
from directoryassistor import DirectoryAssistor
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textract
from webscraper import WebScraper
import pandas as pd
from nltk.corpus import stopwords
from sifter import ContractSifter
from pipeline import ContractPipeline
import pandas as pd
import scipy.stats as stats
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

plt.style.use('tableau-colorblind10')

def make_sns_bar_plot(ax, cols, labels, title, color='blue', label=None):
    """
    Creates a seaborn bar plot

    Parameters:
    ax = axes for the plot
    df = dataframe for the data
    col_name = str name of the dataframe column for the heights
    labels = str name of the dataframe column for the x axis labels
    color = color for the bars in the graph
    label = label the graph

    Returns:
    Bar plot
    """
    tick_loc = np.arange(len(labels))
    xlabel = labels
    sns.barplot(tick_loc, cols, color=color, ax=ax, label=label)
    ax.set_xticks(ticks=tick_loc)
    ax.set_xticklabels([str(x) for x in xlabel], rotation= 45, fontsize=14, horizontalalignment='right')
    ax.set_title(title, fontsize=20)

def make_bar_plot(ax, cols, labels, title, color='blue', label=None):
    """
    Creates a matplotlib bar plot

    Parameters:
    ax = axes for the plot
    df = dataframe for the data
    col_name = str name of the dataframe column for the heights
    labels = str name of the dataframe column for the x axis labels
    color = color for the bars in the graph
    label = label the graph

    Returns:
    Bar plot
    """
    tick_loc = np.arange(len(labels))
    xlabel = labels
    ax.bar(tick_loc, cols, color=color, label=label)
    ax.set_xticks(ticks=tick_loc)
    ax.set_xticklabels([str(x) for x in xlabel], rotation= 45, fontsize=14)
    ax.set_title(title, fontsize=20)


def pad_dic_list(dic, pad):
    """
    Pads dictionary so that all columns have the same number of rows

    Params
    dic: dictionary to be padded
    pad: value to pad with

    Returns
    The padded dictionary
    """
    lmax = 0
    for lname in dic.keys():
        lmax = max(lmax, len(dic[lname]))
    for lname in dic.keys():
        ll = len(dic[lname])
        if ll < lmax:
            dic[lname] += [pad] * (lmax - ll)
    return dic

def create_contract_graph(directory, title):
    """
    Generates a bar graph with the labels as the folders in the directory and the
    bar heights as the number of documents in each folder

    Params
    directory: str where the folders are located
    title: str title for bar graph
    """
    ds = DirectoryAssistor()
    folder_lst = ds.create_content_list(directory)
    folder_dic = {}
    for folder in folder_lst:
        folder_dic[folder] = len(ds.create_content_list(directory+folder))
    sort_folder_dic= sorted(folder_dic.items(), key=lambda x: x[1], reverse=True)
    cols = []
    labels = []
    for i in sort_folder_dic:
        labels.append(i[0])
        cols.append(i[1])
    fig, ax = plt.subplots(figsize=(10,10))
    # cols = list(new_dic.values())
    # labels = list(new_dic.keys())
    make_sns_bar_plot(ax, cols, labels, title=title)

def word_counts(individual_bag):
    """
    Returns the mean, avg, max, and min of the unique word count for all of 
    the contract classes

    Params
    individual_bag: dictionary with the keys as the contract type and the values
    are a bag of words for all the contracts for that type
    """
    contract_dic = {}
    for key, val in individual_bag.items():
        contract_dic[key] = len(val) 
    unique_word_lst = list(contract_dic.values())
    word_mean = np.mean(unique_word_lst)
    word_std = np.std(unique_word_lst)
    max_words = np.max(unique_word_lst)
    min_words = np.min(unique_word_lst)
    
    return word_mean, word_std, max_words, min_words

def graph_word_dist(individual_bag):
    """
    Returns a distribution graph of the unique words for each of the classes

    Params
    individual_bag: dictionary with the keys as the contract type and the values
    are a bag of words for all the contracts for that type
    """

    mu, sig, max_words, min_words = word_counts(individual_bag)
    print(f'The average number of unique words per contract type is {mu} with a \
standard devaition of {sig}.  The max and min number of unique words were {max_words},\
and {min_words}')
    x_axis = np.linspace(-3*sig, 6*sig, 100)
    norm = stats.norm(mu, sig)
    
    fig, ax = plt.subplots()
    ax.plot(x_axis, norm.pdf(x_axis))
    ax.set_title('Distribution of Unique Words', fontsize=16)
    plt.tight_layout()
    plt.show()

def true_pos(row, col1, col2):
    if row[col1] == row[col2]:
        return 1
    else:
        return 0

def true_neg(row, col1, col2):
    if row[col1] != row[col2]:
        return 1
    else:
        return 0



execute = True
if __name__ == "__main__" and execute:

   df = pd.read_csv('../data/predictions.csv')
   df.drop(columns='Unnamed: 0', inplace=True)
   df['Correct'] = df.apply(lambda row: true_pos(row, 'True Label', 'Predicted Label'), axis=1)
   df['Incorrect'] = df.apply(lambda row: true_neg(row, 'True Label', 'Predicted Label'), axis=1)
   contract_group = df.groupby('True Label').sum().sort_values('Correct', ascending=True)
   
   contract_group.plot(kind='barh', stacked=True, figsize=(10,6), linewidth=1,
                       align='center', width=.5, alpha=.7)
   
   plt.legend(loc='upper left', fontsize=16)
   plt.tight_layout()
   plt.xlabel('Predicted Count', fontsize=30, weight='bold')
   plt.ylabel('Contract Class',fontsize=30, weight='bold')
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=20)
   plt.title('Predictions', fontsize=40, weight='bold')
   plt.show()

    x = np.arange(1,8, step=1)
    y = [0.862,0.894,0.894,0.901,0.901,.9,.898]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(x,y)
    ax.set_title('Optimization Plot', fontsize=40, weight='bold')
    plt.xlabel('# of Words', fontsize=30, weight='bold')
    plt.ylabel('Accuracy',fontsize=30, weight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()