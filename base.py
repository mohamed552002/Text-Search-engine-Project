import os
import pandas as pd
from tabulate import tabulate
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math

def word_indexer(word, lst):  # function to get word index in file
    index_dict = {word: []}
    for i in range(0, len(lst)):
        if lst[i] == word:
            index_dict[word].append(i)
    return (index_dict)


def text_preprocessing(text):  # Casefolding and Tokenization
    text.lower()  # Case Folding with lower function
    ##################################################
    # Tokenization step
    token_lst = (word_tokenize(text))
    return token_lst


def stopword_remove(token_lst):  # stopword removal
    stp = list(stopwords.words("english"))
    stp += ",", "?", "'", "\""  # Adding those puctuations to stopword list
    stp.remove("in")
    stp.remove("to")
    filtered_text = [t for t in token_lst if not t in stp and t.isalpha()]  # isalpha here to add only strings
    return filtered_text


def stemming(filtered_txt):  # Stemming
    stemmized_words = []
    stemmer = PorterStemmer()
    for ft in filtered_txt:
        stemmized_words.append(stemmer.stem(ft))
    return stemmized_words


def text_processing(text):  # to do all the work
    token_lst = text_preprocessing(text)
    filtered_txt = stopword_remove(token_lst)
    stemmized_txt = stemming(filtered_txt)
    return stemmized_txt


def PIndexNoStemm(final_text):  # to get positional index
    positional_dict = {}
    for i in range(0, len(final_text)):
        positional_dict.update({final_text[i]: {}})
        for j in range(1, 11):
            file = open(f"file{j}.txt").read()
            final_file = (text_preprocessing(file))
            if (final_file.count(final_text[i])):
                positional_dict[f"{final_text[i]}"].update({f"file{j}": []})
                pos_index = (word_indexer(final_text[i], final_file))
                for index in pos_index[final_text[i]]:
                    positional_dict[final_text[i]][f"file{j}"].append(index)
    return positional_dict

def positional_index(final_text):  # to get positional index
    positional_dict = {}
    for i in range(0, len(final_text)):
        positional_dict.update({final_text[i]: {}})
        for j in range(1, 11):
            file = open(f"file{j}.txt").read()
            final_file = stemming(text_preprocessing(file))
            if (final_file.count(final_text[i])):
                positional_dict[f"{final_text[i]}"].update({f"file{j}": []})
                pos_index = (word_indexer(final_text[i], final_file))
                for index in pos_index[final_text[i]]:
                    positional_dict[final_text[i]][f"file{j}"].append(index)
    return positional_dict


def words_extractor():  # extract all the words in all files
    word_set = []
    for i in range(1, 11):
        file = open(f"file{i}.txt").read()
        file_processed = text_preprocessing(file)
        for token in file_processed:
            word_set.append(token)
    return (list(set(word_set)))


def ALLTF():  # to get term frequency for each file
    tf = {}
    all_words = sorted(words_extractor())
    for i in range(1, 11):
        file = open(f"file{i}.txt").read()
        tf.update({f"file{i}": {}})
        for j in all_words:
            tf_counter = file.count(j)
            tf[f"file{i}"].update({j: tf_counter})
    return tf

def weight():  # to get weight of tf
    w = {}
    all_words = sorted(words_extractor())
    for i in range(1, 11):
        file = open(f"file{i}.txt").read()
        w.update({f"file{i}": {}})
        for j in all_words:
            tf_counter = file.count(j)
            if tf_counter != 0:
                weight_counter = tf_counter*(1 + math.log10(tf_counter))
            else:
                weight_counter = 0
            w[f"file{i}"].update({j: weight_counter})
    return w

def DocumentFrequency():  # to get document frequency
    df = {}
    all_words = words_extractor()
    dic = PIndexNoStemm((all_words))
    for key, value in sorted(dic.items()):
        df.update({key: len(value)})
    return df

def IDF():  # to get inverse document frequency
    idf = DocumentFrequency()
    for key, value in sorted(idf.items()):
        newValue = 10 / value
        idf.update({key: math.log10(newValue)})
    return idf

def IDFXTF():
    idfXtf=ALLTF()
    for key,v in sorted(idfXtf.items()):
        for key2, value in sorted(IDF().items()):
            idfXtf[key][key2]*=value
            
    return idfXtf

def printAsTbl(raw):
    df =((pd.DataFrame(raw)).fillna(0))
    print(tabulate(df,headers="keys",tablefmt="fancy_grid"))
# raw_input=input("what you search for : ")
# final_text=text_processing(raw_input)
# those loops to get the positional index
# PIndexRes = positional_index(final_text)

# df = {'': DocumentFrequency().keys(), 'df': DocumentFrequency().values(), 'idf' : IDF().values()}


printAsTbl(IDFXTF())
# print("*"*40)

