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


def ALLTF(queryLst=words_extractor()):  # to get term frequency for each file
    tf = {}
    all_words = queryLst
    if queryLst==words_extractor():
        for i in range(1, 11):
            file = open(f"file{i}.txt").read()
            file=text_preprocessing(file)
            tf.update({f"file{i}": {}})
            for j in all_words:
                tf_counter = file.count(j)
                tf[f"file{i}"].update({j: tf_counter})
    else:
        for word in set(all_words):
            tf.update({word: queryLst.count(word)})

    return tf

def weight(queryLst=words_extractor()):  # to get weight of tf
    w = {}
    all_words = queryLst
    if queryLst==words_extractor():
        for i in range(1, 11):
            file = open(f"file{i}.txt").read()
            file = text_preprocessing(file)
            w.update({f"file{i}": {}})
            for j in all_words:
                tf_counter = file.count(j)
                if tf_counter != 0:
                    weight_counter = tf_counter*(1 + math.log10(tf_counter))
                else:
                    weight_counter = 0
                w[f"file{i}"].update({j: weight_counter})
    else:
        w=ALLTF(query)
        for key,value in w.items():
            w[key]*=(1 + math.log10(value))
    return w

def DocumentFrequency(queryLst=words_extractor()):  # to get document frequency
    df = {}
    all_words = queryLst
    dic = PIndexNoStemm((all_words))
    for key, value in sorted(dic.items()):
        df.update({key: len(value)})
    return df

def IDF(queryLst=words_extractor()):  # to get inverse document frequency
    idf = DocumentFrequency(queryLst)
    for key, value in sorted(idf.items()):
        if value: 
            newValue = 10 / value
            idf.update({key: math.log10(newValue)})
        else:
            idf.update({key: 0})
    return idf

def IDFXTF(queryLst=words_extractor()): #getting idf * tf
    idfXtf=ALLTF(queryLst)
    idf = IDF(queryLst)
    for key,v in sorted(idfXtf.items()):
        if queryLst==words_extractor():
            for key2, value in sorted(idf.items()):
                idfXtf[key][key2]*=value
        else:
            idfXtf[key]*=idf[key]
            
    return idfXtf
def docLentgh(queryLst=words_extractor()): #getting document length
    all_words = queryLst
    idfXtf=IDFXTF(all_words)
    len_dict={}
    if queryLst == words_extractor():
        for i in range(1,11):
            sum_pow=0
            for word in (all_words):
                sum_pow+=math.pow((idfXtf[f"file{i}"][word]),2)
            len_dict.update({f"lentgh.file{i}":math.sqrt(sum_pow)})
    else:
        sum_pow=0
        for word in set(all_words):
            sum_pow+=math.pow(idfXtf[word],2)
        len = math.sqrt(sum_pow)
        return len
    return len_dict

def normalized(queryLst=words_extractor()): # getting Normalized tf.idf
    all_words = queryLst
    normalized_dict=IDFXTF(queryLst)
    lentgh=docLentgh(queryLst)
    if all_words == words_extractor():
        for i in range(1,11):
            for word in (words_extractor()):
                normalized_dict[f"file{i}"][word]/=lentgh[f"lentgh.file{i}"]
    else:
        for word in set(all_words):
            if lentgh:
                normalized_dict[word]/=lentgh
            else:
                normalized_dict[word]=0
    return normalized_dict

def printAsTbl(raw):
    df =((pd.DataFrame(raw)).fillna(0))
    print(tabulate(df,headers="keys",tablefmt="fancy_grid"))
def printAsQuery(query):
    query_dict={"word":ALLTF(query).keys(),"TF-raw":ALLTF(query).values(),"W-TF":weight(query).values(),
                "IDF":IDF(query).values(),"IDF*TF":IDFXTF(query).values(),"normalized":normalized(query).values()}
    df = (pd.DataFrame(query_dict)).set_index("word")
    
    print(tabulate(df,headers="keys",tablefmt="fancy_grid"))
# raw_input=input("what you search for : ")
# final_text=text_processing(raw_input)
# those loops to get the positional index
# PIndexRes = positional_index(final_text)

# df_idf = {"": DocumentFrequency().keys(), "df": DocumentFrequency().values(), "idf" : IDF().values()}

# length={0:docLentgh().keys(),"  ":docLentgh().values()}
print(" "*40+"Term Frequency Table")
printAsTbl(ALLTF(words_extractor()))
print(" "*40+"Weight Term Frequency Table")
printAsTbl(weight())
print(" "*40+"DF , IDF")
printAsTbl(df_idf)
print(" "*40+"IDF * TF")
printAsTbl(IDFXTF())
print(" "*40+"Length")
print(tabulate(length,tablefmt="fancy_grid"))
print(" "*40+"Normalized Tf.idf")
printAsTbl(normalized())

print("*"*40)

query = input("what u search for")
query =text_preprocessing(query)
printAsQuery(query)
