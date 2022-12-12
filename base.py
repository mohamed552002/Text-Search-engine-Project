import os
import pandas as pd
from tabulate import tabulate
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import time

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
    stemmized = []
    stemmer = PorterStemmer()
    for ft in filtered_txt:
        stemmized.append(stemmer.stem(ft))
    return stemmized


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
            final_file = stopword_remove(final_file)
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

def documents_matched(stemmized_lst):
    docs_matched = []
    final = " ".join(stemmized_lst)
    # posindex=positional_index(final_text_lst)
    # x=len(final_text_lst)
    # if len(final_text_lst) ==1:
    #     docs_matched+=posindex[final_text_lst[0]].keys()
    # else:
    if(final != ""):
        for i in range(1,11):
            file = open((f"file{i}.txt")).read()
            file=text_preprocessing(file)
            file=stemming(file)
            file = " ".join(file)
            if file.count(final)>0 :
                docs_matched+=[f"file{i}"]
            
            # wordptr1=posindex[final_text_lst[i]]
            
            # if i+1  < len(final_text_lst):
            #     wordptr2=posindex[final_text_lst[i+1]]
            #     for key,value in (wordptr1.items()):
            #         if key in (wordptr2.keys()):
            #             docs_matched += [key for j in value if i+1 in wordptr2[key]]
    return (docs_matched)

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
            file = stopword_remove(file)
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
            file = stopword_remove(file)
            w.update({f"file{i}": {}})
            for j in all_words:
                tf_counter = file.count(j)
                if tf_counter != 0:
                    weight_counter = tf_counter*(1 + math.log10(tf_counter))
                else:
                    weight_counter = 0
                w[f"file{i}"].update({j: weight_counter})
    else:
        w=ALLTF(queryLst)
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
def product(query,docs):
    if len(docs):
        pdict={}
        psumdict={}
        NM_files=normalized()
        NM_query=normalized(query)
        for i in docs:
            pdict.update({f"{i}":{}})
            psumdict.update({f"{i}":{"sum":0}})
            for j in query:
                pdict[i].update({j:NM_files[i][j]*NM_query[j]})
                psumdict[i]["sum"]+=NM_files[i][j]*NM_query[j]
                # pdict[f"{i}"][j].append(NM_files[i][j]*NM_query[j])
    else:
        pdict={'':{'':0}}
        psumdict={'':{"sum":0}}
    return [pdict,psumdict]

def similarity(query,file):
    filterd= stopword_remove(query)
    stemmized = stemming(filterd)
    docs=documents_matched(stemmized)
    if len(docs) ==0:
        return 0
    prod = product(query,docs)[1]
    x=prod[file]["sum"]
    return x

def similarity_matched(query):
    filterd= stopword_remove(query)
    stemmized = stemming(filterd)
    docs=documents_matched(stemmized)
    docs_s=product(query,docs)[1]
    dic= sorted(docs_s, key=lambda x: (docs_s[x]['sum']), reverse=True)
    return tuple(dic)

def print_similarity(stemmized_query):
    docs = documents_matched(stemmized_query)
    for d in docs:
        print(f"cosine similarty(q,{d}) : {similarity(tokenized_query,f'{d}')}")

def printAsTbl(raw):
    df =((pd.DataFrame(raw)).fillna(0))
    print(tabulate(df,headers="keys",tablefmt="fancy_grid"))
    

def printAsQueryTbl(query,prod):
    query_dict={"word":ALLTF(query).keys(),"TF-raw":ALLTF(query).values(),"W-TF":weight(query).values(),
                "IDF":IDF(query).values(),"IDF*TF":IDFXTF(query).values(),"normalized":normalized(query).values()}
    df = (pd.DataFrame(query_dict)).set_index("word")
    prod_df = pd.DataFrame(prod)
    result = df.join(prod_df,how="inner")
    print(tabulate(result,headers="keys",tablefmt="fancy_grid"))

def printPosIndex(query):
    x=positional_index(stemming(query))
    stemm=PorterStemmer()
    for word in query:
        word_stem = stemm.stem(word)
        print(f"<{word}, number of docs containing {word} {len(x[f'{word_stem}'])};" )
        for key ,value in (x[word_stem]).items():
                print(f"{key}:{(x[word_stem][key])}")

df_idf = {"": DocumentFrequency().keys(), "df": DocumentFrequency().values(), "idf" : IDF().values()}
length={0:docLentgh().keys(),"  ":docLentgh().values()}
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
raw_query = input("what u search for")
tokenized_query =text_preprocessing(raw_query)
filtered=stopword_remove(tokenized_query)
stemmized_query= text_processing(raw_query)
docs=documents_matched(stemmized_query)
prod=product(filtered,docs)[0]
prod_sum=product(filtered,docs)[1]
printAsQueryTbl(filtered,prod)
productsum_df=pd.DataFrame(prod_sum)
print(tabulate(productsum_df,headers="keys",tablefmt="fancy_grid"))
print(f"Query Length : {docLentgh(tokenized_query)}")
print_similarity(prod_sum)
print(f"Rank docs matched {similarity_matched(tokenized_query)}")


printPosIndex(tokenized_query)
