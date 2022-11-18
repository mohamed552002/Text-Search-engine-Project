
import os
import nltk
from nltk.tokenize import sent_tokenize , word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.arlstem import ARLSTem 
from langdetect import detect 
from nltk.corpus import stopwords
file = open("file0.txt").read().lower() #Case Folding with lower function
#################################################
# /// automatic file creation
# for i in range(11): 
#   f = open(f"file{i}.txt","a")
##################################################

# Tokenization step
token =nltk.Text(word_tokenize(file))
##################################################
# removing stopwords
stp = set(stopwords.words("english"))
stp.remove("at")
stp.remove("to")
filtered_text = [t for t in token if not t in stp]
# ##################################################
#Stemming step
stemmized_words = [] 
stemmer = PorterStemmer()
for ft in filtered_text:
  stemmized_words.append(stemmer.stem(ft))
# ##################################################
# Term frequency in file
term_freq = stemmized_words.count("good")

