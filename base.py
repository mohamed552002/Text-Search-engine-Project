
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#function to get positional index
def positional_index(word , lst):
  index_dict ={word : []}
  for i in range(0,len(lst)):
    if lst[i] == word:
      index_dict[word].append(i)
  return(index_dict)

def text_processing(text):
  text.lower() # Case Folding with lower function
  ##################################################
  # Tokenization step
  token =nltk.Text(word_tokenize(text))
  ##################################################
  #stopword removal
  stp = list(stopwords.words("english"))
  stp+= ",","?","'","\"" # Adding those puctuations to stopword list
  stp.remove("in")
  stp.remove("to")
  filtered_text = [t for t in token if not t in stp and t.isalpha() ] # isalpha here to add only strings
  ####################################################
  #Stemming step
  stemmized_words = [] 
  stemmer = PorterStemmer()
  for ft in filtered_text:
    stemmized_words.append(stemmer.stem(ft))
  return stemmized_words
  ######################################################

#################################################
# /// automatic file creation
positional_dict={}
raw_input=input("what you search for : ")
final_text=text_processing(raw_input)
#those loops to get the positional index
for i in range(0,len(final_text)): 
  positional_dict.update({final_text[i]:[{}]})
  for j in range(1,11):
    file = open(f"file{j}.txt").read()
    final_file=text_processing(file)
    if(final_file.count(final_text[i])):
      positional_dict[f"{final_text[i]}"][0].update({f"file{j}":[]})
      pos_index=(positional_index(final_text[i],final_file))
      for index in pos_index[final_text[i]]:
        positional_dict[final_text[i]][0][f"file{j}"].append(index)
# x.append((f"{inputU[j]},file{i}:{pos_index[inputU[j]]}"))
print(positional_dict) #to print all positional indices