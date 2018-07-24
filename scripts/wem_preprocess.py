#!/usr/bin/env python
# coding: utf-8

# Word Embedding Models: Preprocessing
# Project title: Charter school identities 
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: July 20, 2018


# ## Initialize Python

# IMPORTING KEY PACKAGES
import nltk # for natural language processing tools
import pandas as pd # for working with dataframes
#from pandas.core.groupby.groupby import PanelGroupBy # For debugging
import numpy as np # for working with numbers
import pickle # For working with .pkl files
#from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks

# FOR CLEANING, TOKENIZING, AND STEMMING THE TEXT
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.corpus import stopwords # for eliminating stop words
stopenglish = list(stopwords.words("english")) # assign list of english stopwords
import string # for one method of eliminating punctuation
punctuations = list(string.punctuation) # assign list of common punctuation symbols
punctuations+=['•','©','–'] # Add a few more punctuations also common in web text
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible

# PREP FOR MULTIPROCESSING
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function


# ## Read in data

# Define file paths
charters_path = "../../charters_full_2015.pkl" # All text data; only charter schools (regardless if open or not)
wem_path = "../data/wem_data.pkl"

# Load charter data into DF
df = pd.read_pickle(charters_path)


# ## Preprocessing: Tokenize web text by sentences

#tqdm.pandas(desc="Preprocessing") # To show progress, create & register new `tqdm` instance with `pandas`

def preprocess_wem(tuplist): # inputs were formerly: (tuplist, start, limit)
    
    '''This function cleans and tokenizes sentences, removing punctuation and numbers and making words into lower-case stems.
    Inputs: list of four-element tuples, the last element of which holds the long string of text we care about;
        an integer limit (bypassed when set to -1) indicating the DF row index on which to stop the function (for testing purposes),
        and similarly, an integer start (bypassed when set to -1) indicating the DF row index on which to start the function (for testing purposes).
    This function loops over five nested levels, which from high to low are: row, tuple, chunk, sentence, word.
    Note: This approach maintains accurate semantic distances by keeping stopwords.'''
        
    global words_by_sentence # Grants access to variable holding a list of lists of words, where each list of words represents a sentence in its original order
    global pcount # Grants access to preprocessing counter
    
    # For testing purposes:
    #if limit!=-1 and pcount>int(limit):
    #    return
    #if start!=-1 and pcount<int(start):
    #    return

    known_pages = set() # Initialize list of known pages for a school

    if type(tuplist)==float:
        return # Can't iterate over floats, so exit
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for tup in tuplist: # Iterate over tuples in tuplist (list of tuples)
        if tup[3] in known_pages or tup=='': # Could use hashing to speed up comparison: hashlib.sha224(tup[3].encode()).hexdigest()
            continue # Skip this page if exactly the same as a previous page on this school's website

        for chunk in tup[3].split('\n'):
            for sent in sent_tokenize(chunk):
                words_by_sentence.append(list(stem(word.lower()) 
                                         for word in word_tokenize(sent) 
                                         if not (word in punctuations 
                                                 or word.isdigit() 
                                                 or word.replace('-','').isdigit() 
                                                 or word.replace('.','').isdigit()
                                                 or word=="'s")))

        known_pages.add(tup[3])
    
    pcount += 1 # Add to counter
    
    return


words_by_sentence = [] # Initialize variable to hold list of lists of words (sentences)
pcount=0 # Initialize preprocessing counter
df["WEBTEXT"] = df["WEBTEXT"].astype(list) # Coerce these to lists in order to avoid type errors

# Convert DF into list (of lists of tuples) and call preprocess_wem on element each using Pool():
try:
    weblist = df["WEBTEXT"].tolist() # Convert DF into list to pass to Pool()

    # Use multiprocessing.Pool(numcpus) to run preprocess_wem:
    print("Preprocessing web text into list of sentences...")
    if __name__ == '__main__':
        with Pool(numcpus) as p:
            p.map(preprocess_wem, weblist) 
    
    # Much slower option (no multiprocessing):
    #df["WEBTEXT"].apply(lambda tups: preprocess_wem(tups))

    with open(wem_path, 'wb') as destfile:
        pickle.dump(words_by_sentence, destfile)

except Exception as e:
    print(str(e))
        
sys.exit()
