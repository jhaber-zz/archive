#!/usr/bin/env python
# coding: utf-8

# Word Embedding Models: Preprocessing and Model Training
# Project title: Charter school identities 
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: July 20, 2018
# Date last edited: July 26, 2018


# ## Initialize Python

# Import general packages
import nltk # for natural language processing tools
import pandas as pd # for working with dataframes
#from pandas.core.groupby.groupby import PanelGroupBy # For debugging
import numpy as np # for working with numbers
import pickle # For working with .pkl files
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process

# Import packages for cleaning, tokenizing, and stemming text
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.corpus import stopwords # for eliminating stop words
stopenglish = list(stopwords.words("english")) # assign list of english stopwords
import string # for one method of eliminating punctuation
punctuations = list(string.punctuation) # assign list of common punctuation symbols
punctuations+=['•','©','–','–','``','’','“','”','...','»',"''",'..._...','--','×','|_','_','§','…','⎫'] # Add a few more punctuations also common in web text
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

# Import packages for multiprocessing
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function
import Cython # For parallelizing word2vec


# ## Read in data

# Define file paths
charters_path = "../../charters_full_2015.pkl" # All text data; only charter schools (regardless if open or not)
wordsent_path = "../data/wem_wordsent_data.pkl"
phrasesent_path = "../data/wem_phrasesent_data.pkl"
wemdata_path = "../data/wem_data.pkl"
model_path = "../data/wem_model.txt"

if os.path.exists(wordsent_path): # Check if phrase data already exists, to save time
    sented = True
else:
    sented = False

if os.path.exists(phrasesent_path): # Check if phrase data already exists, to save time
    phrased = True
    print("Existing file detected at " + str(os.path.abspath(phrasesent_path)) + ", preprocessing already done? Aborting.")
    sys.exit()
else:
    phrased = False

# Load charter data into DF
gc.disable() # disable garbage collector
df = pd.read_pickle(charters_path)
gc.enable() # enable garbage collector again


# ## Define helper functions

def quickpickle_load(picklepath, outputvar):
    '''Very time-efficient way to load pickle-formatted objects into Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Filepath to pickled (*.pkl) object.
    Output: Python object (probably a list of sentences or something similar).'''

    output = open(picklepath, 'rb')
    gc.disable() # disable garbage collector

    outputvar = cPickle.load(output)

    gc.enable() # enable garbage collector again
    output.close()
    
    return outputvar

def quickpickle_dump(dumpvar, picklepath):
    '''Very time-efficient way to dump pickle-formatted objects from Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Python object (probably a list of sentences or something similar).
    Output: Filepath to pickled (*.pkl) object.'''

    output = open(picklepath, 'rb')
    gc.disable() # disable garbage collector

    cPickle.dump(dumpvar)

    gc.enable() # enable garbage collector again
    output.close()


# ## Preprocessing I: Tokenize web text by sentences

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
    
    print('Parsing school #' + str(pcount)) # Print number of school being parsed

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
                                                 or word.replace(',','').isdigit()
                                                 or word.replace(':','').isdigit()
                                                 or word.replace(';','').isdigit()
                                                 or word.replace('/','').isdigit()
                                                 or word.replace('k','').isdigit()
                                                 or word=="'s")))

        known_pages.add(tup[3])
    
    pcount += 1 # Add to counter
    
    return


if sented or phrased: # Check if tokenized sentence data already exists. If so, don't bother reparsing it
    pass

else:
    
    words_by_sentence = [] # Initialize variable to hold list of lists of words (sentences)
    pcount=0 # Initialize preprocessing counter
    df["WEBTEXT"] = df["WEBTEXT"].astype(list) # Coerce these to lists in order to avoid type errors

    # Convert DF into list (of lists of tuples) and call preprocess_wem on element each using Pool():
    try:
        tqdm.pandas(desc="Tokenizing sentences") # To show progress, create & register new `tqdm` instance with `pandas`

        #weblist = df["WEBTEXT"].tolist() # Convert DF into list to pass to Pool()

        # Use multiprocessing.Pool(numcpus) to run preprocess_wem:
        #print("Preprocessing web text into list of sentences...")
        #if __name__ == '__main__':
        #    with Pool(numcpus) as p:
        #        p.map(preprocess_wem, weblist) 

        # Much slower option (no multiprocessing):
        df["WEBTEXT"].progress_apply(lambda tups: preprocess_wem(tups))

        tqdm.pandas(desc="Parsing phrases") # Change title of tqdm instance

    except Exception as e:
        print(str(e))

    try:
            # Save data for later
            with open(wordsent_path, 'wb') as destfile:
                gc.disable() # Disable garbage collector to increase speed
                cPickle.dump(tqdm(words_by_sentence, desc="Saving phrase data"), destfile)
                gc.enable() # Enable garbage collector again

        except Exception as e:
            print(str(e))
        
    
# ## Preprocessing II: Detect and parse common phrases in words_by_sentence

if phrased: # Check if phrased data already exists. If so, don't bother recalculating it
    pass

else:

    try:
        print("Detecting and parsing phrases in list of sentences...")
        # Threshold represents a threshold for forming the phrases (higher means fewer phrases). A phrase of words a and b is accepted if (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold, where N is the total vocabulary size. By default this value is 10.0
        phrases = Phrases(tqdm(words_by_sentence,desc="Detecting phrases"), min_count=3, delimiter=b'_', common_terms=stopenglish, threshold=5) # Detect phrases in sentences based on collocation counts
        words_by_sentence = [phrases[sent] for sent in tqdm(words_by_sentence, desc="Parsing phrases")] # Apply phrase detection model to each sentence in data

    except Exception as e:
        print(str(e))
        sys.exit()
    
    # Use quickpickle to dump data into pickle file
    try:
        if __name__ == '__main__':
            print("Saving list of tokenized, phrased sentences to file...")
            t = timeit.Timer(stmt="quickpickle_dump(tqdm(words_by_sentence, desc='Saving data'), wemdata_path)", globals=globals())
            print("Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

    except Exception as e:
        print(str(e), "\nTrying backup save option...")
        try:
            # Slower way to save data:
            with open(wemdata_path, 'wb') as destfile:
                t = timeit.Timer(stmt="pickle.dump(tqdm(words_by_sentence, desc='Saving data'), destfile)", globals=globals())
                print("Success! Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

        except Exception as e:
            print(str(e))
        

# Take a look at the data 
print("Sample of the first 200 sentences:")
print(words_by_sentence[:200])


# ## Train Word Embeddings Model with word2vec in gensim

''' 
Word2Vec parameter choices explained:
- size = 200: Use hundreds of dimensions/degrees of freedom to generate accurate models from this large data set
- window = 6: Observe window of 6 context words in each direction, keeping word-word relationships moderately tight
- min_count = 3: Exclude very rare words, which occur just once or twice and typically are irrelevant proper nouns
- sg = 1: I choose a 'Skip-Gram' model over a CBOW (Continuous Bag of Words) model because skip-gram works better with larger data sets. It predicts words from contexts, rather than smoothing over context information by counting each context as a single observation
- alpha = 0.025: Initial learning rate: prevents model from over-correcting, enables finer tuning
- min_alpha = 0.001: Learning rate linearly decreases to this value over time, so learning happens more strongly at first
- iter = 5: Five passes/iterations over the dataset
- batch_words = 10000: During each pass, sample batch size of 10000 words
- workers = 1: Set to 1 to guarantee reproducibility, OR accelerate by parallelizing model training across the 44 vCPUs of the XXL Jetstream VM
- seed = 43: To increase reproducibility of model training 
- negative = 5: Draw 5 "noise words" in negative sampling in order to simplify weight tweaking
- ns_exponent = 0.75: Shape negative sampling distribution using 3/4 power, which outperforms other exponents (as popularized by original word2vec paper, Mikolov et al 2013) and slightly weights against high-frequency words (1 is exact frequencies, 0 is all words equally)
'''

# Train the model with above parameters:
try:
    model = gensim.models.Word2Vec(tqdm(words_by_sentence, desc="Training word2vec  model"), size=200, window=6, min_count=3, sg=1, alpha=0.025, min_alpha=0.001,\
                                   iter=5, batch_words=10000, workers=1, seed=43, negative=5, ns_exponent=0.75)
    print("word2vec model TRAINED successfully!")

    # Save model for later:
    with open(wem_path, 'wb') as destfile:
        try:
            model.wv.save_word2vec_format(destfile)
            print("word2vec model SAVED to " + str(destfile))
        except Exception as e:
            print(str(e))
            try:
                model.save(destfile)
                print("word2vec model SAVED to " + str(wem_path))
            except Exception as e:
                print(str(e))

except Exception as e:
    print(str(e))

sys.exit() # Kill script when done, just to be safe