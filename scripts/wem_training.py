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
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.corpus import stopwords # for eliminating stop words
stopenglish = list(stopwords.words("english")) # assign list of english stopwords
import string # for one method of eliminating punctuation
punctuations = list(string.punctuation) # assign list of common punctuation symbols
punctuations+=['*','•','©','–','–','``','’','“','”','...','»',"''",'..._...','--','×','|_','_','§','…','⎫'] # Add a few more punctuations also common in web text
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
charters_path = "../../nowdata/traincf_2015_250_v2a_unlappedtext_counts3.tar.gz" # All text data; only charter schools (regardless if open or not)
wordsent_path = "../data/wem_wordsent_data_train250_nostem_unlapped.pkl"
phrasesent_path = "../data/wem_phrasesent_data_train250_nostem_unlapped.pkl"
#wemdata_path = "../data/wem_data.pkl"
model_path = "../data/wem_model_train250_nostem_unlapped.txt"

# Check if sentences data already exists, to save time
try:
    if (os.path.exists(wordsent_path)) and (os.path.getsize(wordsent_path) > 10240): # Check if file not empty (at least 10K)
        print("Existing sentence data detected at " + str(os.path.abspath(wordsent_path)) + ", skipping preprocessing sentences.")
        sented = True
    else:
        sented = False
except FileNotFoundError or OSError: # Handle common errors when calling os.path.getsize() on non-existent files
    sented = False

# Check if sentence phrases data already exists, to save time
try:
    if (os.path.exists(phrasesent_path)) and (os.path.getsize(phrasesent_path) > 10240): # Check if file not empty (at least 10K)
        print("Existing sentence + phrase data detected at " + str(os.path.abspath(phrasesent_path)) + ", skipping preprocessing sentence phrases.")
        phrased = True
    else:
        phrased = False
except FileNotFoundError or OSError: # Handle common errors when calling os.path.getsize() on non-existent files
    phrased = False

# Load charter data into DF
gc.disable() # disable garbage collector
df = pd.read_pickle(charters_path, compression="gzip")
gc.enable() # enable garbage collector again


# ## Define helper functions

def quickpickle_load(picklepath):
    '''Very time-efficient way to load pickle-formatted objects into Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Filepath to pickled (*.pkl) object.
    Output: Python object (probably a list of sentences or something similar).'''

    with open(picklepath, 'rb') as loadfile:
        
        gc.disable() # disable garbage collector
        outputvar = cPickle.load(loadfile) # Load from picklepath into outputvar
        gc.enable() # enable garbage collector again
    
    return outputvar


def quickpickle_dump(dumpvar, picklepath):
    '''Very time-efficient way to dump pickle-formatted objects from Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Python object (probably a list of sentences or something similar).
    Output: Filepath to pickled (*.pkl) object.'''

    with open(picklepath, 'wb') as destfile:
        
        gc.disable() # disable garbage collector
        cPickle.dump(dumpvar, destfile) # Dump dumpvar to picklepath
        gc.enable() # enable garbage collector again
    
    
def load_list(file_path):
    """Loads list into memory. Must be assigned to object."""
    
    textlist = []
    with open(file_path) as file_handler:
        line = file_handler.readline()
        while line:
            textlist.append(line)
            line = file_handler.readline()
    return textlist


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
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for tup in tuplist: # Iterate over tuples in tuplist (list of tuples)
        if tup[3] in known_pages or tup=='': # Could use hashing to speed up comparison: hashlib.sha224(tup[3].encode()).hexdigest()
            continue # Skip this page if exactly the same as a previous page on this school's website

        for chunk in tup[3].split('\n'): #.split('\x').replace('\xa0','').replace('\x92',''):
            for sent in sent_tokenize(chunk): # Clean up words: lower-case; remove unicode spaces ('\xa0'),tabs ('\t'), end-dashes, and any other leftovers ('\u*', '\x*', '\b*')
                words_by_sentence.append(list(re.sub(r"\\x.*|\\u.*|\\b.*|-$|^-|'$|^'|[*+]", "", 
                                                     word.lower().replace(u"\xa0", u" ").replace(u"\x00", u"").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")) 
                                         for word in word_tokenize(sent) 
                                         if not (word in punctuations 
                                                 or "http" in word
                                                 or "www" in word
                                                 or "\\" in word
                                                 or word.isdigit()
                                                 or word=="s"
                                                 or word=="m"
                                                 or word=="t"
                                                 or word=="re"
                                                 or word=="'s"
                                                 or word=="ve"
                                                 or word=="d"
                                                 or word=="ll"
                                                 or word.replace('-','').replace('.','').replace(',','').replace(':','').replace(';','').replace('/','').replace('k','').replace('e','').isdigit())))

        known_pages.add(tup[3])
    
    pcount += 1 # Add to counter
    
    return


if phrased: 
    pass # If parsed sentence phrase data exists, don't bother with tokenizing sentences

elif sented: # Check if tokenized sentence data already exists. If so, don't bother reparsing it
    words_by_sentence = []
    words_by_sentence = quickpickle_load(wordsent_path) # Load data back in for parsing phrases and word embeddings model

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
        
        
        # Save data for later
        try: # Use quickpickle to dump data into pickle file
            if __name__ == '__main__': 
                print("Saving list of tokenized sentences to file...")
                t = timeit.Timer(stmt="quickpickle_dump(words_by_sentence, wordsent_path)", globals=globals())
                print("Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

            '''with open(wordsent_path, 'wb') as destfile:
                gc.disable() # Disable garbage collector to increase speed
                cPickle.dump(words_by_sentence, destfile)
                gc.enable() # Enable garbage collector again'''

        except Exception as e:
            print(str(e), "\nTrying backup save option...")
            try:
                # Slower way to save data:
                with open(wordsent_path, 'wb') as destfile:
                    t = timeit.Timer(stmt="pickle.dump(words_by_sentence, destfile)", globals=globals())
                    print("Success! Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

            except Exception as e:
                print("Failed to save sentence data: " + str(e))

    except Exception as e:
        print("Failed to tokenize sentences: " + str(e))
        sys.exit()
        
    
# ## Preprocessing II: Detect and parse common phrases in words_by_sentence

if phrased: # Check if phrased data already exists. If so, don't bother recalculating it
    words_by_sentence = []
    words_by_sentence = quickpickle_load(phrasesent_path) # Load data back in, for word embeddings model

else:
    tqdm.pandas(desc="Parsing phrases") # Change title of tqdm instance

    try:
        print("Detecting and parsing phrases in list of sentences...")
        # Threshold represents a threshold for forming the phrases (higher means fewer phrases). A phrase of words a and b is accepted if (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold, where N is the total vocabulary size. By default this value is 10.0
        phrases = Phrases(words_by_sentence, min_count=3, delimiter=b'_', common_terms=stopenglish, threshold=10) # Detect phrases in sentences based on collocation counts
        words_by_sentence = [phrases[sent] for sent in tqdm(words_by_sentence, desc="Parsing phrases")] # Apply phrase detection model to each sentence in data

    except Exception as e:
        print("Failed to parse sentence phrases: " + str(e))
        sys.exit()
    
    # Save data for later
    try: # Use quickpickle to dump data into pickle file
        if __name__ == '__main__': 
            print("Saving list of tokenized, phrased sentences to file...")
            t = timeit.Timer(stmt="quickpickle_dump(words_by_sentence, phrasesent_path)", globals=globals())
            print("Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')
                                         
        '''with open(phrasesent_path, 'wb') as destfile:
            gc.disable() # Disable garbage collector to increase speed
            cPickle.dump(words_by_sentence, destfile)
            gc.enable() # Enable garbage collector again'''

    except Exception as e:
        print(str(e), "\nTrying backup save option...")
        try:
            # Slower way to save data:
            with open(phrasesent_path, 'wb') as destfile:
                t = timeit.Timer(stmt="pickle.dump(words_by_sentence, destfile)", globals=globals())
                print("Success! Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

        except Exception as e:
            print("Failed to save parsed sentence phrases: " + str(e))
        

# Take a look at the data 
print("Sample of the first 150 sentences:")
print(words_by_sentence[:150])


# ## Train Word Embeddings Model with word2vec in gensim

''' 
Word2Vec parameter choices explained:
- size = 700: Use hundreds of dimensions/degrees of freedom to generate accurate models from this large data set
- window = 8: Observe window of 6 context words in each direction, keeping word-word relationships moderately tight
- min_count = 3: Exclude very rare words, which occur just once or twice and typically are irrelevant proper nouns
- sg = 1: I choose a 'Skip-Gram' model over a CBOW (Continuous Bag of Words) model because skip-gram works better with larger data sets. It predicts words from contexts, rather than smoothing over context information by counting each context as a single observation
- alpha = 0.025: Initial learning rate: prevents model from over-correcting, enables finer tuning
- min_alpha = 0.001: Learning rate linearly decreases to this value over time, so learning happens more strongly at first
- iter = 10: Five passes/iterations over the dataset
- batch_words = 20000: During each pass, sample batch size of 10000 words
- workers = 1: Set to 1 to guarantee reproducibility, OR accelerate by parallelizing model training across the 44 vCPUs of the XXL Jetstream VM
- seed = 43: To increase reproducibility of model training 
- negative = 5: Draw 5 "noise words" in negative sampling in order to simplify weight tweaking
- ns_exponent = 0.75: Shape negative sampling distribution using 3/4 power, which outperforms other exponents (as popularized by original word2vec paper, Mikolov et al 2013) and slightly weights against high-frequency words (1 is exact frequencies, 0 is all words equally)
'''

# Train the model with above parameters:
try:
    print("Training word2vec model...")
    model = gensim.models.Word2Vec(words_by_sentence, size=700, window=8, min_count=3, sg=1, alpha=0.025, min_alpha=0.001,\
                                   iter=10, batch_words=20000, workers=1, seed=43, negative=5, ns_exponent=0.75)
    print("word2vec model TRAINED successfully!")

    # Save model for later:
    with open(model_path, 'wb') as destfile:
        try:
            model.wv.save_word2vec_format(destfile)
            print("word2vec model SAVED to " + str(model_path))
        except Exception as e:
            print(str(e))
            try:
                model.save(destfile)
                print("word2vec model SAVED to " + str(model_path))
            except Exception as e:
                print(str(e))

except Exception as e:
    print(str(e))

sys.exit() # Kill script when done, just to be safe