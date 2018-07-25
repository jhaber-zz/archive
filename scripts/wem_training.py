#!/usr/bin/env python
# coding: utf-8

# Word Embedding Models: Training using word2vec in gensim
# Project title: Charter school identities
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: July 20, 2018
# Date last modified: July 24, 2018

# For more code see: https://github.com/jhaber-zz/Charter-school-identities

# ## Initialize Python

# Import packages
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec
import Cython # For parallelizing word2vec
import numpy as np # For working with numbers
import pickle # For working with .pkl files
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import os # For navigating file trees
import sys # For shell utilities like terminating script
import timeit # For counting time taken for a process
from tqdm import tqdm # Shows progress over iterations
from nltk.corpus import stopwords # for eliminating stop words
stopenglish = list(stopwords.words("english")) # assign list of english stopwords


# ## Read in data

# Define file paths
charters_path = "../../charters_full_2015.pkl" # All text data; only charter schools (regardless if open or not)
wordsent_path = "../data/wem_data.pkl"
phrasesent_path = "../data/wem_data_phrases.pkl"
wem_path = "../data/wem_model.txt"

if os.path.exists(phrasesent_path): # Check if phrase data already exists, to save time
    phrased = True
else:
    phrased = False

# Load in list of lists of words, where each list of words represents a sentence in its original order

words_by_sentence = [] # Initialize variable holding list of sentences

def load_cpickle_gc():
    '''Very time-efficient way to load pickle-formatted objects into Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Filepath to pickled (*.pkl) object.
    Output: Python object (probably a list of sentences or something similar).'''

    global words_by_sentence # Give access to variable storing list of sentences

    if phrased:
        output = open(phrasesent_path, 'rb')
    else:
        output = open(wordsent_path, 'rb')
    gc.disable() # disable garbage collector

    words_by_sentence = cPickle.load(output)

    gc.enable() # enable garbage collector again
    output.close()

if __name__ == '__main__':
    print("Loading list of sentences...")
    t = timeit.Timer(stmt="load_cpickle_gc()", globals=globals())
    print(round(t.timeit(1),4),'\n')

# Slower alternative for loading file:
#with open(wordsent_path, 'rb') as readfile:
#    print("Loading list of sentences...")
#    words_by_sentence = timeit(pickle.load(readfile))
    

# ## Prep for word2vec with common phrases

if phrased: # Check if phrased data already exists. If so, don't bother recalculating it
    pass

else:

    try:
        print("Detecting and parsing phrases in list of sentences...")
        phrases = Phrases(tqdm(words_by_sentence,desc="Detecting phrases"), min_count=3, delimiter=b'_', common_terms=stopenglish) # Detect phrases in sentences based on collocation counts
        words_by_sentence = [phrases[sent] for sent in tqdm(words_by_sentence, desc="Parsing phrases")] # Apply phrase detection model to each sentence in data

    except Exception as e:
        print(str(e))
        sys.exit()

    try:
        # Save data for later
        with open(phrasesent_path, 'wb') as destfile:
            gc.disable() # Disable garbage collector to increase speed
            cPickle.dump(words_by_sentence, destfile)
            gc.enable() # Enable garbage collector again

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
