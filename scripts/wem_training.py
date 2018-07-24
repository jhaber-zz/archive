#!/usr/bin/env python
# coding: utf-8

# Word Embedding Models: Preprocessing
# Project title: Charter school identities 
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: July 20, 2018


# ## Initialize Python

# Import key packages
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec
import Cython # For parallelizing word2vec
import numpy as np # For working with numbers
import pickle # For working with .pkl files
import sys # For shell utilities like terminating script
from timeit import timeit # For counting length of process
from tqdm import tqdm # Shows progress over iterations


# ## Read in data

# Define file paths
charters_path = "../../charters_full_2015.pkl" # All text data; only charter schools (regardless if open or not)
wordsent_path = "../data/wem_data.pkl"
wem_model = "../data/wem_model.txt"

# Load in list of lists of words, where each list of words represents a sentence in its original order
with open(wem_path, 'rb') as readfile:
    words_by_sentence = pickle.load(readfile)
    
datalen = len(words_by_sentence) # Set length of data for reference when modeling


# ## Prep for word2vec with common phrases

try:
    #print("Parsing phrases in list of sentences...")
    phrases = Phrases(words_by_sentence, min_count=3, delimiter=b'_', common_terms=stopenglish) # Detect phrases in sentences based on collocation counts
    sent = phrases(sent) for sent in tqdm(words_by_sentence, desc="Parsing phrases") # Apply phrase detection model to each sentence in data

    # Take a look at the data 
    print("Sample of the first 50 sentences:")
    print(words_by_sentence[:50])

except Exception as e:
    print(str(e))
    sys.exit()
    

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
- workers = 44: Parallelize model training across the 44 vCPUs of the XXL Jetstream VM (set to 1 to guarantee reproducibility, but with this much data, speed matters)
- seed = 43: To increase reproducibility of model training 
- negative = 5: Draw 5 "noise words" in negative sampling in order to simplify weight tweaking
- ns_exponent = 0.75: Shape negative sampling distribution using 3/4 power, which outperforms other exponents (as popularized by original word2vec paper, Mikolov et al 2013) and slightly weights against high-frequency words (1 is exact frequencies, 0 is all words equally)
'''

# Train the model with above parameters:
try:
    print("Training word2vec model...")
    timeit(model = gensim.models.Word2Vec(words_by_sentence, size=200, window=6, min_count=3, sg=1, alpha=0.025, min_alpha=0.001, iter=5, batch_words=10000, workers=44, seed=43, negative=5, ns_exponent=0.75))
    
    print("word2vec model TRAINED successfully!")

    # Save model for later:
    with open(wem_model, 'wb') as destfile:
        try:
            model.wv.save_word2vec_format(destfile)
        except Exception as e:
            print(str(e))
            print("Trying other save option...")
            try:
                model.save(destfile)
                
    print("word2vec model SAVED to " + str(destfile))
                
except Exception as e:
    print(str(e))
    

sys.exit() # Kill script when done, just to be safe