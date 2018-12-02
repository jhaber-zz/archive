#!/usr/bin/env python
# -*- coding: UTF-8

# Word Embedding Models: Preprocessing and Model Training
# Project title: Charter school identities 
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: July 20, 2018
# Date last edited: November 8, 2018


# ## Initialize

# Import general packages
import imp, importlib # For working with modules
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
import datetime # For working with dates & times

# Import packages for cleaning, tokenizing, and stemming text
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
from nltk.corpus import stopwords # for eliminating stop words
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

# Import packages for multiprocessing
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function
import Cython # For parallelizing word2vec

mpdo = False # Set to 'True' if using multiprocessing--faster for creating words by sentence file, but more complicated
    

# ## Prepare to read data

# Define file paths
if mpdo:
    wordsent_path = "../data/wem_wordsent_data_train250_nostem_unlapped_clean2.txt"
else:
    wordsent_path = "../data/wem_wordsent_data_train250_nostem_unlapped_clean2.pkl"
charters_path = "../../nowdata/traincf_2015.pkl" # All text data; only charter schools (regardless if open or not)
phrasesent_path = "../data/wem_phrasesent_data_train250_nostem_unlapped_clean2.pkl"
#wemdata_path = "../data/wem_data.pkl"
model_path = "../data/wem_model_train250_nostem_unlapped_300d_clean2.txt"
vocab_path = "../data/wem_vocab_train250_nostem_unlapped_300d_clean2.txt"
vocab_path_old = "../data/wem_vocab_train250_nostem_unlapped_300d_clean.txt"

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
    
    
# ## Create lists of stopwords, punctuation, and unicode characters

# Create stopwords list
stop_word_list = list(set(stopwords.words("english"))) # list of english stopwords

# Add dates to stopwords
for i in range(1,13):
    stop_word_list.append(datetime.date(2008, i, 1).strftime('%B'))
for i in range(1,13):
    stop_word_list.append((datetime.date(2008, i, 1).strftime('%B')).lower())
for i in range(1, 2100):
    stop_word_list.append(str(i))

# Add other common stopwords
stop_word_list.append('00') 
stop_word_list.extend(['mr', 'mrs', 'sa', 'fax', 'email', 'phone', 'am', 'pm', 'org', 'com', 
                       'Menu', 'Contact Us', 'Facebook', 'Calendar', 'Lunch', 'Breakfast', 'FAQs', 'FAQ']) # web stopwords
stop_word_list.extend(['el', 'en', 'la', 'los', 'para', 'las', 'san']) # Spanish stopwords
stop_word_list.extend(['angeles', 'diego', 'harlem', 'bronx', 'austin', 'antonio']) # cities with many charter schools

# Add state names & abbreviations (both uppercase and lowercase) to stopwords
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", 
          "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", 
          "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", 
          "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", 
          "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WI", "WV", "WY", 
          "Alabama", "Alaska", "Arizona", "Arkansas", "California", 
          "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", 
          "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", 
          "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", 
          "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", 
          "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
          "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", 
          "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", 
          "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", 
          "Vermont", "Virginia", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
for state in states:
    stop_word_list.append(state)
for state in [state.lower() for state in states]:
    stop_word_list.append(state)

# Add to stopwords useless and hard-to-formalize words/chars from first chunk of previous model vocab (e.g., a3d0, \fs19)
# First create whitelist of useful terms probably in that list, explicitly exclude from junk words list both these and words with underscores (common phrases)
whitelist = ["Pre-K", "pre-k", "pre-K", "preK", "prek", 
             "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]
with open(vocab_path_old) as f: # Load vocab from previous model
    junk_words = f.read().splitlines() 
junk_words = [word for word in junk_words[:8511] if ((not "_" in word) 
                                                     and (not any(term in word for term in whitelist)))]
stop_word_list.extend(junk_words)
    
# Create punctuations list
import string # for one method of eliminating punctuation
punctuations = list(string.punctuation) # assign list of common punctuation symbols
addpuncts = ['*','•','©','–','`','’','“','”','»','.','×','|','_','§','…','⎫'] # a few more punctuations also common in web text
punctuations += addpuncts # Expand punctuations list
punctuations.remove('-') # Don't remove hyphens - dashes at beginning and end of words are handled separately)
punctuations.remove("'") # Don't remove possessive apostrophes - those at beginning and end of words are handled separately
punctstr = "".join([char for char in list(set(punctuations))]) # Turn into string for regex later

# Create list of unicode characters
unicode_list  = []
for i in range(1000,3000):
    unicode_list.append(chr(i))
unicode_list.append("_cid:10") # Common in webtext junk

print("Sentence cleaning preliminaries complete...")


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
    
    
def write_list(file_path, textlist):
    """Writes textlist to file_path. Useful for recording output of parse_school()."""
    
    with open(file_path, 'w') as file_handler:
        
        for elem in textlist:
            file_handler.write("{}\n".format(elem))
    
    return    


def load_list(file_path):
    """Loads list into memory. Must be assigned to object."""
    
    textlist = []
    with open(file_path) as file_handler:
        line = file_handler.readline()
        while line:
            textlist.append(line)
            line = file_handler.readline()
    return textlist
    
    
def write_sentence(sentence, file_path):
    """Writes sentence to file at file_path.
    Useful for recording first row's output of preprocess_wem() one sentence at a time.
    Input: Sentence (list of strings), path to file to save it
    Output: Nothing (saves to disk)"""
    
    with open(file_path, 'w+') as file_handler:
        for word in sentence: # Iterate over words in sentence
            if word == "":
                pass
            else:
                file_handler.write(word + " ") # Write each word on same line, followed by space
            
        file_handler.write("\n") # After sentence is fully written, close line (by inserting newline)
            
    return


def append_sentence(sentence, file_path):
    """Appends sentence to file at file_path. 
    Useful for recording each row's output of preprocess_wem() one sentence at a time.
    Input: Sentence (list of strings), path to file to save it
    Output: Nothing (saves to disk)"""

    with open(file_path, 'a+') as file_handler:
        for word in sentence: # Iterate over words in sentence
            if word == "":
                pass
            else:
                file_handler.write(word + " ") # Write each word on same line, followed by space
            
        file_handler.write("\n") # After sentence is fully written, close line (by inserting newline)
            
    return

    
def load_tokslist(file_path):
    """Loads from file and word-tokenizes list of "\n"-separated, possibly multi-word strings (i.e., sentences). 
    Output must be assigned to object.
    Input: Path to file with list of strings
    Output: List of word-tokenized strings, i.e. sentences"""
    
    textlist = []
    
    with open(file_path) as file_handler:
        line = file_handler.readline() # Read first line
        
        while line: # Continue while there's still a line to read
            textlist.append(word for word in word_tokenize(line)) # Tokenize each line by word while loading in
            line = file_handler.readline() # Read next line
            
    return textlist


def clean_sentence(sentence):
    """Removes numbers, emails, URLs, unicode characters, hex characters, and punctuation from a sentence 
    separated by whitespaces. Returns a tokenized, cleaned list of words from the sentence.
    
    Args: 
        Sentence, i.e. string that possibly includes spaces and punctuation
    Returns: 
        Cleaned & tokenized sentence, i.e. a list of cleaned, lower-case, one-word strings"""
    
    global unicode_list, punctstr, stop_word_list # Access useful lists
    
    # Replace unicode spaces, tabs, and underscores with spaces, and remove whitespaces from start/end of sentence:
    sentence = sentence.replace(u"\xa0", u" ").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")
    
    # Remove hex characters (e.g., \xa0\, \x80):
    sentence = re.sub(r'[^\x00-\x7f]', r'', sentence) #replace anything that starts with a hex character 

    # Replace \\x, \\u, \\b, or anything that ends with \u2605
    sentence = re.sub(r"\\x.*|\\u.*|\\b.*|\u2605$", "", sentence)
        
    # Remove all elements that appear in unicode_list (looks like r'u1000|u10001|'):
    sentence = re.sub(r'|'.join(map(re.escape, unicode_list)), '', sentence)
    
    sent_list = [] # Initialize empty list to hold tokenized sentence (words added one at a time)
    
    for word in word_tokenize(sentence): # Tokenize and iterate over words
        
        # Skip stopwords, emails, and URLs:
        if ((word not in stop_word_list) 
            and ("@" not in word) 
            and ((not word.startswith(('http', 'https', 'www')))) 
            and (not word.endswith(('.com', '.net', '.gov', '.org'))) 
            and (not word.startswith('//')) 
            and (not word.endswith(('.jpg', '.pdf', 'png', 'jpeg', 'php')))): 
            
            # Remove punctuation (only after URLs removed) and lower-case:
            word = re.sub(r"["+punctstr+"]|[-$]+|^-+|['$]+|^'+", r'', word.lower()) # Remove dashes and apostrophes only from start/end of words
            
            if not word.replace('k','').replace('e','').replace('a','').replace('am','').replace('p','').replace('pm', '').replace('-','').isdigit(): # Remove numbers
                
                sent_list.append(word) # Add word to list

    return sent_list # Return clean, tokenized sentence


def preprocess_wem(tuplist): # inputs were formerly: (tuplist, start, limit)
    
    '''This function cleans and tokenizes sentences, removing punctuation and numbers and making words into lower-case stems.
    Inputs: list of four-element tuples, the last element of which holds the long string of text we care about;
        an integer limit (bypassed when set to -1) indicating the DF row index on which to stop the function (for testing purposes),
        and similarly, an integer start (bypassed when set to -1) indicating the DF row index on which to start the function (for testing purposes).
    This function loops over five nested levels, which from high to low are: row, tuple, chunk, sentence, word.
    Note: This approach maintains accurate semantic distances by keeping stopwords.'''
        
    global mpdo # Check if we're doing multiprocessing. If so, then mpdo=True
    global words_by_sentence # Grants access to variable holding a list of lists of words, where each list of words represents a sentence in its original order (only relevant for this function if we're not using multiprocessing)
    global pcount # Grants access to preprocessing counter
    
    known_pages = set() # Initialize list of known pages for a school

    if type(tuplist)==float:
        return # Can't iterate over floats, so exit
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for tup in tuplist: # Iterate over tuples in tuplist (list of tuples)
        if tup[3] in known_pages or tup=='': # Could use hashing to speed up comparison: hashlib.sha224(tup[3].encode()).hexdigest()
            continue # Skip this page if exactly the same as a previous page on this school's website

        for chunk in tup[3].split('\n'): #.split('\x').replace('\xa0','').replace('\x92',''):
            for sent in sent_tokenize(chunk): # Tokenize chunk by sentences (in case >1 sentence in chunk)
                sent = clean_sentence(sent) # Clean and tokenize sentence
                
                # Save preprocessing sentence to file (if multiprocessing) or to object (if not multiprocessing)
                if mpdo:
                    try: 
                        if (os.path.exists(wordsent_path)) and (os.path.getsize(wordsent_path) > 0): 
                            append_sentence(sent, wordsent_path) # If file not empty, add to end of file
                        else:
                            write_sentence(sent, wordsent_path) # If file doesn't exist or is empty, start file
                    except FileNotFoundError or OSError: # Handle common errors when calling os.path functions on non-existent files
                        write_sentence(sent, wordsent_path) # Start file
                
                else:
                    words_by_sentence.append(sent) # If not multiprocessing, just add sent to object
                    
                    
        known_pages.add(tup[3])
    
    pcount += 1 # Add to counter
    
    return


# ## Preprocessing I: Tokenize web text by sentences

df = quickpickle_load(charters_path) # Load charter data into DF
print("DF loaded from " + str(charters_path) + "...")

if phrased: 
    pass # If parsed sentence phrase data exists, don't bother with tokenizing sentences

elif sented: # Check if tokenized sentence data already exists. If so, don't bother reparsing it
    words_by_sentence = []
    
    # Load data back in for parsing phrases and word embeddings model:
    if mpdo:
        words_by_sentence = load_tokslist(wordsent_path) 
    else:
        words_by_sentence = quickpickle_load(wordsent_path) 

else:
    
    words_by_sentence = [] # Initialize variable to hold list of lists of words (sentences)
    pcount=0 # Initialize preprocessing counter
    df["WEBTEXT"] = df["WEBTEXT"].astype(list) # Coerce these to lists in order to avoid type errors

    # Convert DF into list (of lists of tuples) and call preprocess_wem on element each using Pool():
    try:
        tqdm.pandas(desc="Tokenizing sentences") # To show progress, create & register new `tqdm` instance with `pandas`

        # WITH multiprocessing (faster):
        if mpdo:
            weblist = df["WEBTEXT"].tolist() # Convert DF into list to pass to Pool()

            # Use multiprocessing.Pool(numcpus) to run preprocess_wem:
            print("Preprocessing web text into list of sentences...")
            if __name__ == '__main__':
                with Pool(numcpus) as p:
                    p.map(preprocess_wem, tqdm(weblist, desc="Tokenizing sentences")) 

        # WITHOUT multiprocessing (much slower):
        else:
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
        phrases = Phrases(words_by_sentence, min_count=3, delimiter=b'_', common_terms=stop_word_list, threshold=10) # Detect phrases in sentences based on collocation counts
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
- size = 300: Use hundreds of dimensions/degrees of freedom to generate accurate models from this large data set
- window = 8: Observe window of 8 context words in each direction, keeping word-word relationships moderately tight
- min_count = 3: Exclude very rare words, which occur just once or twice and typically are irrelevant proper nouns
- sg = 1: I choose a 'Skip-Gram' model over a CBOW (Continuous Bag of Words) model because skip-gram works better with larger data sets. It predicts words from contexts, rather than smoothing over context information by counting each context as a single observation
- alpha = 0.025: Initial learning rate: prevents model from over-correcting, enables finer tuning
- min_alpha = 0.001: Learning rate linearly decreases to this value over time, so learning happens more strongly at first
- iter = 10: Ten passes/iterations over the dataset
- batch_words = 20000: During each pass, sample batch size of 20000 words
- workers = 1: Set to 1 to guarantee reproducibility, OR accelerate by parallelizing model training across all vCPUs
- seed = 0: To increase reproducibility of model training 
- negative = 5: Draw 5 "noise words" in negative sampling in order to simplify weight tweaking
- ns_exponent = 0.75: Shape negative sampling distribution using 3/4 power, which outperforms other exponents (as popularized by original word2vec paper, Mikolov et al 2013) and slightly weights against high-frequency words (1 is exact frequencies, 0 is all words equally)
'''

# Train the model with above parameters:
try:
    print("Training word2vec model...")
    model = gensim.models.Word2Vec(words_by_sentence, size=300, window=8, min_count=3, sg=1, alpha=0.025, min_alpha=0.001,\
                                   iter=10, batch_words=20000, workers=1, seed=0, negative=5, ns_exponent=0.75)
    print("word2vec model TRAINED successfully!")

    # Save model:
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
               
    # Load word2vec model and save vocab list
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
    write_list(vocab_path, sorted(list(model.vocab)))
    print("word2vec model VOCAB saved to " + str(vocab_path))

except Exception as e:
    print(str(e))

sys.exit() # Kill script when done, just to be safe