#!/usr/bin/env python
# -*- coding: UTF-8

# # Parsing & Categorizing HTML from `wget` run with multiprocessing


"""This script parses .html files previously downloaded into local folders for those schools (or organizations, generally) listed in a .csv directory file. It uses the BeautifulSoup, multiprocessing, and pandas modules to efficiently clean, filter, and merge webtext into various lists; it also uses dictionary methods to count the number of times any word from the provided dictionaries (essentialist and progressivist school ideologies, in this case) occurs in any page for a given school. The script then stores these lists to each school's folder as text files; incorporates them into a large pandas DataFrame; and then finally saves this as an analysis-ready pickle-formatted file.

Author: Jaren Haber, PhD Candidate in UC Berkeley Sociology. 
Date: January 7th, 2018."""


# ## Initializing

# import necessary libraries
import os, re, fnmatch # for navigating file trees and working with strings
import csv # for reading in CSV files
#from glob import glob,iglob # for finding files within nested folders--compare with os.walk
import json, pickle, csv # For saving a loading dictionaries, DataFrames, lists, etc. in JSON, pickle, and CSV formats
from datetime import datetime # For timestamping files
import sys # For working with user input
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words
stemmer = PorterStemmer()
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
import urllib, urllib.request # for testing pages
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from multiprocessing import Pool # for multiprocessing, to increase parsing speed
import pandas as pd # modifies data more efficiently than with a list of dicts
from tqdm import tqdm # For progress information during involved Pandas operations

# Import parser
from bs4 import BeautifulSoup # BS reads and parses even poorly/unreliably coded HTML 
from bs4.element import Comment # helps with detecting inline/junk tags when parsing with BS
import lxml # for fast HTML parsing with BS
bsparser = "lxml"


# ### Set script options

Debug = False # Set to "True" for extra progress reports while algorithms run
notebook = False # Use different file paths depending on whether files are being accessed from shell (False) or within a Jupyter notebook (True)
usefile = False # Set to "True" if loading from file a dicts_list to add to. Confirms with user input first!
workstation = False # If working from office PC
numcpus = int(6) # For multiprocessing

if notebook:
    usefile = False # Prompting user for input file is only useful in command-line

inline_tags = ["b", "big", "i", "small", "tt", "abbr", "acronym", "cite", "dfn",
               "em", "kbd", "strong", "samp", "var", "bdo", "map", "object", "q",
               "span", "sub", "sup"] # this list helps with eliminating junk tags when parsing HTML


# ### Set directories

if workstation and notebook:
    dir_prefix = "C:\\Users\\Jaren\\Documents\\Charter-school-identities\\" # One level further down than the others
elif notebook:
    dir_prefix = "/home/jovyan/work/"
else:
    dir_prefix = "/vol_b/data/"

example_page = "https://westlakecharter.com/about/"
example_schoolname = "TWENTY-FIRST_CENTURY_NM"

if workstation and notebook:
    micro_sample13 = dir_prefix + "data\\micro-sample13_coded.csv" # Random micro-sample of 300 US charter schools
    URL_schooldata = dir_prefix + "data\\charter_URLs_2014.csv" # 2014 population of 6,973 US charter schools
    full_schooldata = dir_prefix + "data\\charter_merged_2014.csv" # Above merged with PVI, EdFacts, year opened/closed
    temp_data = dir_prefix + "data\\school_parser_temp.json" # Full_schooldata dict with output for some schools
    example_file = dir_prefix + "data\\example_file.html" #example_folder + "21stcenturypa.com/wp/default?page_id=27.tmp.html"
    dicts_dir = dir_prefix + "dicts\\" # Directory in which to find & save dictionary files
    save_dir = dir_prefix + "data\\" # Directory in which to save data files

else:
    wget_dataloc = dir_prefix + "wget/parll_wget/" #data location for schools downloaded with wget in parallel (requires server access)
    example_folder = wget_dataloc + "TWENTY-FIRST_CENTURY_NM/" # Random charter school folder
    example_file = dir_prefix + "wget/example_file.html" #example_folder + "21stcenturypa.com/wp/default?page_id=27.tmp.html"

    micro_sample13 = dir_prefix + "Charter-school-identities/data/micro-sample13_coded.csv" #data location for random micro-sample of 300 US charter schools
    URL_schooldata = dir_prefix + "Charter-school-identities/data/charter_URLs_2014.csv" #data location for 2014 population of US charter schools
    full_schooldata = dir_prefix + "Charter-school-identities/data/charter_merged_2014.csv" # Above merged with PVI, EdFacts, year opened/closed
    temp_data = dir_prefix + "Charter-school-identities/data/school_parser_temp.json" # Full_schooldata dict with output for some schools
    dicts_dir = dir_prefix + "Charter-school-identities/dicts/" # Directory in which to find & save dictionary files
    save_dir = dir_prefix + "Charter-school-identities/data/" # Directory in which to save data files
    

# Set input file, if any
if usefile and not notebook:
    print("\nWould you like to load from file a list of dictionaries to add to? (Y/N)")
    answer = input()
    if answer == "Y":
        print("Please indicate file path for dictionary list file.")
        answer2 = input()
        if os.path.exists(answer2):
            input_file = answer2
            usefile = True
        else:
            print("Invalid file path" + str(answer2) + " \nAborting script.")
            sys.exit()

    elif answer == "N":
        print("OK! This script will create a new file for this list of dictionaries.")
        usefile = False
    
    else:
        print("Response " + str(answer) + " not interpretable. \nAborting script.")
        sys.exit()


# ### Define (non-parsing) helper functions

def get_vars(data):
    """Defines variable names based on the data source called."""
    
    if data==URL_schooldata:
        URL_variable = "TRUE_URL"
        NAME_variable = "SCH_NAME"
        ADDR_variable = "ADDRESS"
        
    elif data==full_schooldata:
        URL_variable = "SCH_NAME" # Work-around until URLs merged into full data file
        NAME_variable = "SCH_NAME"
        ADDR_variable = "ADDRESS14"
    
    elif data==micro_sample13:
        URL_variable = "URL"
        NAME_variable = "SCHNAM"
        ADDR_variable = "ADDRESS"
    
    else:
        try:
            print("Error processing variables from data file " + str(data) + "!")
        except Exception as e:
            print("ERROR: No data source established!\n")
            print(e)
    
    return(URL_variable,NAME_variable,ADDR_variable)


def tag_visible(element):
    """Returns false if a web element has a non-visible tag, 
    i.e. one site visitors wouldn't actually read--and thus one we don't want to parse"""
    
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def webtext_from_files(datalocation):
    """Concatenate and return a single string from all webtext (with .txt format) in datalocation"""
    
    string = ""
    for root, dirs, files in os.walk(datalocation):
        for file in files:
            if file.endswith(".txt"):
                fileloc = open(datalocation+file, "r")
                string = string + (fileloc.read())
    return string


def remove_spaces(file_path):
    """Remove spaces from text file at file_path"""
    
    words = [x for x in open(file_path).read().split() if x != ""]
    text = ""
    for word in words:
        text += word + " "
    return text


def write_errors(error_file, error1, error2, error3, file_count):
    """Writes to error_file three binary error flags derived from parse_school(): 
    duplicate_flag, parse_error_flag, wget_fail_flag, and file_count."""
    
    with open(error_file, 'w') as file_handler:
        file_handler.write("duplicate_flag {}\n".format(int(error1)))
        file_handler.write("parse_error_flag {}\n".format(int(error2)))
        file_handler.write("wget_fail_flag {}\n".format(int(error3)))
        file_handler.write("file_count {}".format(int(file_count)))
        return
    

def write_counts(file_path, names_list, counts_list):
    """Writes to file_path the input dict_count names (a list) and counts (another list).
    Assumes these two lists have same length and are in same order--
    e.g., names_list[0]="ess_count" and counts_list[0]=ess_count."""
    
    with open(file_path, 'w') as file_handler:
        for tup in zip(names_list,counts_list): # iterate over zipped list of tuples
            if tup != list(zip(names_list,counts_list))[-1]:
                file_handler.write("{} {}\n".format(tup[0],tup[1]))
            else:
                file_handler.write("{} {}".format(tup[0],tup[1]))
        return

    
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

        
def save_datafile(data, file, mode):
    """Saves data to file using JSON, pickle, or CSV format (whichever was specified).
    Works with Pandas DataFrames or other objects, e.g. a list of dictionaries."""
    
    file = str(file)
    mode = str(mode)
    
    try:
        if mode.upper()=="JSON":
            if not file.endswith(".json"):
                file += ".json"
            with open(file, 'w') as outfile:
                if type(data)=="pandas.core.frame.DataFrame":
                    data.to_json(file)
                else:
                    json.dump(data, outfile)
                print("Data saved to " + file + "!")

        elif mode.lower()=="pickle":
            if not file.endswith(".pickle"):
                file += ".pickle"
            with open(file, 'wb') as outfile:
                if type(data)=="pandas.core.frame.DataFrame":
                    data.to_pickle(file)
                else:
                    pickle.dump(data, outfile)
                print("Data saved to " + file + "!")
                
        elif mode.upper()=="CSV":
            if not file.endswith(".csv"):
                file += ".csv"
            with open(file, 'w') as outfile:
                if type(data)=="pandas.core.frame.DataFrame":
                    data.to_csv(file,index=False)
                else:
                    wr = csv.writer(file)
                    wr.writerows(data)
                print("Data saved to " + file + "!")

        else:
            print("ERROR! Improper arguments. Please include: data object to save (Pandas DataFrames OK), file path, and file format ('JSON' or 'pickle').")
    
    except Exception as e:
        print("Failed to save to " + str(file) + " into memory using " + str(mode) + " format. Please check arguments (data, file, mode) and try again.")
        print(e)
    

def load_datafile(file):
    """Loads dicts_list (or whatever) from file, using either JSON or pickle format. 
    The created object should be assigned when called."""
    
    file = str(file)
    
    if file.lower().endswith(".json"):
        with open(file,'r') as infile:
            var = json.load(infile)
    
    if file.lower().endswith(".pickle"):
        with open(file,'rb') as infile:
            var = pickle.load(infile)
        
    print(file + " successfully loaded!")
    return var


def load_dict(custom_dict, file_path):
    """Loads in a dictionary. Adds each entry from the dict at file_path to the defined set custom_dict (the input), 
    which can also be an existing dictionary. This allows the creation of combined dictionaries!"""

    with open(file_path) as file_handler:
        line = file_handler.readline()
        while line:
            custom_dict.add(stemmer.stem(line.replace("\n", ""))) # Add line after stemming dictionary entries and eliminating newlines
            line = file_handler.readline() # Look for anything else in that line, add that too
    return custom_dict


def list_files(folder_path, extension):
    """Outputs a list of every file in folder_path or its subdirectories that has a specified extension.
    Prepends specified extension with '.' if it doesn't start with it already.
    If no extension is specified, it just returns all files in folder_path."""
    
    matches = []
    if extension:
        extension = str(extension) # Coerce to string, just in case
    
    if extension and not extension.startswith("."):
        extension = "." + extension
    
    for dirpath,dirnames,filenames in os.walk(folder_path):
        if extension:
            for filename in fnmatch.filter(filenames, "*" + extension): # Use extension to filter list of files
                matches.append(os.path.join(dirpath,filename))
        else:
                matches.append(os.path.join(dirpath,filename)) # If no extension, just take all files
    return matches


def has_html(folder_path):
    """Simple function that counts .html files and returns a binary:
    'True' if a specified folder has any .html files in it, 'False' otherwise."""
    
    html_list = []
    for dirpath,dirnames,filenames in os.walk(folder_path):
        for file in fnmatch.filter(filenames, "*.html"): # Check if any HTML files in folder_path
            html_list.append(file)
    
    if len(html_list)==0:
        return False
    else:
        return True

    
def set_failflag(folder_name):
    """The web_fail_flag indicates whether the webcrawl/download operation failed to capture any .html for a particular folder_name.
    This function sets the web_fail_flag depending on two conditions: 
    (1) Whether or not there exists a web download folder corresponding to folder_name, and
    (2) Whether or not that folder contains at least one file with the .html extension."""
    
    global wget_dataloc #,dicts_list # Need access to the list of dictionaries
    web_fail_flag = "" # make output a str to work with currently limited Pandas dtype conversion functionality
    
    folder_path = str(wget_dataloc) + folder_name + "/"
    if (not os.path.exists(folder_path)) or (has_html(folder_path)==False):
        web_fail_flag = str(1) # If folder doesn't exist, mark as fail and ignore when loading files
    else:
        web_fail_flag = str(0) # make str so can work with currently limited Pandas dtype conversion functionality
    
    #match_index = next((index for (index, d) in enumerate(dicts_list) if d["folder_name"] == folder_name), None) # Find dict index of input/folder_name
    #dicts_list[match_index]['wget_fail_flag'] = web_fail_flag # Assign output to dict entry for folder_name
    
    return web_fail_flag


def convert_df(df):
    """Makes a Pandas DataFrame more memory-efficient through intelligent use of Pandas data types: 
    specifically, by storing columns with repetitive Python strings not with the object dtype for unique values 
    (entirely stored in memory) but as categoricals, which are represented by repeated integer values. This is a 
    net gain in memory when the reduced memory size of the category type outweighs the added memory cost of storing 
    one more thing. As such, this function checks the degree of redundancy for a given column before converting it.
    
    # TO DO: Filter out non-object columns, make that more efficient by downcasting numeric types using pd.to_numeric(), 
    merge  that with the converted object columns (see https://www.dataquest.io/blog/pandas-big-data/). 
    For now, since the current DF is ENTIRELY composed of object types, code is left as is. 
    But note that the current code will eliminate any non-object type columns."""
    
    converted_df = pd.DataFrame() # Initialize DF for memory-efficient storage of strings (object types)
    df_obj = df.select_dtypes(include=['object']).copy() # Filter to only those columns of object data type

    for col in df.columns:
        num_unique_values = len(df_obj[col].unique())
        num_total_values = len(df_obj[col])
        if (num_unique_values / num_total_values) < 0.5: # Only convert data types if at least half of values are duplicates
            converted_df.loc[:,col] = df[col].astype('category') # Store these columns as dtype "category"
        else:
            converted_df.loc[:,col] = df[col]
    
    return converted_df


# ### Set parsing keywords

keywords = ['values', 'academics', 'skills', 'purpose',
                       'direction', 'mission', 'vision', 'vision', 'mission', 'our purpose',
                       'our ideals', 'ideals', 'our cause', 'curriculum','curricular',
                       'method', 'pedagogy', 'pedagogical', 'approach', 'model', 'system',
                       'structure','philosophy', 'philosophical', 'beliefs', 'believe',
                       'principles', 'creed', 'credo', 'values','moral', 'history', 'our story',
                       'the story', 'school story', 'background', 'founding', 'founded',
                       'established','establishment', 'our school began', 'we began',
                       'doors opened', 'school opened', 'about us', 'our school', 'who we are',
                       'our identity', 'profile', 'highlights']

mission_keywords = ['mission','vision', 'vision:', 'mission:', 'our purpose', 'our ideals', 'ideals:', 'our cause', 'cause:', 'goals', 'objective']
curriculum_keywords = ['curriculum', 'curricular', 'program', 'method', 'pedagogy', 'pedagogical', 'approach', 'model', 'system', 'structure']
philosophy_keywords = ['philosophy', 'philosophical', 'beliefs', 'believe', 'principles', 'creed', 'credo', 'value',  'moral']
history_keywords = ['history', 'story','our story', 'the story', 'school story', 'background', 'founding', 'founded', 'established', 'establishment', 'our school began', 'we began', 'doors opened', 'school opened']
about_keywords =  ['about us', 'our school', 'who we are', 'overview', 'general information', 'our identity', 'profile', 'highlights']

mission_keywords = set(stemmer.stem(word) for word in mission_keywords)
curriculum_keywords = set(stemmer.stem(word) for word in curriculum_keywords)
philosophy_keywords = set(stemmer.stem(word) for word in philosophy_keywords)
history_keywords = set(stemmer.stem(word) for word in history_keywords)
about_keywords =  set(stemmer.stem(word) for word in about_keywords)
all_keywords = set(stemmer.stem(key) for key in keywords)
    
if Debug:
    print("\nList of keywords:\n", list(all_keywords))


# ### Create dictionaries for each ideology and one for combined ideologies

ess_dict, prog_dict, rit_dict, all_ideol, all_dicts = set(), set(), set(), set(), set()
all_ideol = load_dict(all_ideol, dicts_dir + "ess_dict.txt")
all_ideol = load_dict(all_ideol, dicts_dir + "prog_dict.txt") # For complete ideological list, append second ideological dict
all_dicts = load_dict(all_ideol, dicts_dir + "rit_dict.txt") # For complete dict list, append ritual dict terms too
ess_dict = load_dict(ess_dict, dicts_dir + "ess_dict.txt")
prog_dict = load_dict(prog_dict, dicts_dir + "prog_dict.txt")
rit_dict = load_dict(rit_dict, dicts_dir + "rit_dict.txt")

if Debug:
    print(len(all_ideol), "entries loaded into the combined ideology dictionary.")
    list_dict = list(all_ideol)
    list_dict.sort(key = lambda x: x.lower())
    print("First 10 elements of combined ideology dictionary are:\n", list_dict[:10])
    

# Create tuples for keyword lists and dictionary terms:
keys_tuple = tuple([mission_keywords,curriculum_keywords,philosophy_keywords,history_keywords,about_keywords,\
                        all_ideol,all_keywords])
dicts_tuple = tuple([ess_dict,prog_dict,rit_dict,all_dicts])
    
if Debug:
    print(list(keys_tuple))
    print()
    print(list(dicts_tuple))

    
# ### Define dictionary matching helper functions

def dict_count(text_list, custom_dict):
    
    """Performs dictionary analysis, returning number of dictionary hits found.
    Removes punctuation and stems the phrase being analyzed. 
    Compatible with multiple-word dictionary elements."""
    
    counts = 0 # number of matches between text_list and custom_dict
    dictless_list = [] # Updated text_list with dictionary hits removed
    max_entry_length = max([len(entry.split()) for entry in custom_dict]) # Get length (in words) of longest entry in combined dictionary
    
    for chunk in text_list: # chunk may be several sentences or possibly paragraphs long
        chunk = re.sub(r'[^\w\s]', '', chunk) # Remove punctuation with regex that keeps only letters and spaces

        # Do dictionary analysis for word chunks of lengths max_entry_length down to 1, removing matches each time.
        # This means longer dict entries will get removed first, useful in case they contain smaller entries.
        for length in range(max_entry_length, 0, -1):
            dictless_chunk,len_counts = dict_match_len(chunk,custom_dict,length)
            dictless_list.append(dictless_chunk)
            counts += len_counts
    
    return dictless_list,int(counts)

def dict_match_len(phrase, custom_dict, length):
    
    """Helper function to dict_match. 
    Returns # dictionary hits and updated copy of phrase with dictionary hits removed. 
    Stems phrases before checking for matches."""
    
    hits_indices, counts = [], 0
    splitted_phrase = phrase.split()
    if len(splitted_phrase) < length:
        return phrase, 0 # If text chunk is shorter than length of dict entries being matched, don't continue.
    
    for i in range(len(splitted_phrase) - length + 1):
        to_stem = ""
        for j in range(length):
            to_stem += splitted_phrase[i+j] + " " # Builds chunk of 'length' words
        stemmed_word = stemmer.stem(to_stem[:-1]) # stem chunk
        if stemmed_word in custom_dict:
            hits_indices.append(i) # Store the index of the word that has a dictionary hit
            counts += 1
            if Debug:
                print(stemmed_word)
                
    # Iterate through list of matching word indices and remove the matches
    for i in range(len(hits_indices)-1, -1, -1):
        splitted_phrase = splitted_phrase[:hits_indices[i]] + \
        splitted_phrase[hits_indices[i] + length:]
    modified_phrase = ""
    for sp in splitted_phrase: # Rebuild the modified phrase, with matches removed
        modified_phrase += sp + " "
    return modified_phrase[:-1], counts


# ### Define parsing helper functions

def parsefile_by_tags(HTML_file):
    
    """Cleans HTML by removing inline tags, ripping out non-visible tags, 
    replacing paragraph tags with a random string, and finally using this to separate HTML into chunks.
    Reads in HTML from storage using a given filename, HTML_file."""

    random_string = "".join(map(chr, os.urandom(75))) # Create random string for tag delimiter
    soup = BeautifulSoup(open(HTML_file), bsparser)
    
    [s.extract() for s in soup(['style', 'script', 'head', 'title', 'meta', '[document]'])] # Remove non-visible tags
    for it in inline_tags:
        [s.extract() for s in soup("</" + it + ">")] # Remove inline tags
    
    visible_text = soup.getText(random_string).replace("\n", "") # Replace "p" tags with random string, eliminate newlines
    # Split text into list using random string while also eliminating tabs and converting unicode to readable text:
    visible_text = list(normalize("NFKC",elem.replace("\t","")) for elem in visible_text.split(random_string))
    visible_text = list(filter(lambda vt: vt.split() != [], visible_text)) # Eliminate empty elements
    # Consider joining list elements together with newline in between by prepending with: "\n".join

    return(visible_text)


if Debug:
    example_textlist = parsefile_by_tags(example_file)
    print("Output of parsefile_by_tags:\n\n", example_textlist, "\n\n")


def filter_dict_page(pagetext_list, keyslist):
    
    """Filters webtext of a given .html page, which is parsed and in list format, to only those strings 
    within pagetext_list containing an element (word or words) of inputted keyslist. 
    Returns list filteredtext wherein each element has original case (not coerced to lower-case)."""
    
    filteredtext = [] # Initialize empty list to hold strings of page
    
    for string in pagetext_list:
        lowercasestring = str(string).lower() # lower-case string...
        dict_list = [key.lower() for key in list(keyslist)] # ...compared with lower-case element of keyslist
        for key in dict_list:
            if key in lowercasestring and key in lowercasestring.split(' '): # Check that the word is the whole word not part of another one
                filteredtext.append(string)

    return filteredtext


if Debug:
    print("Output of filter_keywords_page with keywords:\n\n", filter_dict_page(example_textlist, all_keywords), "\n\n")    
    print("Output of filter_keywords_page with ideology words:\n\n", filter_dict_page(example_textlist, all_ideol), "\n\n")


def parse_school(schooltup):
    
    """This core function parses webtext for a given school. Input is tuple: (name, address, url).
    It uses helper functions to run analyses and then returning multiple outputs:
    full (partially cleaned) webtext, by parsing webtext of each .html file (removing inline tags, etc.) within school's folder, via parsefile_by_tags();
    all text associated with specific categories by filtering webtext to those with elements from a defined keyword list, via filter_keywords_page();
    AND COUNTS FOR DICT MATCHES
    
    For the sake of parsimony and manageable script calls, OTHER similar functions/scripts return these additional outputs: 
    parsed webtext, having removed overlapping headers/footers common to multiple pages, via remove_overlaps();
    all text associated with specific categories by filtering webtext according to keywords for 
    mission, curriculum, philosophy, history, and about/general self-description, via categorize_page(); and
    contents of those individual pages best matching each of these categories, via find_best_categories."""
    
    global itervar,numschools,parsed,wget_dataloc,dicts_list,keys_tuple,dicts_tuple # Access variables defined outside function (globally)
        
    itervar +=1 # Count school
    datalocation = wget_dataloc # Define path to local data storage
    school_name,school_address,school_URL,folder_name = schooltup[0],schooltup[1],schooltup[2],schooltup[3] # Assign variables from input tuple (safe because element order for a tuple is immutable)
    
    print("Parsing " + str(school_name) + ", which is ROUGHLY #" + str(6*itervar) + " / " + str(numschools) + " schools...")
    
    school_folder = datalocation + folder_name + "/"
    error_file = school_folder + "error_flags.txt" # Define file path for error text log
    
    if school_URL==school_name:
        school_URL = folder_name # Workaround for full_schooldata, which doesn't yet have URLs
    
    # PRELIMINARY TEST 1: Check if folder exists. If not, do not pass go; do not continue function.
    duplicate_flag,parse_error_flag,wget_fail_flag,file_count = 0,0,0,0 # initialize error flags
    
    if not (os.path.exists(school_folder) or os.path.exists(school_folder.lower()) or os.path.exists(school_folder.upper())):
        print("  !! NO DIRECTORY FOUND, creating " + str(school_folder) + " for 'error_flags.txt' and aborting...")
        wget_fail_flag = 1
        try:
            os.makedirs(school_folder) # Create empty folder for school to hold error_flags.txt (and nothing else)
            write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
            return
        except Exception as e:
            print("  Uh-oh! Failed to log error flags for " + str(school_name) + ".")
            print("    ",e)
            return
    
    # PRELIMINARY TEST 2: Check if this school has already been parsed via its unique school_URL. If so, skip this school to avoid duplication bias.
    if school_URL in parsed: 
        print("  DUPLICATE URL DETECTED. Skipping " + str(school_name) + "...")
        duplicate_flag = 1
        write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
        return
    
    # Next, initialize local (within-function) variables for text output
    webtext,keywords_text,ideology_text,dictless_words = [],[],[],[] # text category lists
    file_list = [] # list of HTML files in school_folder
    
    mission,curriculum,philosophy,history,about,ideology,keywords = [],[],[],[],[],[],[] # matched keyword lists
    ess_count, prog_count, rit_count, alldict_count, all_matches = 0,0,0,0,0 # dict match counts
    ess_dictless, prog_dictless, rit_dictless, alldict_dictless = [],[],[],[] # lists of unmatched words. Why?
    # Later we can revise the dictionaries by looking at what content words were not counted by current dictionaries. 

    titles_list = [mission,curriculum,philosophy,history,about,ideology,keywords] # list of matched keyword lists
    dictsnames_list = [ess_count, prog_count, rit_count, alldict_count] # list of dict match counts
    dictlessnames_list = [ess_dictless, prog_dictless, rit_dictless, alldict_dictless] # list of unmatched word lists

    keysnames_tupzip = zip(keys_tuple, titles_list) # zips together keyword lists with the variables holding their matches
    dictsnames_tupzip = zip(dicts_tuple, dictsnames_list, dictlessnames_list) # zips together dict terms lists with variables holding their matches and their not-matches

    if Debug:
        print(list(keysnames_tupzip))
        print()
        print(list(dictsnames_tupzip))
    
    # Now to parsing:
    try:
        # Parse file only if it contains HTML. This is easy: use the "*.html" wildcard pattern--
        # also wget gave the ".html" file extension to appropriate files when downloading (`--adjust-extension` option)
        # Less efficient ways to check if files contain HTML (e.g., for data not downloaded by wget):
        # if bool(BeautifulSoup(open(fname), bsparser).find())==True: # if file.endswith(".html"):
        # Another way to do this, maybe faster but broken: files_iter = iglob(school_folder + "**/*.html", recursive=True)
            
        file_list = list_files(school_folder, ".html") # Get list of HTML files in school_folder
            
        if file_list==(None or school_folder or "" or []) or not file_list or len(file_list)==0:
            print("  No .html files found. Aborting parser for " + str(school_name) + "...")
            wget_fail_flag = 1
            write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
            return
            
        for file in file_list:
                                    
            file_count+=1 # add to count of parsed files
            parsed_pagetext = []
                
            if Debug:
                print("    Parsing HTML in " + str(file) + "...")
            else:
                pass
                    
            # Parse and categorize page text:
            try:                    
                parsed_pagetext = parsefile_by_tags(file) # Parse page text (filter too?)
                        
                if len(parsed_pagetext) != 0: # Don't waste time adding empty pages
                    webtext.extend(parsed_pagetext) # Add new parsed text to long list
                    keywords_text.extend(filter_dict_page(parsed_pagetext, all_keywords)) # Filter using keywords
                    ideology_text.extend(filter_dict_page(parsed_pagetext, all_ideol)) # Filter using ideology words

                if Debug:
                    print("    Successfully parsed and filtered file " + str(file) + "...")
                else:
                    pass
                        
                file_count+=1

            except Exception as e:
                if Debug:
                    print("    ERROR! Failed to parse file...")
                    print("      ",e)
                else:
                    pass
                        
            # Count dict matches:
            try:
                for adict,count_name,dictless_name in dictsnames_tupzip: # Iterate over dicts to find matches with parsed text of file
                # Dicts are: (ess_dict, prog_dict, rit_dict, alldict_count); count_names are: (ess_count, prog_count, rit_count, alldict_count); dictless_names are: (ess_dictless, prog_dictless, rit_dictless, alldict_dictless)
                    count_add = 0 # Initialize iterator for dict-specific count matches
                    dictless_add,count_add = dict_count(parsed_pagetext,adict)
                    count_name += count_add
                    dictless_name += dictless_add
                    all_matches += count_add
                    
                    print("  Discovered " + str(count_add) + " matches for " + str(file) + ", a total thus far of " + str(count_name) + " matches...")

            except Exception as e:
                if Debug:
                    print("    ERROR! Failed to count number of dict matches while parsing " + str(file) + "...")
                    print("    ",e)
                else:
                    #print("    ERROR! Failed to count number of dict matches while parsing " + str(file) + "...")
                    #print("    ",e)
                    pass
                        
        # Report and save output to disk:
        parsed.append(school_URL)
        file_count = int(file_count-1)
        print("  PARSED " + str(file_count) + " .html file(s) from website of " + str(school_name) + "...")
            
        write_list(school_folder + "webtext.txt", webtext)
        write_list(school_folder + "keywords_text.txt", keywords_text)
        write_list(school_folder + "ideology_text.txt", ideology_text)
            
        print("  Counted " + str(all_matches) + " total dictionary matches for " + str(school_name) + " and found " + str(len(alldict_dictless)) + " uncounted words...") # int(ess_count)+int(prog_count)+int(rit_count)

        write_counts(school_folder + "dict_counts.txt", ["ess_count","prog_count","rit_count"], [ess_count, prog_count, rit_count])
        write_list(school_folder + "dictless_words.txt", alldict_dictless)
                    
        write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)

    except Exception as e:
        print("  ERROR! Failed to parse, categorize, and get dict matches on webtext of " + str(school_name))
        print("    ",e)
        parse_error_flag = 1
        write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)

    return

    
def dictify_webtext(school_dict):
    """Reads parsing output from text files and saves to school_dict multiple parsing outputs:
    webtext, keywords_text, ideology_text, file_count, etc."""
    
    # Allow function to access these variables already defined outside the function (globally)
    global itervar,numschools,parsed,wget_dataloc,URL_var,NAME_var,ADDR_var,save_dir
    
    datalocation = wget_dataloc # Define path to local data storage
    school_name, school_address, school_URL = school_dict[NAME_var], school_dict[ADDR_var], school_dict[URL_var] # Define varnames
    itervar+=1 # Count this school
    
    print("Loading into dict parsing output for " + str(school_name) + ", which is school #" + str(itervar) + " of " + str(numschools) + "...")
    
    school_dict["webtext"], school_dict["keywords_text"], school_dict["ideology_text"] = [[] for _ in range(3)]
    school_dict["duplicate_flag"], school_dict["parse_error_flag"], school_dict["wget_fail_flag"] = [0 for _ in range(3)]
    school_dict['ess_strength'],school_dict['prog_strength'] = [0.0 for _ in range(2)]
    
    folder_name = school_dict["folder_name"]
    
    school_folder = datalocation + folder_name + "/"
    error_file = school_folder + "error_flags.txt" # Define file path for error text log
    
    if school_URL==school_name:
        school_URL = folder_name # Workaround for full_schooldata, which doesn't yet have URLs

    # Check if folder exists. If not, exit function
    if not (os.path.exists(school_folder) or os.path.exists(school_folder.lower()) or os.path.exists(school_folder.upper())):
        print("  !! NO DIRECTORY FOUND matching " + str(school_folder) + ". Aborting dictify function...")
        school_dict['wget_fail_flag'] = 1
        return
    
    try:
        # Load school parse output from disk into dictionary 
        school_dict["webtext"] = load_list(school_folder + "webtext.txt")
        school_dict["keywords_text"] = load_list(school_folder + "keywords_text.txt")
        school_dict["ideology_text"] = load_list(school_folder + "ideology_text.txt")                        
        
        """ # Comment out until dict_count is run
        school_dict["ess_count"] = load_list(school_folder + "ess_count.txt")
        school_dict["prog_count"] = load_list(school_folder + "prog_count.txt")
        school_dict["rit_count"] = load_list(school_folder + "rit_count.txt")
        school_dict['ess_strength'] = float(school_dict['ess_count'])/float(school_dict['rit_count'])
        school_dict['prog_strength'] = float(school_dict['prog_count'])/float(school_dict['rit_count'])
        """

        # load error_file as a list with four pieces, the last element of each of which is the flag value itself:
        error_text = load_list(error_file) 
        school_dict["duplicate_flag"] = error_text[0].split()[-1] # last element of first piece of error_text
        school_dict["parse_error_flag"] = error_text[1].split()[-1]
        school_dict["wget_fail_flag"] = error_text[2].split()[-1]
        school_dict["html_file_count"] = error_text[3].split()[-1]
        
        if int(school_dict["html_file_count"])==0:
            school_dict["wget_fail_flag"] = 1 # If no HTML, then web download failed!
        
        print("  LOADED " + school_dict["html_file_count"] + " .html file(s) from website of " + str(school_name) + "...")
        #save_datafile(dicts_list, save_dir+"school_parser_temp", "JSON") # Save output so we can pick up where left off, in case something breaks before able to save final output
        return school_dict
    
    except Exception as e:
        print("  ERROR! Failed to load into dict parsing output for " + str(school_name))
        print("  ",e)
        school_dict["parse_error_flag"] = 1
        return

    
def pandify_webtext(df):
    """Reads parsing output from text files and saves to DataFrame df multiple parsing outputs:
    webtext, keywords_text, ideology_text, file_count, dict_count outputs, etc."""
    
    # Allow function to access these variables already defined outside the function (globally)
    global numschools,parsed,wget_dataloc,URL_var,NAME_var,ADDR_var,save_dir
    
    datalocation = wget_dataloc # Define path to local data storage
    # VARNAMES ARE: school_name, school_address, school_URL = df[NAME_var], df[ADDR_var], df[URL_var]
    
    print("Loading into DataFrame parsing output for " + str(len(df)) + " school websites out of a total of " + str(numschools) + "...")
    # df["folder_name"] = df[[NAME_var, ADDR_var]].apply(lambda x: re.sub(" ","_",("{} {}".format(str(x[0], x[1][-8:-6])))), axis=1) # This gives name and state separated by "_"  ## school["folder_name"] = re.sub(" ","_",(school[NAME_var]+" "+school[ADDR_var][-8:-6]))  ### Comment out while fixing parser
    df["school_folder"] = df.folder_name.apply(lambda x: str(datalocation) + '{}/'.format(str(x)))
    df["error_file"] = df.school_folder.apply(lambda x: '{}error_flags.txt'.format(str(x))) # Define file path for error text log
    
    #df["folder_path"] = df["folder_name"].apply(lambda x: str(wget_dataloc)+'{}/'.format(str(x)))
    #if df["folder_path"].apply(lambda x: list_files('{}, .html'.format(str(x))))==(None or df["folder_path"]) or not df["folder_path"].apply(lambda x: list_files('{}, .html'.format(str(x)))) or not (os.path.exists(wget_dataloc + school["folder_name"] + "/")) or os.path.exists(school["folder_name"].lower()) or os.path.exists(school["folder_name"].upper()):
    
    try:
        # load error_file as a list with four pieces, the last element of each of which is the flag value itself:
        df["error_text"] = df.error_file.apply(lambda x: load_list('{}'.format(str(x))))
        df["duplicate_flag"] = df.error_text.apply(lambda x: '{}'[0].split[-1].format(str(x))) # int(df.error_text[0].split()[-1]) # last element of first piece of error_text
        df["parse_error_flag"] = df.error_text.apply(lambda x: '{}'[1].split[-1].format(str(x))) #int(df.error_text[1].split()[-1])
        df["wget_fail_flag"] = df.error_text.apply(lambda x: '{}'[2].split[-1].format(str(x))) #int(df.error_text[2].split()[-1])
        df["html_file_count"] = df.error_text.apply(lambda x: '{}'[3].split[-1].format(str(x))) #int(df.error_text[3].split()[-1])
        
        #if df["html_file_count"]==0:
        #    df["wget_fail_flag"] = 1 # If no HTML, then web download failed! ## REDUNDANT with parse_school()
        
        #df['wget_fail_flag'] = df.folder_name.progress_apply(lambda x: set_failflag(x)) # Comment out while fixing parser
        downloaded = df["wget_fail_flag"].map({"1":True,1:True,"0":False,0:False}) == False # This binary conditional filters df to only those rows with downloaded web content (where wget_fail_flag==False and thus does NOT signal download failure)
        
        # Load school parse output from disk into DataFrame
        # df.loc[:,(downloaded,"keywords_text")] = df.loc[:,(downloaded,"school_folder")].progress_apply...
        df[downloaded]["webtext"] = df[downloaded].school_folder.progress_apply(lambda x: load_list("{}webtext.txt".format(str(x)))) # df["wget_fail_flag"]==False # `progress_apply` works because `tqdm` has been set up
        df[downloaded]["keywords_text"] = df[downloaded].school_folder.progress_apply(lambda x: load_list("{}keywords_text.txt".format(str(x))))
        df[downloaded]["ideology_text"] = df[downloaded].school_folder.progress_apply(lambda x: load_list("{}ideology_text.txt".format(str(x))))
        
        # Comment out until dict_count is run
        df[downloaded]["ess_count"] = df[downloaded].school_folder.progress_apply(lambda x: load_list("{}ess_count.txt".format(str(x)))) # load_list(df.school_folder + "ess_count.txt")
        df[downloaded]["prog_count"] = df[downloaded].school_folder.progress_apply(lambda x: load_list("{}prog_count.txt".format(str(x))))
        df[downloaded]["rit_count"] = df[downloaded].school_folder.progress_apply(lambda x: load_list("{}rit_count.txt".format(str(x))))
        df[downloaded]['ess_strength'] = float(df[downloaded]['ess_count'])/float(df[downloaded]['rit_count'])
        df[downloaded]['prog_strength'] = float(df[downloaded]['prog_count'])/float(df[downloaded]['rit_count'])
        
        df[downloaded].drop(["school_folder","error_text","error_file"],axis=1) # Clean up temp variables
        
        print("  LOADED " + df["html_file_count"].sum() + " .html files from into DataFrame!")
        save_datafile(df, save_dir+"df_parser_temp", "pickle") # Save output so we can pick up where left off, in case something breaks before able to save final output
        return df
    
    except Exception as e:
        print("    ERROR! Pandify function failed to load parsing output into DataFrame.")
        print("  ",e)
        sys.exit()
    

def slice_pandify(bigdf, numsplits, df_filepath):
    """This function uses pandify_webtext() to load the parsing output from local storage into a DataFrame.
    It gets around system memory limitations--which otherwise lead terminal to kill any attempts to pandify() all of bigdf--
    by splitting bigdf into numsplits smaller dfslices, parsing webtext into each slice, and recombining them
    by appending them to a big CSV on file. 
    The number of slices equals numsplits, and bigdf is split by numschools/ numsplits."""
    
    global numschools # Access numschools from within function (this is roughly 7000)
    wheresplit = int(round(float(numschools)/numsplits)) # Get number on which to split (e.g., 1000) based on total number of schools data
    
    for num in range(numsplits): # tqdm(range(numsplits), desc="Loading dfslices"): # Wrap iterator with tqdm to show progress bar
        try:
            dfslice = pd.DataFrame()
            dfslice = bigdf.iloc[(wheresplit*int(num)):(wheresplit*int(num+1)),:]
            dfslice = pandify_webtext(dfslice) # Load parsed output into the DF
            if num==1:
                save_datafile(dfslice,df_filepath,"CSV") # Save this first chunk of results to new file
            else:
                dfslice.to_csv(df_filepath,mode="a",columns=False,index=False) # Append this next chunk of results to existing saved results
            del dfslice # Free memory by deleting this temporary, smaller slice
            
        except Exception as e:
            print("  ERROR! Script failed to load parsing output into DataFrame slice #" + str(num) + " of " + str(numsplits) + ".")
            print("  ",e)
            sys.exit()
            
    return
            
            
# ### Preparing data to be parsed

itervar = 0 # initialize iterator that counts number of schools already parsed--useless when multiprocessing
parsed = [] # initialize list of URLs that have already been parsed
dicts_list = [] # initialize list of dictionaries to hold school data
schooldf = pd.DataFrame() # initialize DataFrame to hold school data

# If input_file was defined by user input in beginning of script, use that to load list of dictionaries. We'll add to it!
if usefile and not dicts_list:
    dicts_list = load_datafile(input_file)
    data_loc = full_schooldata # If loading data, assume we're running on full charter population

else:
    # set charter school data file and corresponding varnames:
    
    data_loc = full_schooldata # Run at scale using URL list of full charter population
    # data_loc = micro_sample13 # This seems nice for debugging--except directories don't match because different data source
        
    # Create dict list from CSV on file, with one dict per school
    with open(data_loc, 'r', encoding = 'Latin1') as csvfile: # open data file
        reader = csv.DictReader(csvfile) # create a reader
        for row in reader: # loop through rows
            dicts_list.append(row) # append each row to the list
        
URL_var,NAME_var,ADDR_var = get_vars(data_loc) # get varnames depending on data source
numschools = int(len(dicts_list)) # Count number of schools in list of dictionaries
names,addresses,urls,folder_names = [[] for _ in range(4)]


for school in dicts_list: # tqdm(dicts_list, desc="Setting web_fail_flags"): # Wrap iterator with tqdm to show progress bar
    names.append(school[NAME_var])
    addresses.append(school[ADDR_var])
    urls.append(school[URL_var])
    school["folder_name"] = re.sub(" ","_",(school[NAME_var]+" "+school[ADDR_var][-8:-6])) # This gives name and state separated by "_"
    folder_names.append(school["folder_name"])
    # school['wget_fail_flag'] = set_failflag(school["folder_name"]) # REDUNDANT with parse_school()
    # save_datafile(dicts_list, save_dir+"school_parser_temp", "JSON") # Save output so we can pick up where left off, in case something breaks
    
tuplist_zip = zip(names, addresses, urls, folder_names) # Create list of tuples to pass to parser function

""" # REDUNDANT with parse_school
if __name__ == '__main__':
    with Pool(numcpus) as p: # Use multiprocessing.Pool(numcpus) to speed things up
        p.map(set_failflag, tqdm(folder_names, desc="Setting web_fail_flags"), chunksize=numcpus)
        """
    
# Now create a Pandas DataFrame (from dicts_list or from file) and store the data in a memory-efficient way:
schooldf = pd.DataFrame.from_dict(dicts_list) # Convert dicts_list into a DataFrame
#schooldf = pd.read_pickle(save_dir+"school_dicts_2018-01-10.pickle") # Use existing file while debugging pandify_webtext()
#schooldf = pd.read_csv(data_loc) # Creating school_df from scratch
#schooldf = convert_df(schooldf) # Make this DF memory-efficient by converting appropriate columns to category data type # TO DO: UN-DO
tqdm.pandas(desc="Loading DF") # To show progress, create & register new `tqdm` instance with `pandas`


# ### Run parsing algorithm on schools (requires access to webcrawl output)

if Debug:
    test_dicts = dicts_list[:1] # Limit number of schools to test/refine methods
    for school in test_dicts:
        parse_school(school)
    dictfile = "testing_dicts_" + str(datetime.today())
    save_datafile(test_dicts, save_dir+dictfile, "JSON")
    sys.exit()
                
# Run parse_school() with multiprocessing.Pool(numcpus), 
# which parses downloaded webtext and saves the results to local storage:
if __name__ == '__main__':
    with Pool(numcpus) as p:
        p.map(parse_school, tqdm(list(tuplist_zip)), chunksize=numcpus)


# ### Load parsing output from disk into analyzable Python object (DataFrame or dicts_list)
        
"""# Now use dictify_webtext to load the parsing output from local storage into the list of dictionaries:
for school in dicts_list:
    try:
        school = dictify_webtext(school)
    except Exception as e:
        print("  ERROR! Failed to load into dict parsing output for " + school[NAME_var])
        print("  ",e)
        school_dict["parse_error_flag"] = 1
        continue"""


"""
# To work with limited system memory (almost there!), split this df into chunks, compute on each, and recombine later.
# The number of chunks equals numcpus, and schooldf is split by numschools/numcpus. 
for num in range(numcpus):
    try:
        "splitdf{}".format(str(num)) = pd.DataFrame()
        dfslice = pd.DataFrame()
        dfslice = schooldf.iloc[(splitnum*int(num-1)):(splitnum*int(num)),:]
    except Exception as e:
        print("  ERROR! Script failed to split schooldf into smaller DataFrame #" + str(num) + " of " + str(numcpus) + ".")
        print("  ",e)
        sys.exit()

# Now use pandify_webtext to load the parsing output from local storage into the DataFrame:
splitnum = int(round(float(numschools)/numcpus)) # Get chunk number based on total number of schools data
names_list = ["" for _ in range(numcpus)]
for num in numcpus:
    try:
        names_list[num] = "splitdf{}".format(str(num))
        dfslice = pd.DataFrame()
        dfslice = schooldf.iloc[(splitnum*int(num-1)):(splitnum*int(num)),:]
        dfslice = pandify_webtext(dfslice) # Load parsed output into the DF
        if num==1:
            save_datafile(dfslice,merged_df_file,"CSV") # Save this first chunk of results to new file
        else:
            dfslice.to_csv(merged_df_file,mode="a",columns=False,index=False) # Append this next chunk of results to existing saved results
        del dfslice # Free memory by deleting smaller slice
        
        if num = numcpus:
            del schooldf # Free memory by deleting full df, now that all the slices have been taken out
        
        splitdf1,splitdf2,splitdf3 = [pd.DataFrame() for _ in range(3)] # Initialize DFs to split into
        splitnum = int(round(float(numschools)/3)) # Get chunk number based on total number of schools data
        splitdf1 = schooldf.iloc[:splitnum,:]
        splitdf2 = schooldf.iloc[splitnum:splitnum*2,:]
        splitdf3 = schooldf.iloc[splitnum*2:,:]

        splitdf1 = pandify_webtext(splitdf1)
        save_datafile(splitdf1,merged_df_file,"CSV") # Save this first half of results
        del splitdf1 # Free memory

        splitdf2 = pandify_webtext(splitdf2)
        splitdf2.to_csv(merged_df_file,mode="a",columns=False,index=False) # Append these results to existing saved results
        del splitdf2 # Free memory

        splitdf3 = pandify_webtext(splitdf3)
        splitdf3.to_csv(merged_df_file,mode="a",columns=False,index=False) # Append these results to existing saved results
        del splitdf3 # Free memory
                  
    except Exception as e:
        print("  ERROR! Script failed to load parsing output into DataFrame #" + str(num) + " of " + str(numcpus) + "."")
        print("  ",e)
        sys.exit()"""
                  

# Load parsing output into big pandas DataFrame through slices (to work with limited system memory):
if dicts_list:
    del dicts_list # Free memory
    
merged_df_file = save_dir+"mergedf_"+str(datetime.today().strftime("%Y-%m-%d"))+".csv" # Prepare file name
slice_pandify(schooldf, numcpus, merged_df_file)
print("Larger DF successfully split into " + str(numpcus) + " smaller DFs, parsed, combined, and saved to file!")

if schooldf:
    del schooldf # Free memory
    
    
# Save final output:
print("\nSCHOOL PARSING COMPLETE!!!")
schooldf = pd.read_csv(merged_df_file) # Load full DF so we can save it pickle-style
dictfile = "school_dicts_" + str(datetime.today().strftime("%Y-%m-%d"))
save_datafile(schooldf, save_dir+dictfile, "pickle")