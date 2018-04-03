#!/usr/bin/env python
# -*- coding: UTF-8

# # Parsing & Categorizing HTML from `wget` run with multiprocessing


"""This script parses .html files previously downloaded into local folders for those schools (or organizations, generally) listed in a .csv directory file. It uses the BeautifulSoup and multiprocessing to efficiently clean, filter, and merge webtext into various lists; it also uses dictionary methods to count the number of times any word from the provided dictionaries (essentialist and progressivist school ideologies, in this case) occurs in any page for a given school.

Author: Jaren Haber, PhD Candidate in UC Berkeley Sociology. 
Date: January 7th, 2018."""


# ## Initializing

# import necessary libraries
import os, re, fnmatch # for navigating file trees and working with strings
import csv # for reading in CSV files
#from glob import glob,iglob # for finding files within nested folders--compare with os.walk
import json, pickle, csv # For saving a loading dictionaries, DataFrames, lists, etc. in JSON, pickle, and CSV formats
from datetime import datetime # For timestamping files
import time, timeout_decorator # To prevent troublesome files from bottlenecking the parsing process, use timeouts
import sys # For working with user input
import logging # for logging output, to help with troubleshooting
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words
stemmer = PorterStemmer()
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
import urllib, urllib.request # for testing pages
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from multiprocessing import Pool # for multiprocessing, to increase parsing speed
from tqdm import tqdm # For progress information over iterations, including with Pandas operations via "progress_apply"

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
    temp_dir = dir_prefix + "data\\temp\\" # Directory in which to save temporary data files

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
    temp_dir = dir_prefix + "Charter-school-identities/data/temp/" # Directory in which to save temporary data files
    
# Set logging options
log_file = temp_dir + "logfile_" + str(datetime.today()) + ".log"
logging.basicConfig(filename=log_file,level=logging.INFO)
    
    
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
        print("OK! This script will create a new data file at " + str(save_dir) + ".")
        usefile = False
    
    else:
        print("Error: " + str(answer) + " not an interpretable response. Aborting script.")
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

        
def save_datafile(data, file, thismode):
    """Saves data to file using JSON, pickle, or CSV format (whichever was specified).
    Works with Pandas DataFrames or other objects, e.g. a list of dictionaries.
    Deletes file first to reduce risk of data duplication."""
    
    file = str(file)
    thismode = str(thismode)
    
    try:
        if os.path.exists(file):
            os.remove(file) # Delete file first to reduce risk of data duplication
        else:
            pass
        
        if thismode.upper()=="JSON" or thismode.upper()==".JSON":
            if not file.endswith(".json"):
                file += ".json"
            with open(file, 'w') as outfile:
                if type(data)=="pandas.core.frame.DataFrame":
                    data.to_json(outfile)
                else:
                    json.dump(data, outfile)
                print("Data saved to " + file + "!")

        elif thismode.lower()=="pickle" or thismode.lower()==".pickle":
            if not file.endswith(".pickle"):
                file += ".pickle"
            with open(file, "wb") as outfile:
                if type(data)=="pandas.core.frame.DataFrame":
                    data.to_pickle(outfile)
                else:
                    pickle.dump(data, outfile)
                print("Data saved to " + file + "!")
                
        elif thismode.upper()=="CSV" or thismode.upper()==".CSV":
            if not file.endswith(".csv"):
                file += ".csv"
            with open(file, "w") as outfile:
                if type(data)=="pandas.core.frame.DataFrame":
                    data.to_csv(outfile,mode="w",index=False) # ,header=data.columns.values
                else:
                    wr = csv.writer(outfile)
                    wr.writerows(data)
                print("Data saved to " + file + "!")

        else:
            print("ERROR! Improper arguments. Please include: data object to save (Pandas DataFrames OK), file path, and file format ('JSON', 'pickle', or 'CSV').")
    
    except Exception as e:
        print("Failed to save to " + str(file) + " into memory using " + str(thismode) + " format. Please check arguments (data, file, file format) and try again.")
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

logging.info("List of keywords:\n" + str(list(all_keywords)))


# ### Create dictionaries for each ideology and one for combined ideologies

ess_dict, prog_dict, rit_dict, all_ideol, all_dicts = set(), set(), set(), set(), set()
all_ideol = load_dict(all_ideol, dicts_dir + "ess_dict.txt")
all_ideol = load_dict(all_ideol, dicts_dir + "prog_dict.txt") # For complete ideological list, append second ideological dict
all_dicts = load_dict(all_ideol, dicts_dir + "rit_dict.txt") # For complete dict list, append ritual dict terms too
ess_dict = load_dict(ess_dict, dicts_dir + "ess_dict.txt")
prog_dict = load_dict(prog_dict, dicts_dir + "prog_dict.txt")
rit_dict = load_dict(rit_dict, dicts_dir + "rit_dict.txt")

logging.info(str(len(all_ideol)) + " entries loaded into the combined ideology dictionary.")
list_dict = list(all_ideol)
list_dict.sort(key = lambda x: x.lower())
logging.info("First 10 elements of combined ideology dictionary are:\n" + str(list_dict[:10]))
    

# Create tuples for keyword lists and dictionary terms:
keys_tuple = tuple([mission_keywords,curriculum_keywords,philosophy_keywords,history_keywords,about_keywords,all_keywords])
dicts_tuple = tuple([ess_dict,prog_dict,rit_dict,all_ideol,all_dicts])
    
logging.info("The contents of keys_tuple:")
logging.info(str(list(keys_tuple)))
logging.info("The contents of dicts_tuple:")
logging.info(str(list(dicts_tuple)))

    
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
            logging.info(stemmed_word)
                
    # Iterate through list of matching word indices and remove the matches
    for i in range(len(hits_indices)-1, -1, -1):
        splitted_phrase = splitted_phrase[:hits_indices[i]] + \
        splitted_phrase[hits_indices[i] + length:]
    modified_phrase = ""
    for sp in splitted_phrase: # Rebuild the modified phrase, with matches removed
        modified_phrase += sp + " "
    return modified_phrase[:-1], counts

                  
@timeout_decorator.timeout(20, use_signals=False)
def dictmatch_file_helper(file, listlists, allmatch_count):
    """Counts number of matches in file for each list of terms given, and also collects the terms not matched.
    listlists is a list of lists, each list containing:
    a list of key terms--e.g., for dictsnames_biglist, currently essentialism, progressivism, ritualism, and all three combined (ess_dict, prog_dict, rit_dict, all_dicts);
    the variables used to store the number of matches for each term lit (e.g., ess_count, prog_count, rit_count, alldict_count); 
    and the not-matches--that is, the list of words leftover from the file after all matches are removed (e.g., ess_dictless, prog_dictless, rit_dictless, alldict_dictless). """         
    
    for i in range(len(dictsnames_biglist)): # Iterate over dicts to find matches with parsed text of file
        # For dictsnames_list, dicts are: (ess_dict, prog_dict, rit_dict, alldict_count); count_names are: (ess_count, prog_count, rit_count, alldict_count); dictless_names are: (ess_dictless, prog_dictless, rit_dictless, alldict_dictless)
        # adict,count_name,dictless_name = dictsnames_tupzip[i]
        dictless_add,count_add = dict_count(parsed_pagetext,listlists[i][0])
        listlists[i][1] += count_add
        listlists[i][2] += dictless_add
        allmatch_count += count_add
        
        logging.info("Discovered " + str(count_add) + " matches for " + str(file) + \
                     ", a total thus far of " + str(allmatch_count) + " matches...")
                  
    return listlists,allmatch_count
                  

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
    # TO DO: Eliminate anything with a '\x' in it (after splitting by punctuation)
    visible_text = list(filter(lambda vt: vt.split() != [], visible_text)) # Eliminate empty elements
    # Consider joining list elements together with newline in between by prepending with: "\n".join

    return(visible_text)


logging.info("Output of parsefile_by_tags:\n" + str(parsefile_by_tags(example_file)))

    
@timeout_decorator.timeout(20, use_signals=False)
def parse_file_helper(file, webtext, keys_tuple, keys_vars, dicts_tuple, dicts_vars):
    """Parses file into (visible) webtext, both as raw webtext and filtered by keywords (held in keys_tuple) into the corresponding list of strings in keys_vars."""
    
    parsed_pagetext = []
    parsed_pagetext = parsefile_by_tags(file) # Parse page text

    if len(parsed_pagetext) == 0: # Don't waste time adding empty pages
        logging.warning("    Nothing to parse in " + str(file) + "!")
        return
    
    webtext.extend(parsed_pagetext) # Add to long list (no filter)
    
    # Filter parsed text and add to appropriate variables using `.extend`:
    for i in range(len(keys_tuple)):
        keys_vars[i].extend(filter_dict_page(parsed_pagetext, keys_tuple[i]))
    for i in range(len(dicts_tuple)):
        dicts_vars[i].extend(filter_dict_page(parsed_pagetext, dicts_tuple[i]))
        
    logging.info("Successfully parsed and filtered file " + str(file) + "...")
    
    return webtext, keys_vars, dicts_vars
                  

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


logging.info("Output of filter_keywords_page with keywords:\n" + str(filter_dict_page(example_textlist, all_keywords)))   
logging.info("Output of filter_keywords_page with ideology words:\n\n" + str(filter_dict_page(example_textlist, all_ideol)))


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
    
    logging.info("Parsing " + str(school_name) + " at " + str(school_address) + " in folder <" + datalocation + str(folder_name) + "/>, which is ROUGHLY #" + str(6*itervar) + " / " + str(numschools) + " schools...")
    
    school_folder = datalocation + folder_name + "/"
    error_file = school_folder + "error_flags.txt" # Define file path for error text log
    counts_file = school_folder + "dict_counts.txt" # File path for dictionary counts output
    
    if school_URL==school_name:
        school_URL = folder_name # Workaround for full_schooldata, which doesn't yet have URLs: use folder name, which is more or less unique
        
    duplicate_flag,parse_error_flag,wget_fail_flag,file_count = 0,0,0,0 # initialize error flags
    
    # PRELIMINARY TEST 1: Check if parsing is already done. If so, no need to parse--stop function!
    if os.path.exists(error_file) and os.path.exists(counts_file):
        logging.info("Parsing output already detected in " + str(school_folder) + ", aborting parser...")
        return
    
    # PRELIMINARY TEST 2: Check if folder exists. If not, nothing to parse. Thus, do not pass go; do not continue function.
    
    if not (os.path.exists(school_folder) or os.path.exists(school_folder.lower()) or os.path.exists(school_folder.upper())):
        logging.warning("NO DIRECTORY FOUND, creating " + str(school_folder) + " for 'error_flags.txt' and aborting...")
        wget_fail_flag = 1
        try:
            os.makedirs(school_folder) # Create empty folder for school to hold error_flags.txt (and nothing else)
            write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
            write_counts(counts_file, ["ess_count","prog_count","rit_count"], [0,0,0]) # empty counts file simplifies parsing
            return
        except Exception as e:
            logging.debug("Uh-oh! Failed to log error flags for " + str(school_name) + ".\n" + str(e))
            return
    
    # PRELIMINARY TEST 3: Check if this school has already been parsed--via its unique school_URL. If so, skip this school to avoid duplication bias.
    if school_URL in parsed: 
        logging.error("DUPLICATE URL DETECTED. Skipping " + str(school_folder) + "...")
        duplicate_flag = 1
        write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
        write_counts(counts_file, ["ess_count","prog_count","rit_count"], [0,0,0]) # empty counts file simplifies parsing
        return
    
    logging.info("Preliminary tests passed. Parsing data in " + str(school_folder) + "...")
    
    # Next, initialize local (within-function) variables for text output
    webtext,keywords_text,ideology_text = [],[],[],[] # text category lists
    file_list = [] # list of HTML files in school_folder
    
    mission_count,curriculum_count,philosophy_count,history_count,about_count,keywords_count,allkey_matches = 0,0,0,0,0,0,0 # matched keyword counts
    mission_dictless,curriculum_dictless,philosophy_dictless,history_dictless,about_dictless,keywords_dictless = [],[],[],[],[],[] # lists of words not matched for each keyword list
    ess_count, prog_count, rit_count, allideol_count, alldict_count, alldict_matches = 0,0,0,0,0,0 # dict match counts
    ess_dictless, prog_dictless, rit_dictless, allideol_dictless, alldict_dictless = [],[],[],[],[] # lists of words not matched for each dictionary
    # Initialize counts and  unmatched lists so later we can revise the dictionaries by looking at what content words were not counted by current dictionaries

    keysnames_list = [mission_count,curriculum_count,philosophy_count,history_count,about_count,keywords_count] # list holding match counts per keywords list
    keylessnames_list = [mission_dictless,curriculum_dictless,philosophy_dictless,history_dictless,about_dictless,keywords_dictless] # list of terms not matched to each list of keywords
    dictsnames_list = [ess_count, prog_count, rit_count, allideol_count, alldict_count] # list holding match counts per dict
    dictlessnames_list = [ess_dictless, prog_dictless, rit_dictless, allideol_dictless, alldict_dictless] # list of terms not matched to each list of dicts

    ''' Reminder/ FYI of what's in these global tuples of lists:
    keys_tuple = tuple([mission_keywords, curriculum_keywords, philosophy_keywords, history_keywords, about_keywords, all_keywords])
    dicts_tuple = tuple([ess_dict,prog_dict,rit_dict,all_ideol,all_dicts])'''
    
    # Now link together dict terms lists with variables holding their matches (and if applicable, their not-matches):
    #keysnames_tupzip = zip(keys_tuple, titles_list) # zips together keyword lists with the variables holding their matches
    #dictsnames_tuplist = zip(dicts_tuple, dictsnames_list, dictlessnames_list)
    keysnames_biglist = [[keys_tuple[i],keysnames_list[i],keylessnames_list[i]] for i in range(len(keys_tuple))] # big list of lists for dict matching with keywords
    dictsnames_biglist = [[dicts_tuple[i],dictsnames_list[i],dictlessnames_list[i]] for i in range(len(dicts_tuple))] # big list of lists for matching with ideology dictionaries

    logging.info(str(list(keysnames_biglist)))
    logging.info(str(list(dictsnames_biglist)))
    
    # Now to parsing:
    try:
        # Parse file only if it contains HTML. This is easy: use the "*.html" wildcard pattern--
        # also wget gave the ".html" file extension to appropriate files when downloading (`--adjust-extension` option)
        # Less efficient ways to check if files contain HTML (e.g., for data not downloaded by wget):
        # if bool(BeautifulSoup(open(fname), bsparser).find())==True: # if file.endswith(".html"):
        # Another way to do this, maybe faster but broken: files_iter = iglob(school_folder + "**/*.html", recursive=True)
            
        file_list = list_files(school_folder, ".html") # Get list of HTML files in school_folder
            
        if file_list==(None or school_folder or "" or []) or not file_list or len(file_list)==0:
            logging.info("No .html files found. Aborting parser for " + str(school_name) + "...")
            wget_fail_flag = 1
            write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
            write_counts(counts_file, ["ess_count","prog_count","rit_count"], [0,0,0]) # empty counts file simplifies parsing
            return
        
        for file in tqdm(file_list, desc=("Parsing files")):
                
            logging.info("Parsing HTML in " + str(file) + "...")
                    
            # Parse and categorize page text:
            try:                    
                # TO DO: Correct these inputs
                keys_vars = []
                dicts_vars = []
                webtext, keys_vars, dicts_vars = parse_file_helper(file, webtext, keys_tuple, keys_vars, dicts_tuple, dicts_vars )
                        
                file_count+=1 # add to count of parsed files

            except Exception as e:
                logging.error("ERROR! Failed to parse file...\n" + str(e))
                        
            # Count dict matches:
            try:
                dictsnames_biglist, dicts_matches = dictmatch_file_helper(file, dictsnames_biglist, alldict_matches)
                keysnames_biglist, keys_matches = dictmatch_file_helper(file, keysnames_biglist, allkey_matches)

            except Exception as e:
                logging.info("ERROR! Failed to count number of dict matches while parsing " + str(file) + "...\n" + str(e))
                    
        # Report and save output to disk:
        parsed.append(school_URL)
        file_count = int(file_count-1)
        print("  PARSED " + str(file_count) + " .html file(s) from website of " + str(school_name) + "...")
            
        write_list(school_folder + "webtext.txt", webtext)
        write_list(school_folder + "keywords_text.txt", keywords_text)
        write_list(school_folder + "ideology_text.txt", ideology_text)
            
        print("  Found " + str(all_matches) + " total dictionary matches and " + str(len(dictsnames_biglist[3][2])) + " uncounted words for " + str(school_name) + "...")

        write_counts(counts_file, ["ess_count","prog_count","rit_count"], [dictsnames_biglist[0][1], dictsnames_biglist[1][1], dictsnames_biglist[2][1]])
        write_list(school_folder + "dictless_words.txt", dictsnames_biglist[3][2])
                    
        write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
        print("Got to end of parsing script")

    except Exception as e:
        logging.error("ERROR! Failed to parse, categorize, and get dict matches on webtext of " + str(school_name) + "...\n" + str(e))
        parse_error_flag = 1
        write_errors(error_file, duplicate_flag, parse_error_flag, wget_fail_flag, file_count)
        write_counts(counts_file, ["ess_count","prog_count","rit_count"], [0,0,0]) # empty counts file simplifies parsing

    return
                  
    
# ### Preparing data to be parsed

itervar = 0 # initialize iterator that counts number of schools already parsed--useless when multiprocessing
parsed = [] # initialize list of URLs that have already been parsed
dicts_list = [] # initialize list of dictionaries to hold school data

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
    # save_datafile(dicts_list, temp_dir+"school_parser_temp", "JSON") # Save output so we can pick up where left off, in case something breaks
    
tuplist_zip = zip(names, addresses, urls, folder_names) # Create list of tuples to pass to parser function


# ### Run parsing algorithm on schools (requires access to webcrawl output)

if Debug:
    test_dicts = dicts_list[:1] # Limit number of schools to test/refine methods
    for school in test_dicts:
        parse_school(school)
    dictfile = "testing_dicts_" + str(datetime.today())
    save_datafile(test_dicts, temp_dir+dictfile, "JSON")
    sys.exit()
                
# Use multiprocessing.Pool(numcpus) to run parse_school(),
# which parses downloaded webtext and saves the results to local storage:
if __name__ == '__main__':
    with Pool(numcpus) as p:
        p.map(parse_school, tqdm(list(tuplist_zip), desc="Parsing HTML into text"), chunksize=numcpus)

# save_datafile(dicts_list, temp_dir+"school_dicts_temp", "CSV") # makes translating into DF (below) easier?
        
print("\nSCHOOL PARSING COMPLETE!!!")