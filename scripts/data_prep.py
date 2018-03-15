#!/usr/bin/env python
# -*- coding: UTF-8

# # Loading local text files into analyzable Python object--Pandas DataFrame or dense list of dictionaries


"""This script loads data from text files stored in a local folder for each school; incorporates these files into a large pandas DataFrame (or list of dictionaries, memory allowing); and then saves this as an analysis-ready CSV file.

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
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
import pandas as pd # modifies data more efficiently than with a list of dicts
from tqdm import tqdm # For progress information over iterations, including with Pandas operations via "progress_apply"


# ### Set script options

Debug = False # Set to "True" for extra progress reports while algorithms run
notebook = False # Use different file paths depending on whether files are being accessed from shell (False) or within a Jupyter notebook (True)
usefile = False # Set to "True" if loading from file a dicts_list to add to. Confirms with user input first!
workstation = False # If working from office PC

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
    
data_year = int(2014)
    
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
    """BROKEN for saving to CSV Pandas DataFrames (only saves header) and lists of dicts (only saves keys).
    Saves data to file using JSON, pickle, or CSV format (whichever was specified).
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
            
            if type(data)=="pandas.core.frame.DataFrame":
                data.to_json(file)
            
            else:
                with open(file, 'w') as outfile:
                    json.dump(data, outfile, encoding="utf-8")
            
            #print("Data saved to " + file + "!")

        elif thismode.lower()=="pickle" or thismode.lower()==".pickle":
            if not file.endswith(".pickle"):
                file += ".pickle"
                
            if type(data)=="pandas.core.frame.DataFrame":
                data.to_pickle(file, encoding="utf-8")
                
            else:
                with open(file, "wb") as outfile:
                    pickle.dump(data, outfile, encoding="utf-8")
                    
            #print("Data saved to " + file + "!")
                
        elif thismode.upper()=="CSV" or thismode.upper()==".CSV":
            if not file.endswith(".csv"):
                file += ".csv"
                
            if type(data)=="pandas.core.frame.DataFrame":
                if os.path.exists(file): # If file already exists, assume we are appending to it (with same column names)
                    data.to_csv(file,mode="a",index=False,sep="\t",header=False,encoding="utf-8")
                else: # If file doesn't exist, create it
                    data.to_csv(file,mode="w",index=False,sep="\t",header=data.columns.values,encoding="utf-8")
                
            else:
                with open(file, "w") as outfile:
                    wr = csv.writer(outfile)
                    wr.writerows(data)
                
            #print("Data saved to " + file + "!")

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


def convert_df(df):
    """Makes a Pandas DataFrame more memory-efficient through intelligent use of Pandas data types: 
    specifically, by storing columns with repetitive Python strings not with the object dtype for unique values 
    (entirely stored in memory) but as categoricals, which are represented by repeated integer values. This is a 
    net gain in memory when the reduced memory size of the category type outweighs the added memory cost of storing 
    one more thing. As such, this function checks the degree of redundancy for a given column before converting it."""
    
    converted_df = pd.DataFrame() # Initialize DF for memory-efficient storage of strings (object types)
    # TO DO: Infer dtypes of df
    df_obj = df.select_dtypes(include=['object']).copy() # Filter to only those columns of object data type

    for col in df.columns: 
        if col in df_obj: 
            num_unique_values = len(df_obj[col].unique())
            num_total_values = len(df_obj[col])
            if (num_unique_values / num_total_values) < 0.5: # Only convert data types if at least half of values are duplicates
                converted_df.loc[:,col] = df[col].astype('category') # Store these columns as dtype "category"
            else: 
                converted_df.loc[:,col] = df[col]
        else:    
            converted_df.loc[:,col] = df[col]
                      
    converted_df.select_dtypes(include=['float']).apply(pd.to_numeric,downcast='float')
    converted_df.select_dtypes(include=['int']).apply(pd.to_numeric,downcast='signed')
    
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
keys_tuple = tuple([mission_keywords,curriculum_keywords,philosophy_keywords,history_keywords,about_keywords,\
                        all_ideol,all_keywords])
dicts_tuple = tuple([ess_dict,prog_dict,rit_dict,all_dicts])
    
logging.info(str(list(keys_tuple)))
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
def dictmatch_file_helper(file,dictsnames_biglist,all_keywords,all_ideol,all_matches):
    """Counts number of matches in file for each list of terms given, and also collects the terms not matched.
    Dictsnames_biglist is a list of lists, each list containing:
    a list of key terms, currently essentialism, progressivism, ritualism, and all three combined (ess_dict, prog_dict, rit_dict, all_dicts);
    the variables used to store the number of matches for each term lit (ess_count, prog_count, rit_count, alldict_count);
    and the not-matches--that is, the list of words leftover from the file after all matches are removed (ess_dictless, prog_dictless, rit_dictless, alldict_dictless). """
    
                  
    for i in range(len(dictsnames_biglist)): # Iterate over dicts to find matches with parsed text of file
        # Dicts are: (ess_dict, prog_dict, rit_dict, alldict_count); count_names are: (ess_count, prog_count, rit_count, alldict_count); dictless_names are: (ess_dictless, prog_dictless, rit_dictless, alldict_dictless)
        # adict,count_name,dictless_name = dictsnames_tupzip[i]
        dictless_add,count_add = dict_count(parsed_pagetext,dictsnames_biglist[i][0])
        dictsnames_biglist[i][1] += count_add
        dictsnames_biglist[i][2] += dictless_add
        all_matches += count_add
                    
        logging.info("Discovered " + str(count_add) + " matches for " + str(file) + ", a total thus far of " + str(dictsnames_biglist[i][1]) + " matches...")
                  
    return dictsnames_biglist,all_matches
                  

# ### Define parsing helper functions
    
@timeout_decorator.timeout(20, use_signals=False)
def parse_file_helper(file,webtext,keywords_text,ideology_text):
    """Parses file into (visible) webtext, both complete and filtered by terms in 'keywords' and 'ideology' lists."""
    
    parsed_pagetext = []
    parsed_pagetext = parsefile_by_tags(file) # Parse page text

    if len(parsed_pagetext) == 0: # Don't waste time adding empty pages
        logging.warning("    Nothing to parse in " + str(file) + "!")
    
    else:
        webtext.extend(parsed_pagetext) # Add new parsed text to long list
        keywords_text.extend(filter_dict_page(parsed_pagetext, all_keywords)) # Filter using keywords
        ideology_text.extend(filter_dict_page(parsed_pagetext, all_ideol)) # Filter using ideology words

        logging.info("Successfully parsed and filtered file " + str(file) + "...")
        
    return webtext,keywords_text,ideology_text
                  

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

#logging.info("Output of filter_keywords_page with keywords:\n" + str(filter_dict_page(example_textlist, all_keywords)))   
#logging.info("Output of filter_keywords_page with ideology words:\n\n" + str(filter_dict_page(example_textlist, all_ideol)))
                  

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
    
    folder_name = re.sub(" ","_",(school_dict[NAME_var]+" "+school_dict[ADDR_var][-8:-6])) # This gives name and state separated by "_"
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
    global numschools,wget_dataloc,save_dir,NAME_var,ADDR_var,URL_var
    datalocation = wget_dataloc # Define path to local data storage
    
    #logging.info("Loading into DataFrame parsing output for " + str(len(df)) + " school websites out of a total of " + str(numschools) + "...")
    
    # Initialize text strings and counts as empty, then convert data types:
    empty = ["" for elem in range(len(df["NCESSCH"]))] # Create empty string column length of longest variable (NCESCCH used for matching)
    df = df.assign(FOLDER_NAME=empty, TOTETH=empty, PCTETH=empty, AGE=empty, PCTFRL=empty, PLACE=empty, WEBTEXT=empty, KEYWORDS_TEXT=empty, IDEOLOGY_TEXT=empty, ESS_COUNT=empty, PROG_COUNT=empty, RIT_COUNT=empty, ESS_STRENGTH=empty, PROG_STRENGTH=empty) # Add empty columns to df
    df.loc[:,["PLACE", "WEBTEXT", "KEYWORDS_TEXT", "IDEOLOGY_TEXT", "FOLDER_NAME"]] = df.loc[:,["PLACE", "WEBTEXT", "KEYWORDS_TEXT", "IDEOLOGY_TEXT", "FOLDER_NAME"]].apply(lambda x: x.astype(object)) # Convert to object type--holds text
    df.loc[:,["AGE", "PCTFRL", "TOTETH", "PCTETH", "ESS_COUNT", "PROG_COUNT", "RIT_COUNT", "ESS_STRENGTH", "PROG_STRENGTH"]] = df.loc[:,["AGE", "PCTFRL", "TOTETH", "PCTETH", "ESS_COUNT", "PROG_COUNT", "RIT_COUNT", "ESS_STRENGTH", "PROG_STRENGTH"]].apply(pd.to_numeric, downcast="unsigned") # Convert to int dtype--holds positive numbers (no decimals)
    
    df.loc[:,"FOLDER_NAME"] = df.loc[:,[NAME_var,ADDR_var]].apply(lambda x: re.sub(" ","_","{} {}".format(str(x[0]),str(x[1][-8:-6]))), axis=1) # This gives name and state separated by "_"
    df.loc[:,"school_folder"] = df.loc[:,"FOLDER_NAME"].apply(lambda x: str(datalocation) + '{}/'.format(str(x)))
    df.loc[:,"error_file"] = df.loc[:,"school_folder"].apply(lambda x: '{}error_flags.txt'.format(str(x))) # Define file path for error text log
    df.loc[:,"counts_file"] = df.loc[:,"school_folder"].apply(lambda x: '{}dict_counts.txt'.format(str(x)))
    
    try:
        # Compute demographic variables:
        df["TOTETH"] = df[["AM", "AS", "BL", "HI", "HP", "TR"]].apply(sum, axis=1) # Number of nonwhite K-12 students
        df["PCTETH"] = df["TOTETH"]/df["MEMBER"] # Percent nonwhite K-12 students
        df["PCTFRL"] = df["TOTFRL"]/df["MEMBER"] # Percent receiving free/ reduced-price lunch
        df["AGE"] = data_year - df["YEAR_OPENED"] # Number of years school has been open
        
        # Recode variables:
        df["PLACE"] = df["LOCALE"].map({11.0:"City", 12.0:"City", 13.0:"City", 21.0:"Suburb", 22.0:"Suburb", 23.0:"Suburb", 31.0:"Town", 32.0:"Town", 33.0:"Town", 41.0:"Rural", 42.0:"Rural", 43.0:"Rural"}).astype('category')
        df["LOCALE"] = df["LOCALE"].map({11.0:"City (large)", 12.0:"City (midsize)", 13.0:"City (small)", 21.0:"Suburb (large)", 22.0:"Suburb (midsize)", 23.0:"Suburb (small)", 31.0:"Town (fringe)", 32.0:"Town (distant)", 33.0:"Town (remote)", 41.0:"Rural (fringe)", 42.0:"Rural (distant)", 43.0:"Rural (remote)"}).astype('category')
        df["TITLEI"] = df["TITLEI"].map({"Yes":1, "No":0}).astype(category)
        
        # load error_file as a list with four pieces, the last element of each of which is the flag value itself:
        df.loc[:,"error_text"] = df.loc[:,"error_file"].apply(lambda x: load_list('{}'.format(str(x))))
        df.loc[:,"duplicate_flag"] = df.loc[:,"error_text"].apply(lambda x: '{}'.format(str(x[0].split()[-1]))) #  # last element of first piece of error_text
        df.loc[:,"parse_error_flag"] = df.loc[:,"error_text"].apply(lambda x: '{}'.format(str(x[1].split()[-1]))) 
        df.loc[:,"wget_fail_flag"] = df.loc[:,"error_text"].apply(lambda x: '{}'.format(str(x[2].split()[-1]))) 
        df.loc[:,"html_file_count"] = df.loc[:,"error_text"].apply(lambda x: '{}'.format(str(x[3].split()[-1]))) 
        
        downloaded = df["wget_fail_flag"].map({"1":True,1:True,"0":False,0:False}) == False # This binary conditional filters df to only those rows with downloaded web content--where wget_fail_flag==False and thus does NOT signal download failure
        
        logging.info("Loading webtext from disk into DF...")
        
        # Load school parse output from disk into DataFrame:
        #df.loc[downloaded,"WEBTEXT"] = df.loc[downloaded,"school_folder"].apply(lambda x: load_list("{}webtext.txt".format(str(x)))) # df["wget_fail_flag"]==False
        #df.loc[downloaded,"KEYWORDS_TEXT"] = df.loc[downloaded,"school_folder"].apply(lambda x: load_list("{}keywords_text.txt".format(str(x))))
        df.loc[downloaded,"IDEOLOGY_TEXT"] = df.loc[downloaded,"school_folder"].apply(lambda x: load_list("{}ideology_text.txt".format(str(x))))
        
        df["counts_text"] = df.counts_file.apply(lambda x: load_list("{}".format(str(x))))
        df.loc[downloaded,"ESS_COUNT"] = df.loc[downloaded,"counts_text"].apply(lambda x: "{}".format(str(x[0].split()[-1]))).apply(pd.to_numeric,downcast='unsigned') # 2nd element of 1st row in counts_text: take as uint dtype (no negatives)
        df.loc[downloaded,"PROG_COUNT"] = df.loc[downloaded,"counts_text"].apply(lambda x: "{}".format(str(x[1].split()[-1]))).apply(pd.to_numeric,downcast='unsigned') # 2nd element of 2nd row
        df.loc[downloaded,"RIT_COUNT"] = df.loc[downloaded,"counts_text"].apply(lambda x: "{}".format(str(x[2].split()[-1]))).apply(pd.to_numeric,downcast='unsigned') # 2nd element of 3nd row
        df.loc[downloaded,"ESS_STRENGTH"] = (df.loc[downloaded,"ESS_COUNT"]/df.loc[downloaded, "RIT_COUNT"]).apply(pd.to_numeric, downcast='float') # calculate ideology ratio, use most memory-efficient float dtype
        df.loc[downloaded,"PROG_STRENGTH"] = (df.loc[downloaded,"PROG_COUNT"]/df.loc[downloaded, "RIT_COUNT"]).apply(pd.to_numeric, downcast='float') 
        #logging.info(str(df.loc[downloaded,'prog_strength']))
        
        df = df.drop(["school_folder","error_text","error_file","counts_text", "AM", "AS", "BL", "HI", "HP"],axis=1) # Clean up temp variables
        
        logging.info("LOADED " + df["html_file_count"].sum() + " .html files into DataFrame!")
        #save_datafile(df, save_dir+"df_parser_temp", "pickle") # Save output so we can pick up where left off, in case something breaks before able to save final output
        
        return df
    
    except Exception as e:
        logging.critical("ERROR! Pandify function failed to load parsing output into DataFrame.\n" + str(e))
        print("    ERROR! Pandify function failed to load parsing output into DataFrame.")
        print("  ",str(e))
        sys.exit()
    

def slice_pandify(bigdf_iter, numsplits, df_filepath):
    """This function uses pandify_webtext() to load the parsing output from local storage into a DataFrame.
    It gets around system memory limitations--which otherwise lead terminal to kill any attempts to pandify() all of bigdf--
    by splitting bigdf into numsplits smaller dfslices, parsing webtext into each slice, and recombining them
    by appending them to a big CSV on file. 
    The number of slices equals numsplits, and bigdf is split by numschools/ numsplits."""
    
    global numschools # Access numschools from within function (this is roughly 7000)
    wheresplit = int(round(float(numschools)/float(numsplits))) # Get number on which to split (e.g., 1000) based on total number of schools data. This splitting number will be used to iterate over numsplits
    
    for num in tqdm(range(numsplits), desc="Loading " + str(numsplits) + " DF slices"): # Wrap iterator with tqdm to show progress bar
        try:
            dfslice = pd.DataFrame()
            dfslice = bigdf_iter.get_chunk(wheresplit) # Get next chunk of rows 
            dfslice = dfslice[dfslice.ADDRESS14 != 'ADDRESS14'] # Clean out any cases of header being written as row
            dfslice = convert_df(dfslice) # Make this slice memory-efficient by appropriately converting column dtypes

            startnum, endnum = wheresplit*int(num),wheresplit*int(num+1)
            #dfslice = bigdf_iter.iloc[startnum:endnum,:]
            #print("Loading DF parsing output for slice #" + str(num) + " of " + str(numschools) + " school websites, from #" + str(startnum) + "-" + str(endnum) + "...")
            logging.info("Loading parsing output for slice #" + str(num) + " of " + str(numschools) + " school websites, from #" + str(startnum) + "-" + str(endnum) + "...")
            
            '''if num==0: # Save first slice to new file (overwriting if needed)
                dfslice = pandify_webtext(dfslice) # Load parsed output into the DF
                logging.info("Slice #" + str(num) + " loaded! Saving file...")
                dfslice.to_csv(df_filepath, mode="w", index=False, header=dfslice.columns.values, sep="\t", encoding="utf-8")
                print("Slice #" + str(num) + " saved to " + df_filepath + "!")
                logging.info("Slice #" + str(num) + " saved to " + df_filepath + "!")
            if num<20:
                pass
            elif num==20:
                dfslice = pandify_webtext(dfslice) # Load parsed output into the DF
                logging.info(dfslice[["SCH_NAME", "FOLDER_NAME", "html_file_count"]])
                print(dfslice[["SCH_NAME", "FOLDER_NAME", "html_file_count"]])
                sys.exit()
            else:
                dfslice = pandify_webtext(dfslice) # Load parsed output into the DF
                logging.info("Slice loaded! Saving file...")
                dfslice.to_csv(df_filepath, mode="a", index=False, header=False, sep="\t", encoding="utf-8")
                print("Slice #" + str(num) + " saved to " + df_filepath + "!")
                logging.info("Slice #" + str(num) + " saved to " + df_filepath + "!")'''
                
            dfslice = pandify_webtext(dfslice) # Load parsed output into the DF
            logging.info(dfslice[["FOLDER_NAME", "html_file_count"]])
            logging.info("Slice #" + str(num) + " loaded! Saving file...")
            
            if num==0: # Save first slice to new file (overwriting if needed)
                dfslice.to_csv(df_filepath, mode="w", index=False, header=dfslice.columns.values, sep="\t", encoding="utf-8")
            #elif num==1:
            #    sys.exit()
            elif num==(284 or 441 or 593 or 594 or 595 or 596 or 1159 or 1218 or 1219 or 1271 or 1297 or 1303 or 1667 or 1861 or 3361 or 4467 or 4836 or 4871 or 4910 or 5418): # or num==441 or num==593: # Skip Primavera_-_Online_AZ', which is slice #284 if numsplits = 6752
                continue # Move on to next slice
            # TO DO: Clean out excess HTML (e.g., blog posts) in wget downloads for these schools
            else: # Append next slice to existing file
                dfslice.to_csv(df_filepath, mode="a", index=False, header=False, sep="\t", encoding="utf-8")
            #save_datafile(dfslice,df_filepath,"CSV") # BROKEN function--Save slice to file--should work whether writing new file or appending to CSV
            
            #print("Slice #" + str(num) + " saved to " + df_filepath + "!")
            logging.info("Slice #" + str(num) + " saved to " + df_filepath + "!")
            del dfslice # Free memory by deleting this temporary, smaller slice
            
        except Exception as e:
            logging.critical("\nERROR! Script failed to load parsing output into DataFrame slice #" + str(num) + " of " + str(numsplits) + ", for schools #" + str(startnum) + "-" + str(endnum) + ".\n" + str(e))
            print("  ERROR! Script failed to load parsing output into DataFrame slice #" + str(num) + " of " + str(numsplits) + ", for schools #" + str(startnum) + "-" + str(endnum) + ".", str(e))
            #sys.exit()
            continue
            
    return
            

# ### Load parsing output from disk into analyzable object (Pandas DataFrame or list of dicts)

data_loc = full_schooldata # assume we're running on full charter population
URL_var,NAME_var,ADDR_var = get_vars(data_loc) # get varnames depending on data source

"""# Use dictify_webtext to load the parsing output from local storage into the list of dictionaries:

itervar = 0 # initialize iterator that counts number of schools already parsed--useless when multiprocessing
parsed = [] # initialize list of URLs that have already been parsed
dicts_list = [] # initialize list of dictionaries to hold school data

# If input_file was defined by user input in beginning of script, use that to load list of dictionaries. We'll add to it!
if usefile and not dicts_list:
    dicts_list = load_datafile(input_file)    
# data_loc = micro_sample13 # This seems nice for debugging--except directories don't match because different data source
    
# Create dict list from CSV on file, with one dict per school
with open(data_loc, 'r', encoding = 'Latin1') as csvfile: # open data file
    reader = csv.DictReader(csvfile) # create a reader
    for row in reader: # loop through rows
        dicts_list.append(row) # append each row to the list

numschools = int(len(dicts_list)) # Count number of schools in list of dictionaries

for school in dicts_list:
    try:
        school = dictify_webtext(school)
    except Exception as e:
        print("  ERROR! Failed to load into dict parsing output for " + school[NAME_var])
        print("  ",e)
        school_dict["parse_error_flag"] = 1
        continue
    
    save_datafile(dicts_list, temp_dir+"school_parser_temp", "JSON") # Save output so we can pick up where left off, in case something breaks
    
del dicts_list # Free memory"""

        
# Create DF from dicts_list or from file in which to store the data:
#schooldf = pd.DataFrame() # initialize DataFrame to hold school data
#schooldf = pd.DataFrame(dicts_list) # Convert dicts_list into a DataFrame
#schooldf = pd.read_csv(temp_dir+"school_dicts_temp.csv") # Use existing file while debugging pandify_webtext()
#schooldf = schooldf[schooldf.ADDRESS14 != 'ADDRESS14'] # Clean out any cases of header being written as row
#schooldf = convert_df(schooldf) # Make this DF memory-efficient by converting appropriate columns to category data type

with open(data_loc, "r"): # Limits memory drain
    numschools = int(len(pd.read_csv(data_loc, encoding = "Latin1", sep="\t"))) # Count number of schools in file
splits = numschools # Number of times to slice up the big CSV

schooldf_iter = pd.read_csv(data_loc, encoding = "Latin1", low_memory=False, iterator=True, chunksize=splits, na_values={"TITLEI":["M","N"]}) # Create DF from source file

tqdm.pandas(desc="Loading webtext->DF") # To show progress, create & register new `tqdm` instance with `pandas`

# Load parsing output into big pandas DataFrame through slices (to work with limited system memory):
merged_df_file = temp_dir+"mergedf_"+str(datetime.today().strftime("%Y-%m-%d"))+".csv" # Prepare file name
slice_pandify(schooldf_iter, splits, merged_df_file)
print("Larger DF successfully split into " + str(splits) + " smaller DFs, parsed, combined, and saved to file!")

'''if schooldf is not None:
    del schooldf # Free memory
else:
    pass'''
    
# Save final output:
print("\nSCHOOL PARSING COMPLETE!!!")
#schooldf = pd.read_csv(merged_df_file, sep="\t", header=0, low_memory=False, encoding="utf-8") # Load full DF so we can save it in analysis-ready format    #,header=198
#schooldf = schooldf[schooldf.ADDRESS14 != 'ADDRESS14'] # Clean out any bad rows--where header is written as row
#newfile = "charters_parsed_" + str(datetime.today().strftime("%Y-%m-%d"))
#save_datafile(schooldf, save_dir+newfile, "csv")