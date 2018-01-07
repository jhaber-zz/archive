#!/usr/bin/env python
# -*- coding: UTF-8

# # Parsing & Categorizing HTML from `wget` run!

# ## Initializing

# import necessary libraries
import os, re, fnmatch # for navigating file trees and working with strings
import csv # for reading in CSV files
#from glob import glob,iglob # for finding files within nested folders--compare with os.walk
import json, pickle # For saving a loading dictionaries, etc. from file with JSON and pickle formats
from datetime import datetime # For timestamping files
import sys # For working with user input
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words
stemmer = PorterStemmer()
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
import urllib, urllib.request # for testing pages
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format

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
    micro_sample13 = dir_prefix + "data\\micro-sample13_coded.csv" #data location for random micro-sample of 300 US charter schools
    URL_schooldata = dir_prefix + "data\\charter_URLs_2014.csv" #data location for 2014 population of US charter schools
    full_schooldata = dir_prefix + "data\\charter_merged_2014.csv"
    example_file = dir_prefix + "data\\example_file.html" #example_folder + "21stcenturypa.com/wp/default?page_id=27.tmp.html"
    dicts_dir = dir_prefix + "dicts\\" # Directory in which to find & save dictionary files
    save_dir = dir_prefix + "data\\" # Directory in which to save data files

else:
    wget_dataloc = dir_prefix + "wget/parll_wget/" #data location for schools downloaded with wget in parallel (requires server access)
    example_folder = wget_dataloc + "TWENTY-FIRST_CENTURY_NM/"
    example_file = dir_prefix + "wget/example_file.html" #example_folder + "21stcenturypa.com/wp/default?page_id=27.tmp.html"

    micro_sample13 = dir_prefix + "Charter-school-identities/data/micro-sample13_coded.csv" #data location for random micro-sample of 300 US charter schools
    URL_schooldata = dir_prefix + "Charter-school-identities/data/charter_URLs_2014.csv" #data location for 2014 population of US charter schools
    full_schooldata = dir_prefix + "Charter-school-identities/data/charter_merged_2014.csv"
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
            print("Invalid file path. Aborting script.")
            sys.exit()

    elif answer == "N":
        print("OK! This script will create a new file for this list of dictionaries.")
        usefile = False
    
    else:
        print("Response not interpretable. Aborting script.")
        sys.exit()


# ### Define (non-parsing) helper functions

def get_vars(data):
    """Defines variable names based on the data source called."""
    
    if data==URL_schooldata:
        URL_variable = "TRUE_URL"
        NAME_variable = "SCH_NAME"
        ADDR_variable = "ADDRESS"
        
    elif data==full_schooldata:
        URL_variable = "SCH_NAME" # Stand-in until URLs merged into full data file
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
            print(e)
            print("ERROR: No data source established!\n")
    
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


def save_to_file(dicts_list, file, mode):
    """Saves dicts_list to file using JSON or pickle format (whichever was specified)."""
    
    file = str(file)
    mode = str(mode)
    
    try:
        if mode.upper()=="JSON":
            if not file.endswith(".json"):
                file += ".json"
            with open(file, 'w') as outfile:
                json.dump(dicts_list, outfile)
                print("Dicts saved to " + file + " in JSON format!\n")

        elif mode.lower()=="pickle":
            if not file.endswith(".pickle"):
                file += ".pickle"
            with open(file, 'wb') as outfile:
                pickle.dump(dicts_list, outfile)
                print("Dicts saved to " + file + " in pickle format!\n")

        else:
            print("ERROR! Save failed due to improper arguments. These are: file, object to be saved, and file format to save in.\n                  Specify either 'JSON' or 'pickle' as third argument ('mode' or file format) when calling this function.")
    
    except Exception as e:
        print(e)
    

def load_file(file):
    """Loads dicts_list (or whatever) from file, using either JSON or pickle format. 
    The created object should be assigned when called."""
    
    file = str(file)
    
    if file.lower().endswith(".json"):
        with open(file,'r') as infile:
            var = json.load(infile)
    
    if file.lower().endswith(".pickle"):
        with open(file,'rb') as infile:
            var = pickle.load(infile)
        
    print(file + " successfully loaded!\n")
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
keys_dict = set(stemmer.stem(key) for key in keywords)
    
if Debug:
    print("\nList of keywords:\n", list(keys_dict))


# To use with filtering, create combined dictionary for ideologies:

ideol_dict = set()
ideol_dict = load_dict(ideol_dict, dicts_dir + "ess_dict.txt")
ideol_dict = load_dict(ideol_dict, dicts_dir + "prog_dict.txt")

if Debug:
    print(len(ideol_dict), "entries loaded into the combined ideology dictionary.")
    list_dict = list(ideol_dict)
    list_dict.sort(key = lambda x: x.lower())
    print("First 10 elements of combined ideology dictionary are:\n", list_dict[:10])


# ### Compare parsing by newlines vs. by HTML tags

def parseurl_by_newlines(urlstring):
    """Uses BS to parse HTML from a given URL and looks for three newlines to separate chunks of text."""
    
    # Read HTML from a given url:
    with urllib.request.urlopen(urlstring) as url:
        s = url.read()
    
    # Parse raw text from website body:
    soup = BeautifulSoup(s, bsparser)
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    webtext = u" ".join(t.strip() for t in visible_texts)
    
    return re.split(r'\s{3,}', webtext)


def parseurl_by_tags(urlstring):
    """Cleans HTML by removing inline tags, ripping out non-visible tags, 
    replacing paragraph tags with a random string, and finally using this to separate HTML into chunks.
    Reads in HTML from the web using a given website address, urlstring."""
    
    with urllib.request.urlopen(urlstring) as url:
        HTML_page = url.read()

    random_string = "".join(map(chr, os.urandom(75))) # Create random string for tag delimiter
    soup = BeautifulSoup(HTML_page, bsparser)
    
    [s.extract() for s in soup(['style', 'script', 'head', 'title', 'meta', '[document]'])] # Remove non-visible tags
    for it in inline_tags:
        [s.extract() for s in soup("</" + it + ">")] # Remove inline tags
    
    visible_text = soup.getText(random_string).replace("\n", "") # Replace "p" tags with random string, eliminate newlines
    visible_text = list(elem.replace("\t","").replace(u'\xa0', u' ') for elem in visible_text.split(random_string)) # Split text into list using random string while eliminating tabs and unicode; OR: normalize("NFKC", elem) 
    visible_text = list(filter(lambda vt: vt.split() != [], visible_text)) # Eliminate empty elements
    # Consider joining list elements together with newline in between by prepending with: "\n".join
    
    return(visible_text)


# Text chunking accuracy of parsing by tags is superior to parsing by newlines:
# Compare each of these with the browser-displayed content of example_page:
if Debug:
    print(parseurl_by_newlines(example_page),"\n\n",parseurl_by_tags(example_page))
    

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
    print("Output of filter_keywords_page with keywords:\n\n", filter_dict_page(example_textlist, keys_dict), "\n\n")    
    print("Output of filter_keywords_page with ideology words:\n\n", filter_dict_page(example_textlist, ideol_dict), "\n\n")


def parse_school(school_dict, school_name, school_address, school_URL, datalocation, parsed, numschools):
    
    """This core function parses webtext for a given school, using helper functions to run analyses and then saving multiple outputs to school_dict:
    full (partially cleaned) webtext, by parsing webtext of each .html file (removing inline tags, etc.) within school's folder, via parsefile_by_tags();
    and all text associated with specific categories by filtering webtext to those with elements from a defined keyword list, via filter_keywords_page().
    
    For the sake of parsimony and manageable script calls, OTHER similar functions/scripts collect these additional outputs: 
    parsed webtext, having removed overlapping headers/footers common to multiple pages, via remove_overlaps();
    all text associated with specific categories by filtering webtext according to keywords for 
    mission, curriculum, philosophy, history, and about/general self-description, via categorize_page(); and
    contents of those individual pages best matching each of these categories, via find_best_categories."""
    
    global itervar # This allows function to access global itervar counter
    itervar+=1
    
    print("Parsing " + str(school_name) + ", which is school #" + str(itervar) + " of " + str(numschools) + "...")
    
    school_dict["webtext"], school_dict["keywords_text"], school_dict["ideology_text"], school_dict["duplicate_flag"], school_dict["parse_error_flag"], school_dict["wget_fail_flag"] = [], [], [], 0, 0, 0
    
    folder_name = re.sub(" ","_",(school_name+" "+school_address[-8:-6]))
    school_dict["folder_name"] = folder_name
    
    school_folder = datalocation + folder_name + "/"
    if school_URL==school_name:
        school_URL = folder_name # Workaround for full_schooldata, which doesn't yet have URLs

    # Check if folder exists. If not, exit function
    if not (os.path.exists(school_folder) or os.path.exists(school_folder.lower()) or os.path.exists(school_folder.upper())):
        print("!! NO DIRECTORY FOUND matching " + str(school_folder) + ".\n  Aborting parsing function...\n\n")
        school_dict['wget_fail_flag'] = 1
        return
    
    if school_URL not in parsed: #check if this URL has already been parsed. If so, skip this school to avoid duplication bias
        parsed.append(school_URL)
        
        try:
            file_count,school_dict["html_file_count"] = 0,0 # initialize count of files parsed
            
            # Parse file only if it contains HTML. This is easy: use the "*.html" wildcard pattern--
            # also wget gave the ".html" file extension to appropriate files when downloading (`--adjust-extension` option)
            # Less efficient ways to check if files contain HTML (e.g., for data not downloaded by wget):
            # if bool(BeautifulSoup(open(fname), bsparser).find())==True: # if file.endswith(".html"):
            # Another way to do this, maybe faster but broken: files_iter = iglob(school_folder + "**/*.html", recursive=True)
            
            file_list = list_files(school_folder, ".html")
            
            if file_list==(None or school_folder) or not file_list:
                print("ERROR! File gathering function broken!\n  Aborting parser for " + str(school_name) + "...")
                return
            
            elif file_list==("" or []):
                print("  No .html files found.\n  Aborting parser for " + str(school_name) + "...")
                return
            
            for file in file_list:
                                    
                file_count+=1 # add to count of parsed files
                if Debug:
                    print("    Parsing HTML in " + str(file) + "...")
                    
                try:                    
                    parsed_pagetext = parsefile_by_tags(file) # Parse page text (filter too?)
                        
                    school_dict["webtext"].extend(parsed_pagetext) # Add new parsed text to long list

                    #school_dict["keywords_text"].extend(filter_dict_page(parsed_pagetext, keys_dict)) # Filter parsed file using keywords list
                    #school_dict["ideology_text"].extend(filter_dict_page(parsed_pagetext, ideol_dict)) # Filter parsed file using keywords list

                    if Debug:
                        print("      Successfully parsed and filtered file " + str(file) + "...")
                        
                    file_count+=1
                        
                    continue

                except Exception as e:
                    if Debug:
                        print("      ERROR! Failed to parse file...")
                        print("      ",e)
                        continue
                    else:
                        continue
            
            print("  Parsed page text for " + str(file_count-1) + " .html file(s) belonging to " + str(school_name) + "...")
            school_dict["html_file_count"] = int(file_count-1)
            
            print("  SUCCESS! Parsed and categorized website text for " + str(school_name) + "...\n")
            return

        except Exception as e:
            print("    ERROR! Failed to parse & categorize webtext of " + str(school_name))
            print("    ",e)
            school_dict["parse_error_flag"] = 1
    
    else:
        print("DUPLICATE URL DETECTED. Skipping " + str(school_name) + "...\n\n")
        school_dict["duplicate_flag"] = 1
        return


# ### Preparing data to be parsed

itervar = 0 # initialize iterator that counts number of schools already parsed
parsed = [] # initialize list of URLs that have already been parsed
dicts_list = [] # initialize list of dictionaries to hold school data

# If input_file was defined by user input in beginning of script, use that to load list of dictionaries. We'll add to it!
if usefile and not dicts_list:
    dicts_list = load_file(input_file)
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
        
# Note on data structures: each row, dicts_list[i] is a dictionary with keys as column name and value as info.
# This will be translated into pandas data frame once (rather messy) website text is parsed into consistent variables


# ### Run parsing algorithm on schools (requires access to webcrawl output)

test_dicts = dicts_list[:1] # Limit number of schools to analyze, in order to refine methods

if Debug:
    for school in test_dicts:
        parse_school(school, school[NAME_var], school[ADDR_var], school[URL_var], wget_dataloc, parsed, len(dicts_list))
        
else:
    for school in dicts_list:
        parse_school(school, school[NAME_var], school[ADDR_var], school[URL_var], wget_dataloc, parsed, len(dicts_list))
        save_to_file(dicts_list, save_dir+"school_dicts_temp", "JSON") # Save output so we can pick up where left off, in case something breaks before able to save final output


# Save final output:
if Debug:
    dictfile = "testing_dicts_" + str(datetime.today())
    save_to_file(test_dicts, save_dir+dictfile, "JSON")
else:
    dictfile = "school_dicts_" + str(datetime.today())
    save_to_file(dicts_list, save_dir+dictfile, "JSON")

