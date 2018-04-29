# import necessary libraries
import os, re # for navigating file trees and working with strings
import json, pickle, csv # For saving a loading dictionaries, DataFrames, lists, etc. in JSON, pickle, and CSV formats
from datetime import datetime # For timestamping files
import time, timeout_decorator # To prevent troublesome files from bottlenecking the parsing process, use timeouts
import sys # For working with user input
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from tqdm import tqdm # For progress information over iterations, including with Pandas operations via "progress_apply"

# Import parser
from bs4 import BeautifulSoup # BS reads and parses even poorly/unreliably coded HTML 
from bs4.element import Comment # helps with detecting inline/junk tags when parsing with BS
import lxml # for fast HTML parsing with BS

def parsefile_by_tags(HTML_file):
    
    """Cleans HTML by removing inline tags, ripping out non-visible tags, 
    replacing paragraph tags with a random string, and finally using this to separate HTML into chunks.
    Reads in HTML from storage using a given filename, HTML_file."""

    random_string = "".join(map(chr, os.urandom(75))) # Create random string for tag delimiter
    soup = BeautifulSoup(open(HTML_file), "lxml")
    
    inline_tags = ["b", "big", "i", "small", "tt", "abbr", "acronym", "cite", "dfn",
               "em", "kbd", "strong", "samp", "var", "bdo", "map", "object", "q",
               "span", "sub", "sup"] # this list helps with eliminating junk tags when parsing HTML

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

