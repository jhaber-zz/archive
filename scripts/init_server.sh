#!/bin/bash

# Simple shell script to install additional packages to a Jetstream VM based on an Ubuntu 16.04 image

# Author: Jaren Haber
# First created: Spring 2018
# Last updated: July 20, 2018

pip install --upgrade pip # Get pip up to date
pip install --upgrade pandas # Get pandas up to date
pip install tqdm # For pretty progress bar etc
pip install nltk # For NLP
pip install timeout-decorator # To limit how long a task can take
pip install beautifulsoup4 # For reading and parsing web text
pip install lxml # More efficient BeautifulSoup parser than HTML5
pip install boxsdk # For Box API
pip install geopandas # For loading shapefiles, doing GIS with matplotlib
pip install shapely # For graphs
pip install gensim # For Word2vec and other NLP things
pip install spacy # For NER, tokenizing, parsing
spacy download en # For spacy in English
pip install seaborn # For beautiful graphs (builds on matplotlib)
pip install Cython # For parallelizing Word2vec etc
pip install openpyxl # For saving DataFrames in Excel format
pip install google # For automated Google searching 
pip install https://github.com/slimkrazy/python-google-places/zipball/master # Google Places API
pip install scrapy #  For web-scraping
pip install pyldavis # For TM visualization

# Write and call function to import NLP tools from within Python:
function import_NLP_tools {
python - <<END
import nltk
nltk.download('punkt')
nltk.download('stopwords')
END
}

import_NLP_tools
