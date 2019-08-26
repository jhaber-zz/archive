[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/jhaber-zz/Charter-school-identities/master)
# OBSOLETE (see my other repos)
# Stratification through organizational identity:
## Do charter schools' ideologies reinforce social inequalities?

### Generally useful scripts (in "scripts" folder):
#### Downloading and parsing 
- `scraping_URLs.py`: Given search terms, scrapes the best URL using a combination of Google Places API (you'll need an API key from Google) and automated Google searching (Thanks Mario Vilas!).
- `wget_parallel.py`: Highly customized application of GNU software `parallel` and `wget` to efficiently download to disk the static web contents (no JavaScript) of all URLs in a given CSV.
- `webparser_mp.py`: Uses `BeautifulSoup` and `multiprocessing.Pool` to clean, filter, and merge webtext into various analysis-ready formats. Also uses custom dictionaries to count the number of matches for a given school/ organization/ website. 
- `data_prep.py`: Loads text from files on disk into `Pandas` DataFrame and saves as CSV. Customized to work with large amounts of data and/or computational settings with limited system memory. 
#### Text analysis
- `analysis_prelim.ipynb`: Applies various computational text analysis methods (histograms, topic models, word embeddings) to a small sample of charter schools' website texts.

## Description
This repo is for data and code related to my dissertation research on charter school identities as analyzed from mission statements (MSs) on their public websites. The code is in Python 3 Jupyter Notebooks and Python scripts.

I am working to categorize identities using text analytic methods including Natural Language Processing (NLP; e.g., distinctive words, concordances), custom dictionaries, and unsupervised approaches (e.g., Structural Topic Models, Word Embeddings). I will then use regression models to connect identity patterns with community characteristics (e.g., class, race, political leanings).

At present this research is cross-sectional, looking at the population of currently open U.S. charter schools, but plans are in the works to get longitudinal MS data using the Internet Archive. I will use these data to examine survival and geographic dispersion of the different identity categories over time.

For lots more details on my data and method, see my [April 2017 pre-registration with the Open Science Foundation](https://osf.io/zgh5u/), especially the [Prereg Challenge form](https://osf.io/zgh5u/register/565fb3678c5e4a66b5582f67).

If you have questions, exciting ideas or comments, or want to congratulate me on something, do email me at jhaber@berkeley.edu. Thanks!


## Visual examples
Charter school density by state in 2014-15 school year:

![Charter school density by state 2014](data/charters_map_alpha.png)


Charter school philosophy and community income in SF Bay Area 2018:

![Charter school philosophy and community income in SF Bay Area 2018](data/SF_charters_phil_income.png)
