#!/usr/bin/env python
# -*- coding: UTF-8


# # Google search using Python
# 
# ### Description and external resources:
# 
# > This script uses two related functions to scrape the best URL from online sources: 
# >> The Google Places API. See the [GitHub page](https://github.com/slimkrazy/python-google-places) for the Python wrapper and sample code, [Google Web Services](https://developers.google.com/places/web-service/) for general documentation, and [here](https://developers.google.com/places/web-service/details) for details on Place Details requests.
# 
# >> The Google Search function (manually filtered). See [here](https://pypi.python.org/pypi/google) for source code and [here](http://pythonhosted.org/google/) for documentation.
# 
# > To get an API key for the Google Places API (or Knowledge Graph API), go to the [Google API Console](http://code.google.com/apis/console).
# > To upgrade your quota limits, sign up for billing--it's free and raises your daily request quota from 1K to 150K (!!).
# 
# > The code below doesn't use Google's Knowledge Graph (KG) Search API because this turns out NOT to reveal websites related to search results (despite these being displayed in the KG cards visible at right in a standard Google search). The KG API is only useful for scraping KG id, description, name, and other basic/ irrelevant info. TO see examples of how the KG API constructs a search URL, etc., (see [here](http://searchengineland.com/cool-tricks-hack-googles-knowledge-graph-results-featuring-donald-trump-268231)).
# 
# > Possibly useful note on debugging: An issue causing the GooglePlaces package to unnecessarily give a "ValueError" and stop was resolved in [July 2017](https://github.com/slimkrazy/python-google-places/issues/59).
# > Other instances of this error may occur if Google Places API cannot identify a location as given. Dealing with this is a matter of proper Exception handling (which seems to be working fine below).


# ## Initializing Python search environment

# IMPORTING KEY PACKAGES
from google import search  # automated Google Search package
from googleplaces import GooglePlaces, types, lang  # Google Places API

import csv, re, os  # Standard packages
import urllib, requests  # for scraping

# Initializing Google Places API search functionality
places_api_key = re.sub("\n", "", open("places_api_key.txt").read())
print(places_api_key)

google_places = GooglePlaces(places_api_key)

# Here's a list of sites we DON'T want to spider, 
# but that an automated Google search might return...
# and we might thus accidentally spider unless we filter them out (as below)!

bad_sites = []
with open('../bad_sites.csv', 'r', encoding = 'utf-8') as csvfile:
    for row in csvfile:
        bad_sites.append(re.sub('\n', '', row))

# print(bad_sites)


# ## Reading in data

sample = []  # make empty list in which to store the dictionaries

if os.path.exists('../sample.csv'):  # first, check if modified file (with some data written already) is available on disk
    file_path = '../sample.csv'
else:  # use original data if no existing results are available on disk
    file_path = '../charter_URLs_Apr17.csv'

with open(file_path, 'r', encoding = 'utf-8')\
as csvfile: # open file                      
    print('  Reading in ' + str(file_path) + ' ...')
    reader = csv.DictReader(csvfile)  # create a reader
    for row in reader:  # loop through rows
        sample.append(row)  # append each row to the list


# Create new "URL" variable for each school, without overwriting any with data there already:
for school in sample:
    try:
        if school["URL"]:
            pass
        
    except (KeyError, NameError):
        school["URL"] = ""

        
def count_left(list_of_dicts, varname):
    '''This helper function determines how many dicts in list_of_dicts don't have a valid key/value pair with key varname.'''
    
    count = 0
    for school in list_of_dicts:
        if school[varname] == "" or school[varname] == None:
            count += 1

    print(str(count) + " schools in this data are missing " + str(varname) + "s.")

count_left(sample, 'URL')


# Take a look at the first entry's contents and the variables list in our sample (a list of dictionaries)
print(sample[1]["SEARCH"], "\n", sample[1]["OLD_URL"], "\n", sample[1]["ADDRESS"], "\n")
print(sample[1].keys())


# ## Getting URLs

def getURL(school_name, address, bad_sites_list, manual_url, known_urls):
    
    '''This function finds the one best URL for a school using two methods:
    
    1. If a school with this name can be found within 20 km (to account for proximal relocations) in
    the Google Maps database (using the Google Places API), AND
    if this school has a website on record, then this website is returned.
    If no school is found, the school discovered has missing data in Google's database (latitude/longitude, 
    address, etc.), or the address on record is unreadable, this passes to method #2. 
    
    2. An automated Google search using the school's name + address. This is an essential backup plan to 
    Google Places API, because sometimes the address on record (courtesy of Dept. of Ed. and our tax dollars) is not 
    in Google's database. For example, look at: "3520 Central Pkwy Ste 143 Mezz, Cincinnati, OH 45223". 
    No wonder Google Maps can't find this. How could it intelligibly interpret "Mezz"?
    
    Whether using the first or second method, this function excludes URLs with any of the 62 bad_sites defined above, 
    e.g. trulia.com, greatschools.org, mapquest. It returns the number of excluded URLs (from either method) 
    and the first non-bad URL discovered.'''
    
    
    ## INITIALIZE
    
    new_urls = []    # start with empty list
    good_url = ""    # output goes here
    k = 0    # initialize counter for number of URLs skipped
    
    radsearch = 15000  # define radius of Google Places API search, in km
    numgoo = 20  # define number of google results to collect for method #2
    
    search_terms = school_name + " " + address
    print("Getting URL for", school_name + ", " + address + "...")    # show school name & address
    
    
    
    ## FIRST URL-SCRAPE ATTEMPT: GOOGLE PLACES API
    # Search for nearest school with this name within radsearch km of this address
    
    try:
        query_result = google_places.nearby_search(
            location=address, name=school_name,
            radius=radsearch, types=[types.TYPE_SCHOOL], rankby='distance')
        
        for place in query_result.places:
            place.get_details()  # Make further API call to get detailed info on this place

            found_name = place.name  # Compare this name in Places API to school's name on file
            found_address = place.formatted_address  # Compare this address in Places API to address on file

            try: 
                url = place.website  # Grab school URL from Google Places API, if it's there

                if any(domain in url for domain in bad_sites_list):
                    k+=1    # If this url is in bad_sites_list, add 1 to counter and move on
                    print("  URL in Google Places API is a bad site. Moving on.")

                else:
                    good_url = url
                    known_urls.append(good_url)
                    print("    Success! URL obtained from Google Places API with " + str(k) + " bad URLs avoided.")
                    
                    '''
                    # For testing/ debugging purposes:
                    
                    print("  VALIDITY CHECK: Is the discovered URL of " + good_url + \
                          " consistent with the known URL of " + manual_url + " ?")
                    print("  Also, is the discovered name + address of " + found_name + " " + found_address + \
                          " consistent with the known name/address of: " + search_terms + " ?")
                    '''
                    
                    if manual_url != "":
                        if manual_url == good_url:
                            print("    Awesome! The known and discovered URLs are the SAME!")
                            
                    return(k, good_url)  # Returns valid URL of the Place discovered in Google Places API
        
            except:  # No URL in the Google database? Then try next API result or move on to Google searching.
                print("  No URL available through Google Places API. Moving on to Google search.")
                pass
    
    except:
        print("  Google Places API search failed. Moving on to Google search.")
        pass
    
    

    ## SECOND URL-SCRAPE ATTEMPT: FILTERED GOOGLE SEARCH
    # Automate Google search and take first result that doesn't have a bad_sites_list element in it.
    
    
    # Loop through google search output to find first good result:
    try:
        new_urls = list(search(search_terms, stop=numgoo, pause=5.0))  # Grab first numgoo Google results (URLs)
        print("  Google search completed OK.")
        
        for url in new_urls:
            if any(domain in url for domain in bad_sites_list):
                k+=1    # If this url is in bad_sites_list, add 1 to counter and move on
                print("  Bad site detected. Moving on.")
            else:
                good_url = url
                known_urls.append(good_url)
                print("    Success! URL obtained by Google search with " + str(k) + " bad URLs avoided.")
                break    # Exit for loop after first good url is found
                
    
    except:
        print("  Problem with collecting Google search results. Try this by hand instead.")
            
        
    '''
    # For testing/ debugging purposes:
    
    if k>2:  # Print warning messages depending on number of bad sites preceding good_url
        print("  WARNING!! CHECK THIS URL!: " + good_url + \
              "\n" + str(k) + " bad Google results have been omitted.")
    if k>1:
        print(str(k) + " bad Google results have been omitted. Check this URL!")
    elif k>0:
        print(str(k) + " bad Google result has been omitted. Check this URL!")
    else: 
        print("  No bad sites detected. Reliable URL!")
    '''
    
    
    if manual_url != "":
        if manual_url == good_url:
            print("    Awesome! The known and discovered URLs are the SAME!")
    
    if good_url == "":
        print("  WARNING! No good URL found via API or google search.\n")
    
    return(k, good_url)


numschools = 0  # initialize scraping counter
known_URLs = []  # initialize list of known URLs

keys = sample[0].keys()  # define keys for writing function
fname = "../sample.csv"  # define file name for writing function


def dicts_to_csv(list_of_dicts, file_name, header):
    '''This helper function writes a list of dictionaries to file_name.csv, with column names given in header.'''
    
    with open(file_name, 'w') as output_file:
        print("Saving to " + str(file_name) + "...")
        dict_writer = csv.DictWriter(output_file, header)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)
        

'''
# Now to call the above function and actually scrape these things!
for school in sample: # loop through list of schools
    if school["URL"] == "":  # if URL is missing, fill that in by scraping
        try:
            numschools += 1
            school["NUM_BAD_URLS"], school["URL"] = "", "" # start with empty strings
            school["NUM_BAD_URLS"], school["URL"] = getURL(school["SCH_NAME"], school["ADDRESS"], bad_sites, school["MANUAL_URL"], known_URLs)
        except:  # Save sample to file (can continue to load and add to it)
            dicts_to_csv(sample, fname, keys)
    
    else:
        if school["URL"]:
            pass  # If URL exists, don't bother scraping it again

        else:  # If URL doesn't yet have content, then scrape it!
            try:
                numschools += 1
                school["NUM_BAD_URLS"], school["URL"] = "", "" # start with empty strings
                school["NUM_BAD_URLS"], school["URL"] = getURL(school["SCH_NAME"], school["ADDRESS"], bad_sites, school["MANUAL_URL"], known_URLs)
            except:  # Save sample to file (can continue to load and add to it)
                dicts_to_csv(sample, fname, keys)

print("\n\nURLs discovered for " + str(numschools) + " schools.")

dicts_to_csv(sample, fname, keys)
'''

# different approach for 75 remaining sites--do them by hand!

for school in sample:
    if school["URL"] == "":
        k = 0  # initialize counter for number of URLs skipped
        school["NUM_BAD_URLS"] = ""

        print("Scraping URL for " + school["SEARCH"] + "...")
        urls_list = list(search(school["SEARCH"], stop=20, pause=10.0))
        print("  URLs list collected successfully!")

        for url in urls_list:
            if any(domain in url for domain in bad_sites):
                k+=1    # If this url is in bad_sites_list, add 1 to counter and move on
                print("  Bad site detected. Moving on.")
            else:
                good_url = url
                print("    Success! URL obtained by Google search with " + str(k) + " bad URLs avoided.")

                school["URL"] = good_url
                school["NUM_BAD_URLS"] = k
                
                dicts_to_csv(sample, fname, keys)
                count_left(sample, 'URL')
                break    # Exit for loop after first good url is found                               
                                           
    else:
        pass

dicts_to_csv(sample, fname, keys)
count_left(sample, 'URL')