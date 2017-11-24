#!/usr/bin/env python
# -*- coding: UTF-8

# # Wget using reject, THEN accept!

# import necessary libraries for smart use of wget
import os, csv
import shutil
import urllib
from urllib.request import urlopen
from urllib.parse import urlparse
from socket import error as SocketError

#setting directories
micro_sample13 = "/vol_b/data/Charter-school-identities/data/micro-sample13_coded.csv"
full_data = "/vol_b/data/Charter-school-identities/data/charter_URLs_2014.csv"
wget_folder = "/vol_b/data/wget/Nov_2017/"


# ### Helper Functions

'''def get_parent_link(text):
    """Function to get parents' links. Return a list of valid links."""
    ls= get_parent_link_helper(5, text, []);
    if len(ls) > 1:
        return ls[0]
    return text

def get_parent_link_helper(level, text, result):
    """This is a tail recursive function
    to get parent link of a given link. Return a list of urls """
    if not check(text):
        return ''
    elif level != 0:
        result += [text]
        return get_parent_link_helper(level-1, text[: str.rindex(text, '/')], result)
        #return text[-1: str(text.rindex('/'))
    else:
        return(result)'''

    
def check(url):
    """ Helper function, check if url is a valid list <- our backup plan
    This function helps to check the url that has service unavailable issues
    Since status code fails to check this."""
    
    try:
        urlopen(url)
        
    except urllib.error.URLError:
        print(url + " :URLError")
        return False
    except urllib.error.HTTPError:
        print(url +' :HTTPError')
        return False
    except SocketError:
        print(url + 'SocketError')
        return False
    return True


def check_url(url):
    """This functions uses the status code to determine if the link is valid. 
    This resolves the links that redirect and most cases of authentication problems"""
    
    code = "[no code collected]"
    if url == "":
        return False
    
    try:
        r = requests.get(url, auth=HTTPDigestAuth('user', 'pass'), headers= {'User-Agent':"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"})
        code = r.status_code
        #backup plan for service unavailable issues
        if code == 503:
            return check(url)
        if code < 400:
            return True   
    
    except:
        pass
    print("Encountered this invalid link: " + str(url) +" ---Error code: " + str(code))
    return False    


def format_folder_name (k, name):
    """Format a folder nicely for easy access"""
    if k < 10: # Add two zeros to the folder name if k is less than 10 (for ease of organizing the output folders)
        dirname = "00" + str(k) + " " + name
    elif k < 100: # Add one zero if k is less than 100
        dirname = "0" + str(k) + " " + name
    else: # Add nothing if k>100
        dirname = str(k) + " " + name
    return dirname

def contains_html(my_folder):
    """check if a wget is success by checking if a directory has a html file"""

    for r,d,f in os.walk(my_folder):
        for file in f:
            if file.endswith('.html'):
                return True
    return False

def count_with_file_ext(folder, ext):
    count = 0
    for r,d,f in os.walk(my_folder):
        for file in f:
            if file.endswith(ext):
                count +=1
    return count 

# write a file and add num line at the beginning of line
def write_to_file(num, link, file_name):
    with open(file_name, "a") as text_file:
        text_file.write(str(num) + "\t" + link +"\n")

# just write str to file
def write_file(str, file_name):
    with open(file_name, "a") as text_file:
        text_file.write(str)
        
def reset(folder, text_file_1, text_file_2):
    """Deletes all files in a folder and set 2 text files to blank"""
    parent_folder = folder[: folder.rindex('/')]
    shutil.rmtree(folder)
    os.makedirs(folder)
    filelist = [ f for f in os.listdir(folder) if f.endswith(".bak") ]
    for f in filelist:
        os.unlink(f)
    for file_name in [text_file_1, text_file_2]:
        reset_text_file(file_name)
        
def reset_text_file(file_name):
    if os.path.exists(file_name):
            with open(file_name, "w") as text_file:
                text_file.write("")



def read_txt(txt_file):
    links = []
    count = 0
    with open(txt_file) as f:
        for line in f:   
            
            elem =  line.split('\t')[1].rstrip()
            count +=1
    
#             print(elem)
            links += [elem.rstrip()]
    return links, count

def read_txt_2(txt_file):
    links = []
    count = 0
    with open(txt_file) as f:
        for line in f:   
            
#             elem =  line.split('\t')[1].rstrip()
#             if elem.endswith('\'):
#                 elem = elem[:-1]
            count +=1
    
#             print(elem)
            links += [line.rstrip()]
    return links, count



def run_wget_command(link, parent_folder, my_folder):
    """wget on link and print output to appropriate folders. Uses two kinds of wget:
    Reject approach is more comprehensive and thus restrictive, we'll try it first;
    If that doesn't give any .html files, then use accept approach! This gives less results but is more reliable.
    """
    
    os.chdir(parent_folder) #navigate to parent folder
    if not os.path.exists(my_folder): #create dir my_folder if it doesn't exist yet
        os.makedirs(my_folder)
    os.chdir(my_folder) #navigate to the correct folder, ready to wget
    specific_folder = parent_folder + '/'+ my_folder
    
    # Define parameters for wget command
    wget_reject_options = '    --no-parent --show-progress --progress=dot --page-requisites --recursive --append-output=wgetNov17_log --level inf     --warc-file={} --warc-cdx --directory-prefix= ' + parent_folder + ' --referer= ' + urlparse(link).hostname + '     --random-wait --timestamping --show-progress --progress=dot --verbose     --no-remove-listing --follow-ftp --no-clobber --adjust-extension --convert-links     --retry-connrefused --tries=10 -execute robots=off --no-cookies --header "Host: jrs-s.net" --secure-protocol=auto --no-check-certificate     --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:11.0) Gecko/20100101 Firefox/11.0"     --reject .mov,.MOV,.avi,.AVI,.mpg,.MPG,.mpeg,.MPEG,.mp3,.MP3,.mp4,.MP4,.ppt,.PPT,.pptx,.PPTX'
    
    wget_accept_options = '    --no-parent --show-progress --progress=dot --page-requisites --recursive --append-output=wgetNov17_log --level inf     --warc-file={} --warc-cdx --directory-prefix= ' + parent_folder + ' --referer= ' + urlparse(link).hostname + '     --random-wait --timestamping --show-progress --progress=dot --verbose     --no-remove-listing --follow-ftp --no-clobber --adjust-extension --convert-links     --retry-connrefused --tries=10 -execute robots=off --no-cookies --header "Host: jrs-s.net" --secure-protocol=auto --no-check-certificate     --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:11.0) Gecko/20100101 Firefox/11.0"     --accept .htm,.html,.asp,.aspx,.php,.shtml,.cgi,.php,.pl,.jsp'
    
    # Run wget reject, then wget accept if necessary!
    os.system('parallel -j 100 --no-notice wget ' + wget_reject_options + ' ' + link) #use concurrency to speed up the web-crawl
    if not contains_html(specific_folder):
        os.system('parallel -j 100 --no-notice wget ' + wget_accept_options + ' ' + link) #back-up plan if reject fails: wget accept!


# ### Running wget

sample = [] # make empty list
with open(micro_sample13, 'r', encoding = 'Latin1')as csvfile: # open file
    reader = csv.DictReader(csvfile) # create a reader
    for row in reader: # loop through rows
        sample.append(row) # append each row to the list
        
#note: each row, sample[i] is a dictionary with keys as column name and value as info



# turning this into tuples we can use with wget!
# first, make some empty lists
url_list = []
name_list = []
terms_list = []

# now let's fill these lists with content from the sample
for school in sample:
    url_list.append(school["URL"])
    name_list.append(school["SCHNAM"])
    terms_list.append(school["ADDRESS"])



tuple_list = list(zip(url_list, name_list))
# Let's check what these tuples look like:
#print(tuple_list[:3])
#print("\n", tuple_list[1][1].title())



k=0 # initialize this numerical variable k, which keeps track of which entry in the sample we are on.

#testing the first 10 tuples
tuple_test = tuple_list[:10]


for tup in tuple_test:
    school_title = tup[1].title()

    k += 1 # Add one to k, so we start with 1 and increase by 1 all the way up to entry # 300
    print("Capturing website data for", school_title + ", which is school #" + str(k), "of 300...")
    
    # use the tuple to create a name for the folder
    dirname = format_folder_name(k, school_title)
    
    run_wget_command(tup[0], wget_folder, dirname)
    
    school_folder = wget_folder + '/'+ dirname

print("done!")    


# ### Limitation of wget
# 
# -only works for static HTML and it doesn’t support JavaScript. Thus any element generated by JS will not be captured. 

# More info:
# 
# https://www.petekeen.net/archiving-websites-with-wget
# 
# http://askubuntu.com/questions/411540/how-to-get-wget-to-download-exact-same-web-page-html-as-browser
# 
# https://www.reddit.com/r/linuxquestions/comments/3tb7vu/wget_specify_dns_server/
# failed: nodename nor servname provided, or not known.
# 
