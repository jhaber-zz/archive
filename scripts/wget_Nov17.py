
# coding: utf-8

# # Wget using reject, THEN accept!

# In[12]:


# import necessary libraries for smart use of wget
import os, csv
import shutil
import urllib
from urllib.request import urlopen
from socket import error as SocketError


# In[26]:


#setting directories
micro_sample13 = "/vol_b/data/Charter-school-identities/data/micro-sample13_coded.csv"
full_data = "/vol_b/data/Charter-school-identities/data/charter_URLs_2014.csv"
wget_folder = "/vol_b/data/wget/Nov_2017/"


# ### Helper Functions

# In[14]:


def get_parent_link(text):
    """Function to get parents' links. Return a list of valid links."""
    ls= get_parent_link_helper(5, text, []);
    if len(ls) > 1:
        return ls[0]
    return str

def get_parent_link_helper(level, text, result):
    """This is a tail recursive function
    to get parent link of a given link. Return a list of urls """
    if level == 0 or not check(str):
        return ''
    else:
        result += [text]
        return get_parent_link_helper(num -1, str[: text.rindex('/')], result)


# In[15]:


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


# In[16]:


#testing methods
print(format_folder_name(30, "name me"))



# In[17]:


def check(url):
    """ Helper function, check if url is a valid list"""
    try:
        urlopen(url)
        
    except urllib.error.URLError:
        print("urllib.error.URLError")
        return False
    except urllib.error.HTTPError:
        print('urllib.error.HTTPError')
        return False
    except SocketError:
        print('SocketError')
        return False
    return True


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


# In[18]:


def count_valid_links(list_of_links, valid_file, invalid_file):
    count_success, count_fail = 0, 0
    valid, invalid = '', ''
    for l in list_of_links:
#         print(l)
        if check(l):
            valid += l + '\n'
            count_success +=1
        else:
            invalid += l + '\n'
            count_fail += 1
#             print(l)
    write_file(valid, valid_file)
    write_file(invalid, invalid_file)
    return count_success, count_fail


# In[19]:


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
    wget_reject_options = '    --no-parent --show-progress --progress=dot --page-requisites --recursive --append-output=wgetNov17_log --level inf     --warc-file={} --warc-cdx --directory-prefix= ' + parent_folder + ' --referer= ' + get_parent_link(link) + '     --random-wait --timestamping --show-progress --progress=dot --verbose     --no-remove-listing --follow-ftp --no-clobber --adjust-extension --convert-links     --retry-connrefused --tries=10 -execute robots=off --no-cookies --header "Host: jrs-s.net" --secure-protocol=auto --no-check-certificate     --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:11.0) Gecko/20100101 Firefox/11.0"     --reject .mov,.MOV,.avi,.AVI,.mpg,.MPG,.mpeg,.MPEG,.mp3,.MP3,.mp4,.MP4,.ppt,.PPT,.pptx,.PPTX'
    
    wget_accept_options = '    --no-parent --show-progress --progress=dot --page-requisites --recursive --append-output=wgetNov17_log --level inf     --warc-file={} --warc-cdx --directory-prefix= ' + parent_folder + ' --referer= ' + get_parent_link(link) + '     --random-wait --timestamping --show-progress --progress=dot --verbose     --no-remove-listing --follow-ftp --no-clobber --adjust-extension --convert-links     --retry-connrefused --tries=10 -execute robots=off --no-cookies --header "Host: jrs-s.net" --secure-protocol=auto --no-check-certificate     --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:11.0) Gecko/20100101 Firefox/11.0"     --accept .htm,.html,.asp,.aspx,.php,.shtml,.cgi,.php,.pl,.jsp'
    
    # Run wget reject, then wget accept if necessary!
    os.system('parallel -j 100 wget ' + wget_reject_options + ' ' + link) #use concurrency to speed up the web-crawl
    if not contains_html(specific_folder):
        os.system('parallel -j 100 wget ' + wget_accept_options + ' ' + link) #back-up plan if reject fails: wget accept!


# ### Running wget

# In[27]:


sample = [] # make empty list
with open(micro_sample13, 'r', encoding = 'Latin1')as csvfile: # open file
    reader = csv.DictReader(csvfile) # create a reader
    for row in reader: # loop through rows
        sample.append(row) # append each row to the list
        
#note: each row, sample[i] is a dictionary with keys as column name and value as info


# In[6]:


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


# In[7]:


tuple_list = list(zip(url_list, name_list))
# Let's check what these tuples look like:
print(tuple_list[:3])
print("\n", tuple_list[1][1].title())


# In[12]:


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
