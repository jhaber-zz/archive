#!/usr/bin/env python
# -*- coding: UTF-8

# # Wget using reject, THEN accept!

# import necessary libraries for smart use of wget
import os, subprocess #for running terminal commands and folder management
import csv #for reading and writing data to .csv format
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
    """Format a folder nicely for readability and easy access"""
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

def write_numstr(num, content, file_name):
    '''write a file and add num line at the beginning of line'''
    with open(file_name, "a") as text_file:
        text_file.write(str(num) + "\t" + content +"\n")

def write_str(content, file_name):
    '''Write str to file'''
    with open(file_name, "a") as text_file:
        text_file.write(content)
        
def write_list(alist, file_name):
    with open(file_name, 'w') as file_handler:
        for elem in alist:
            file_handler.write("{}\n".format(elem))
        
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

    

# Aaron's command using parallel + wget (for reference):
'''parallel -j 100 wget --mirror --warc-file={} --warc-cdx --page-requisites --html-extension \
--convert-links --execute robots=off --directory-prefix=. --user-agent=Mozilla --follow-tags=a http://{} < ../list.txt'''


#Define most general wget parameters (more specific params below)
wget_general_options = '--no-parent --recursive --level inf --warc-cdx \
--append-output=wgetNov17_log.txt --convert-file-only --no-check-certificate \
--exclude-directories = "/event*,/calendar*,/*login*,/misc,/portal,/news"'

#user_agent = Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:40.0) Gecko/20100101 Firefox/40.0


def wget_params(link, title, parent_folder, wget_genopts):
    '''Define parameters for wget command, given the input URL `link` and the root of directory hierarchy `parent_folder`.'''

    wget_locs = '--directory-prefix=' + parent_folder + ' --referer=' + urlparse(link).hostname +\
    ' --warc-file=warc-' + title
    
    wget_reject_options = wget_genopts + wget_locs + ' --reject .mov,.MOV,.avi,.AVI,.mpg,.MPG,.mpeg,.MPEG,.mp3,.MP3,.mp4,.MP4,.ppt,.PPT,.pptx,.PPTX'
    
    wget_accept_options = wget_genopts + wget_locs + ' --accept .htm,.html,.asp,.aspx,.php,.shtml,.cgi,.php,.pl,.jsp'
    
    return(wget_reject_options, wget_accept_options)



def run_wget_command(tuple_list, parent_folder):
    """wget on list of tuples (holding link, school name, address) and print output to appropriate folders. 
    Uses two kinds of wget: Reject approach is more comprehensive and thus restrictive, we'll try it first;
    If that doesn't give any .html files, then use accept approach! This gives less results but is more reliable.
    """  
    
    for tup in tuple_list:
        # process tuple_list into variables of use
        school_link = tup[0]
        school_title = tup[1]
        school_address = tup[2]
        
        # use tuple to create a name for the folder
        dirname = school_title + " " + school_address #format_folder_name(k, school_title)
        
        os.chdir(parent_folder) #everything points to parent folder, so start here
        if not os.path.exists(dirname): #create dir my_folder if it doesn't exist yet
            os.makedirs(dirname)
        os.chdir(dirname) #navigate to the correct folder, ready to wget
        specific_folder = parent_folder + '/'+ dirname
        
        k = 0 # initialize this numerical variable k, which keeps track of which entry in the sample we are on.
        k += 1 # Add one to k, so we start with 1 and increase by 1 all the way up to length of list used to call command
        print("Capturing website data for " + school_title + ", which is school #" + str(k) + " of " + str(len(tuple_list)) + "...")
        
        reject_options, accept_options = wget_params(school_link, school_title, parent_folder, wget_general_options)
        
        print("  Running wget with reject options...")
        os.system('wget ' + reject_options + ' ' + school_link) #use concurrency to speed up the web-crawl
        if not contains_html(specific_folder):
            print("  Nope! Back-up plan: Running wget with accept options...")
            os.system('wget ' + accept_options + ' ' + school_link) #back-up plan if reject fails: wget accept!
    
    print("done!")    
            
            

def run_wget_parallel(tuple_list, parent_folder):
    """wget on list of tuples (holding link, school name, address) and print output to appropriate folders.
    Uses reject options together with concurrency via the `parallel` package to accelerate the download.
    """  
    
    os.chdir(parent_folder) #everything points to parent folder, so start here
    
    #make these new lists for input to Parallel: [0] link; [1] folder name (school name + address); [2] root host name
    #new_tuplist = [[tup[0], tup[1]+" "+tup[2], urlparse(tup[0]).hostname] for tup in tuple_list]
    links_list = [tup[0] for tup in tuple_list]
    names_list = [(tup[1]+" "+tup[2]) for tup in tuple_list]
    hosts_list = [urlparse(tup[0]).hostname for tup in tuple_list]
    
    links_location = parent_folder + "links_list.txt"
    names_location = parent_folder + "names_list.txt"
    hosts_location = parent_folder + "hosts_list.txt"
    
    write_list(links_list, links_location)
    write_list(names_list, names_location)
    write_list(hosts_list, hosts_location)
    
    # OLD OPTIONS:
    # "(echo {};sleep 0.1"
    # --bar --progress --will-cite --max-replace-args=3
    # :::: ' + links_location + ' ' + names_location + ' ' + hosts_location
    
    os.system('echo_and_run() { echo "\$ $@" ; "$@" ; }; echo_and_run \
    parallel --progress --eta --max-replace-args=3 -link -keep-order --jobs 100 wget ' + wget_general_options +\
              ' directory-prefix=' + parent_folder + '{1} --warc-file=warc-{0} --referer={2} \
              --reject .mov,.MOV,.avi,.AVI,.mpg,.MPG,.mpeg,.MPEG,.mp3,.MP3,.mp4,.MP4,.ppt,.PPT,.pptx,.PPTX\
              {0} :::: links_list.txt names_list.txt hosts_list.txt')
    
    '''os.system('echo_and_run() { echo "\$ $@" ; "$@" ; }; echo_and_run \
    parallel --jobs 100 --eta --bar --will-cite --link --keep-order wget ' + wget_general_options +\
              ' directory-prefix=' + parent_folder + '{} --warc-file=warc-{} --referer={} \
              --reject .mov,.MOV,.avi,.AVI,.mpg,.MPG,.mpeg,.MPEG,.mp3,.MP3,.mp4,.MP4,.ppt,.PPT,.pptx,.PPTX\
              {} < ' + links_location)'''
    
    print("done!")    


    
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

        
school_tuple_list = list(zip(url_list, name_list, terms_list))
# Let's check what these tuples look like:
#print(tuple_list[:3])
#print("\n", tuple_list[1][1].title())


# define crawling sample--how much of the micro-sample of 300 or 2014 pop. of 6,752?
school_tuple_test = school_tuple_list[:10]

#run_wget_command(school_tuple_test, wget_folder)

run_wget_parallel(school_tuple_test, wget_folder)


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
