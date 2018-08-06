#!/usr/bin/env python
# -*- coding: UTF-8

# # Wget with `parallel` and --accept!

# import necessary libraries for smart use of wget
import os, subprocess #for running terminal commands and folder management
import csv #for reading and writing data to .csv format
import re #regular expressions
import time #for dating things
import urllib
from urllib.parse import urlparse

#setting directories
micro_sample13 = "/vol_b/data/Charter-school-identities/data/micro-sample13_coded.csv"
full_data = "/vol_b/data/Charter-school-identities/data/charter_URLs_2014.csv"
wget_folder = "/vol_b/data/wget/2017parllwget/"


# ### Defining helper functions

def get_vars(data):
    """This sets variables based on the data source called."""
    
    if data==full_data:
        URL_variable = "TRUE_URL"
        NAME_variable = "SCH_NAME"
        ADDR_variable = "ADDRESS"
    
    elif data==micro_sample13:
        URL_variable = "URL"
        NAME_variable = "SCHNAM"
        ADDR_variable = "ADDRESS"
    
    else:
        try:
            print("Error processing variables from data file " + str(URL_data) + "!")
        except:
            print("ERROR: No data source established!")
    
    return(URL_variable,NAME_variable,ADDR_variable)


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
        return(result)


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
'''


def contains_html(my_folder):
    """check if a wget is success by checking if a directory has a html file"""

    for r,d,f in os.walk(my_folder):
        for file in f:
            if file.endswith('.html'):
                return True
    return False


def write_list(alist, file_name):
    '''Write alist to file_name'''
    with open(file_name, 'w') as file_handler:
        for elem in alist:
            file_handler.write("{}\n".format(elem))
        return


# ### Setting wget parameters

# Template command-line use of parallel + wget:
'''parallel -j 100 wget --mirror --warc-file={} --warc-cdx --page-requisites --html-extension \
--convert-links --execute robots=off --directory-prefix=. --user-agent=Mozilla --follow-tags=a http://{} < ../list.txt'''


#List of non-essential and likely huge directories to exclude from web download:
exclude_dirs = "/event*,/Event*,/event,/Event,/events,/Events,/*/Event,/*/event,/*/*/Event,/*/*/event,/*/*/Events,/*/*/events,\
/apps/events,/apps/Events,/Apps/events,/Apps/Events,/apps/event,/apps/Event,/Apps/event,Apps/Event,\
/attend-event,/attend-events,/Attend-Event,/Attend-Events,/apps/attend-events,/apps/attend-event,\
/event-calendar,/event_calendar,/Event-Calendar,/Event-calendar,/Event_Calendar,/apps/event-calendar,/apps/event_calendar,/apps/Event-Calendar,/apps/Event_Calendar,\
/Carmichael/Events,/Carmichael/Events:Month,/Carmichael/Events:Week,/Carmichael/Events:Day,/Carmichael/Events:Year,\
/pride/events,/community-events,/about/events,/tribe-events,/tribe-event,/tribe_event,/tribe_events,\
/apps/events2,/events2,/apps/Events2,\
/family-caregiver-resources/calendar,/en/family-caregiver-resources/calendar,/home/calendar,\
/*calendar*,/*Calendar*,/calendar,/calendars,/Calendar,/Calendars,/*/calendar,/*/Calendar/,/*/*/calendar,/*/*/Calendar/,\
/calendar-core,/calendar_core,/Calendar-Core,/Calendar_Core,/calendarcore,/CalendarCore,\
/school-calendar,/apps/school-calendar,/school_calendar,/apps/school_calendar,/student-life/school-calendar,\
/about/calendar,/about-us/calendar,/AboutUs/calendar,/calendar-2,/calendar2,\
*login*,/*Login*,/login,/Login,/*/login,/*/Login,/*/*/login,/*/*/Login,/_login,\
/misc,/Misc,/*/misc,/*/Misc,/*/*/misc,/*/*/Misc,/miscellaneous,/*/miscellaneous,/*/*/miscellaneous,\
/portal,/Portal,/portal*,/Portal*,/portals,/Portals,/*/portals,/*/Portals,/*/portal,/*/Portal,/*/*/portal,/*/*/portals,/*/*/Portal,/*/*/Portals,\
/gateway,/apps/gateway,/Gateway,\
/news,/News,/*/news,/*/News,/*/*/news,/*/*/News,/news-events,\
/category/events,/category/event,/category/calendar,\
/file,/files,/File,/Files,/*/file,/*/files,/*/File,/*/Files,/*/*/file,/*/*/files,/*/*/File,/*/*/Files,\
/contact,/Contact,/contact-us,/Contact-us,/Contact-Us,/contactus,/ContactUs,/Contactus,/contact_us,/Contact_us,/Contact_Us,\
/*/contact,/*/Contact,/*/contact-us,/*/Contact-us,/*/Contact-Us,/*/contactus,/*/ContactUs,/*/Contactus,/*/contact_us,/*/Contact_us,/*/Contact_Us,\
/*/*/contact,/*/*/Contact,/*/*/contact-us,/*/*/Contact-us,/*/*/Contact-Us,/*/*/contactus,/*/*/ContactUs,/*/*/Contactus,/*/*/contact_us,/*/*/Contact_us,/*/*/Contact_Us,\
/UserFiles,/userfiles,/Userfiles,/User-Files,/user-files,/User-files,\
/*/UserFiles,/*/userfiles,/*/Userfiles,/*/User-Files,/*/user-files,/*/User-files,\
/*/*/UserFiles,/*/*/userfiles,/*/*/Userfiles,/*/*/User-Files,/*/*/user-files,/*/*/User-files,\
/facilities,/Facilities,/apps/facilities,/apps/Facilities,\
/2007,/2008,/2009,/2010,/2011,/2012,/2013,/2014,/2015,/2016,/2017,/2018,\
/css,/CSS,/*/css,/*/CSS,/*/*/css,/*/*/CSS,\
/cms,/CMS,/*/cms,/*/CMS,/*/*/cms,/*/*/CMS,/cms_files,/cms_file,/apps/cms_files,/apps/cms_file,\
/Blackboard,/blackboard,/chalkboard,/apps/chalkboard,/apps/Chalkboard,/apps/chalk_board,/chalk_board,/chalk-board,/apps/chalk-board,/DesktopModules,/desktopmodules,/Desktop_Modules,/desktop_modules,/wp-content,/wp_content,\
/apps/email,/email,/blog,/Blog,/apps/blog,/apps/Blog,/site/blog,/site/Blog,/blog-and-news,\
/protected,/_protected,/Protected,/_Protected,/apps/protected,/apps/protected,/apps/Protected,/apps/_protected,/apps/_Protected,\
/plugin,/plugins,/Plugin,/Plugins,/*/plugin,/*/Plugin,/*/plugins,/*/Plugins,/*/*/plugin,/*/*/plugins,/*/*/Plugin,/*/*/Plugins\
/upload,/uploads,/Upload,/Uploads,/*/upload,/*/uploads,/*/Upload,/*/Uploads,/*/*/upload,/*/*/uploads,/*/*/Upload,/*/*/Uploads\
/downloads,/download,/Download,/Downloads"

#Define files to exclude from download; STARTS WITH SPACE:
reject_files = ' --reject "events,Events,news,News,calendar,calendars,Calendar,Calendars,contact,Contact,contact-us,Contact-Us,login,Login,SignIn,Download,download,\
*events*,*Events*,*News*,*calendar*,*calendars*,*Calendar*,*Calendars*,*contact*,*Contact*,*contact-us*,*Contact-Us*,*login*,*Login*,*SignIn*,*Download*"'


#---------------------------------------------------------------
#Define most general wget parameters (more specific params below)
#This list would not be so long if Parallel would allow wget to read from /usr/local/etc/wgetrc
wget_general_options = '--no-parent --level 7 --no-check-certificate \
--recursive --adjust-extension --convert-links --page-requisites --wait=2 --random-wait \
-e --robots=off --follow-ftp --secure-protocol=auto --retry-connrefused --tries=12 --no-remove-listing \
--local-encoding=UTF-8 --no-cookies --default-page=default --server-response --trust-server-names \
--header="Accept:text/html" --exclude-directories=' + exclude_dirs + reject_files
#---------------------------------------------------------------

#Other options:
#--verbose --convert-file-only --force-directories --show-progress 
#--user_agent = Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:40.0) Gecko/20100101 Firefox/40.0

# Some of these options explained: 
'''
--warc-file turns on WARC output to the specified file
--warc-cdx tells wget to dump out an index file for our new WARC file
--page-requisites will grab all of the linked resources necessary to render the page (images, css, javascript, etc)
--adjust-extension appends .html to the files when appropriate
--convert-links will turn links into local links as appropriate
--execute robots=off turns off wget's automatic robots.txt checking
--exclude-directories includes a comma-separated list of directories that wget should exclude in the archive
--user-agent overrides wget's default user agent
--random-wait will randomize that wait to between 5 and 15 seconds
'''


#To help with parsimony/download speeds, these lists expand rejections and limit acceptances:
#Note: THESE EXT LISTS START WITH A SPACE.

reject_exts = ' --reject .mov,.MOV,.avi,.AVI,.mpg,.MPG,.mpeg,.MPEG,.mp3,.MP3,.mp4,.MP4,.ppt,.PPT,.pptx,.PPTX,.zip,.7z,.pkg,.deb' + \
',.png,.PNG,.gif,.GIF,.jpg,.JPG,.jpeg,.JPEG,.pdf,.PDF,.pdf_,.PDF_,.doc,.DOC,.docx,.DOCX,.xls,.XLS,.xlsx,.XLSX,.csv,.CSV' #drop this 2nd line when running at scale

accept_exts = ' --accept .htm,.html,.asp,.aspx,.php,.shtml,.cgi,.php,.pl,.jsp'


def wget_params(link, host, title, parent_folder, wget_genopts):
    '''Define parameters for wget command, given the input URL `link` and the root of directory hierarchy `parent_folder`.'''

    wget_locs = ' --directory-prefix="' + parent_folder + title + '/" --referer=' + host + ' \
    --warc-cdx --warc-file=' + title + '_warc --warc-max-size=1.5G'
    #'--append-output=wgetNov17_log.txt
    
    wget_reject_options = wget_genopts + wget_locs + reject_exts
    
    wget_accept_options = wget_genopts + wget_locs + accept_exts
    
    return(wget_reject_options, wget_accept_options)


# ### Defining wget and parallel wget functions

'''def run_wget_command(tuple_list, parent_folder):
    """wget on list of tuples (holding link, school name, address) and print output to appropriate folders. 
    Uses two kinds of wget: Reject approach is more comprehensive and thus restrictive, we'll try it first;
    If that doesn't give any .html files, then use accept approach! This gives less results but is more reliable.
    """  
    
    for tup in tuple_list:
        # process tuple_list into useful variables
        school_link = tup[0]
        school_title = re.sub(" ","_",(tup[1]+" "+tup[2][-8:-6]))
        school_address = tup[2]
        school_host = urlparse(school_link).hostname
        
        # use tuple to create a name for the folder
        dirname = school_title #+ " " + school_address #format_folder_name(k, school_title)
        
        os.chdir(parent_folder) #everything points to parent folder, so start here
        if not os.path.exists(dirname): #create dir my_folder if it doesn't exist yet
            os.makedirs(dirname)
        os.chdir(dirname) #navigate to the correct folder, ready to wget
        specific_folder = parent_folder + '/'+ dirname
        
        k = 0 # initialize this numerical variable k, which keeps track of which entry in the sample we are on.
        k += 1 # Add one to k, so we start with 1 and increase by 1 all the way up to length of list used to call command
        print("Capturing website data for " + school_title + ", which is school #" + str(k) + " of " + str(len(tuple_list)) + "...")
        
        reject_options, accept_options = wget_params(school_link, school_host, school_title, parent_folder, wget_general_options)
        
        print("  Running wget with reject options...")
        subprocess.run('time wget --user-agent="Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:40.0) Gecko/20100101 Firefox/40.0" ' + reject_options + ' ' + school_link, stdout=subprocess.PIPE, shell=True)
        if not contains_html(specific_folder):
            print("  Nope! Back-up plan: Running wget with accept options...")
            subprocess.run('time wget --user-agent="Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:40.0) Gecko/20100101 Firefox/40.0" ' + accept_options + ' ' + school_link, stdout=subprocess.PIPE, shell=True) #back-up plan if reject fails: wget accept!
        #cwd=specific_folder
    
    print("\nDone!")
    '''
            
            

def run_wget_parallel(tuple_list, parent_folder):
    """wget on list of tuples (holding link, school name, address) and print output to appropriate folders.
    Uses reject options together with concurrency via the `parallel` package to accelerate the download.
    """  
    
    #make these new lists for input to Parallel: [0] link; [1] readable name (school name + state); [2] root host name
    #new_tuplist = [[tup[0], tup[1]+" "+tup[2], urlparse(tup[0]).hostname] for tup in tuple_list]
    links_list = [tup[0] for tup in tuple_list]
    names_list = [re.sub(" ","_",(tup[1]+" "+tup[2][-8:-6])) for tup in tuple_list] #to include state (and reduce odds of folder name duplication), add last two chars from ADDRESS
    hosts_list = [urlparse(tup[0]).hostname for tup in tuple_list]
    
    os.chdir(parent_folder) #everything points to parent folder, so start here
        
    write_list(links_list, 'links_list.txt') #save lists as .txt files for easy reference and visibility in shell
    write_list(names_list, 'names_list.txt')
    write_list(hosts_list, 'hosts_list.txt')
    
    for host in set(hosts_list): #build folder hierarchy for parallel wget to use
        if not os.path.exists(host): #create dir host if it doesn't exist yet
            os.makedirs(host)
        
    # wget and parallel are shell commands, so we run with the subprocess module:
    # Note: unlike Python, the parallel package uses standard indexing (1,2,3 not 0,1,2)
    subprocess.run('time parallel --jobs 350 --eta --progress --bar --will-cite --link --keep-order \
    --timeout 1.5h --resume-failed --joblog joblog_parllwget_' + re.sub("/","-",time.strftime("%m/%d/%Y")) + ' \
    -- wget ' + wget_general_options + accept_exts + ' --user-agent=Mozilla \
    --warc-file={2}_warc --warc-cdx --warc-max-size=1.5G \
    --directory-prefix="' + parent_folder + '{2}/" --referer={3} {1} \
    :::: links_list.txt names_list.txt hosts_list.txt', stdout=subprocess.PIPE, shell=True)
    
    #Other options:
    #cwd=parent_folder --verbose --append-output={3}/{2}_wgetNov17_log.txt
    
    
    #A THOROUGH BUT TIME-INEFFICIENT ADDENDUM: 
    
    #If a site produces no HTML files, then we run (non-parallel) wget accept as a backup
    k=0 #initialize numerical variable k, which keeps track of which entry in the sample we are on.
    m=0 #initialize numerical variable m, which keeps track of how many schools don't yet have HTML in their folders. 
    
    for tup in tuple_list:
        k+=1
        # process tuple_list into useful variables
        school_link = tup[0]
        school_title = re.sub(" ","_",(tup[1]+" "+tup[2][-8:-6]))
        #school_address = tup[2]
        school_host = urlparse(school_link).hostname
        
        if not contains_html(os.path.dirname(school_host)):
            m+=1 # Add one to k, so we start with 1 and increase by 1 all the way up to length of list used to call command
            print("No HTML detected from parallel wget! Using (non-parallel) wget accept to capture HTML for " + school_title + ", which is overall school #" + str(k) + " and non-HTML schools #" + str(m) + "  of a total of " + str(len(tuple_list)) + " schools...")
            
            reject_options, accept_options = wget_params(school_link, school_host, school_title, parent_folder, wget_general_options)
            
            subprocess.run('time wget --user-agent="Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:40.0) Gecko/20100101 Firefox/40.0" ' + accept_options + ' ' + school_link, stdout=subprocess.PIPE, shell=True, cwd=parent_folder)
            
            
    # OLD OPTIONS ETC FOR REFERENCE
    '''"(echo {};sleep 0.1"
    --bar --progress --will-cite --max-replace-args=3
    :::: ' + links_location + ' ' + names_location + ' ' + hosts_location
    encoding='UTF-8'
    echo_and_run() { echo "\$ $@" ; "$@" ; }; echo_and_run  #for debugging the command
    
    links_location = parent_folder + "links_list.txt"
    names_location = parent_folder + "names_list.txt"
    hosts_location = parent_folder + "hosts_list.txt"
    
    subprocess.run('echo_and_run() { echo "\$ $@" ; "$@" ; }; echo_and_run \
    parallel --jobs 100 --progress --eta --bar --will-cite --link --keep-order --verbose -- wget ' + wget_general_options +\
              ' directory-prefix=' + parent_folder + '{} --warc-file=' + parent_folder + 'warc-{} \
              --referer={} --append-output=' + parent_folder + 'wgetNov17_log.txt \
              {} < links_list.txt', stdout=subprocess.PIPE, shell=True, cwd=parent_folder)
    '''
    
    print("\nDone!")    


    
# ### Preparing data for wget run

#set charter school data file and corresponding varnames
URL_data = full_data #running now on URL list of full charter population
URL_var,NAME_var,ADDR_var = get_vars(URL_data) #get varnames depending on data source

sample = [] # make empty list
with open(URL_data, 'r', encoding = 'Latin1') as csvfile: # open file
    reader = csv.DictReader(csvfile) # create a reader
    for row in reader: # loop through rows
        sample.append(row) # append each row to the list   
#note: each row, sample[i] is a dictionary with keys as column name and value as info

# turning vars into tuples we can use with wget!
# first, make some empty lists
url_list = []
name_list = []
terms_list = []

# now fill lists with content from sample
for school in sample:
    if (school[URL_var] is not None and school[URL_var]!=0 and school[URL_var]!="0"):
        url_list.append(school[URL_var])
        name_list.append(school[NAME_var])
        terms_list.append(school[ADDR_var])

school_tuple_list = list(zip(url_list, name_list, terms_list))
#Here's what these tuples look like:
#print(tuple_list[:3])
#print("\n", tuple_list[1][1].title())


# define crawling sample--how much of the micro-sample of 300 or 2014 pop. of 6,752?

#expt_links = ['https://vangoghcs-lausd-ca.schoolloop.com/', 'https://cms.springbranchisd.com/wais/']
#school_tuple_test = [school for school in school_tuple_list if school[0] in expt_links]
#school_tuple_test = school_tuple_list[:50]


# ### Running wget

#run_wget_command(school_tuple_list, wget_folder)

run_wget_parallel(school_tuple_list, wget_folder)


# ### Limits of wget--and alternatives!

# Wget only works for static HTML and it doesn’t support JavaScript. Thus any element generated by JS will not be captured. 

# More info:
# https://www.petekeen.net/archiving-websites-with-wget
# http://askubuntu.com/questions/411540/how-to-get-wget-to-download-exact-same-web-page-html-as-browser
# https://www.reddit.com/r/linuxquestions/comments/3tb7vu/wget_specify_dns_server/

# Selenium provides a better way to scrape JS-heavy sites:
# https://medium.com/@hoppy/how-to-test-or-scrape-javascript-rendered-websites-with-python-selenium-a-beginner-step-by-c137892216aa

# To see Python scripts using Selenium to crawl, check out our team's repo here:
# https://github.com/URAP-charter/web_scraping