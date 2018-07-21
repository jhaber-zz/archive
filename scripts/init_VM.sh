#!/usr/bin/env python
# -*- coding: UTF-8

# # Initializing new VM environment with docker, text analysis tools, etc.
# RUN THIS SCRIPT AS SUPER-USER, i.e: `sudo bash init_VM.sh`

# Install latest python 3, pip, and easy-install:
apt-get install python3
pip install --upgrade pip
apt-get install python3-pip

# Docker environment:
#apt-get install docker-machine
#docker-machine create default
#docker-machine env default # Run the lines returned by this command
pip install docker
pip install docker-compose

# Scraping web content and URLs:
pip install bs4
pip install lxml
pip install google
pip install https://github.com/slimkrazy/python-google-places/zipball/master
pip install selenium
ansible-galaxy install cmprescott.chrome

# Processing text:
pip3 install nltk
pip3 install pandas
pip3 install tqdm
pip install sklearn
pip install gensim
pip install scipy
pip3 install timeout_decorator

# Shell utilities:
apt install htop # More readable version of top, for process management
apt install ncdu # Fast, comprehensive disk investigation
byobu-enable # Make sure window management software is turned on

# Setting up git-LFS:
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt-get install git-lfs
git lfs install
git lfs track "*.csv" # Possibly add other file types here
git config --global push.default simple

# Install Box SDK for working with files
pip install boxsdk

# Write and call function to import NLP tools from within Python:
function import_NLP_tools {
python - <<END
import nltk
nltk.download('punkt')
nltk.download('stopwords')
END
}
import_NLP_tools

# Set user permissions with custom playbook:
ansible-playbook jetstream-playbook.yaml

# Make sure permission structure on volume allows people to do things
#chmod -R 1777 /vol_b/data
