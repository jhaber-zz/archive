#!/usr/bin/env python
# -*- coding: UTF-8

# # Initializing new VM environment with docker, text analysis tools, etc.

# Install latest python 3 and pip:
apt-get install python3
pip install --upgrade pip

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
pip install googleplaces
pip install selenium

# Processing text:
pip install nltk
pip install sklearn
pip install gensim
pip install scipy

# Shell utilities:
apt install htop # More readable version of top, for process management
apt install ncdu # Fast, comprehensive disk investigation
byobu-enable # Make sure window management software is turned on
