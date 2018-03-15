#!/usr/bin/env python
# -*- coding: UTF-8

# # Visualizing data


"""This script uses matplotlib and seaborn to plot a big DataFrame.

Author: Jaren Haber, PhD Candidate in UC Berkeley Sociology. 
Date: January 7th, 2018."""

import pandas as pd
#import numpy as np, re, os, csv
#from nltk.stem.porter import PorterStemmer # an approximate method of stemming words
#stemmer = PorterStemmer()

# FOR VISUALIZATIONS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns

# Visualization parameters
#% pylab inline 
#% matplotlib inline
matplotlib.style.use('ggplot')


# Define file paths:

dir_prefix = "/vol_b/data/" # For VM terminal
outfolder = dir_prefix + "Charter-school-identities/data/graphs/"
counts_file = dir_prefix + 'Charter-school-identities/data/charters_parsed_03-04_no-text_SMALL.csv'
bigfile = dir_prefix + 'Charter-school-identities/data/charters_parsed_03-08.csv'


# Define helper functions:

def convert_df(df):
    """Makes a Pandas DataFrame more memory-efficient through intelligent use of Pandas data types: 
    specifically, by storing columns with repetitive Python strings not with the object dtype for unique values 
    (entirely stored in memory) but as categoricals, which are represented by repeated integer values. This is a 
    net gain in memory when the reduced memory size of the category type outweighs the added memory cost of storing 
    one more thing. As such, this function checks the degree of redundancy for a given column before converting it.
    
    # TO DO: Filter out non-object columns, make that more efficient by downcasting numeric types using pd.to_numeric(), 
    merge  that with the converted object columns (see https://www.dataquest.io/blog/pandas-big-data/). 
    For now, since the current DF is ENTIRELY composed of object types, code is left as is. 
    But note that the current code will eliminate any non-object type columns."""
    
    converted_df = pd.DataFrame() # Initialize DF for memory-efficient storage of strings (object types)
    df_obj = df.select_dtypes(include=['object']).copy() # Filter to only those columns of object data type

    for col in df.columns: 
        if col in df_obj: 
            num_unique_values = len(df_obj[col].unique())
            num_total_values = len(df_obj[col])
            if (num_unique_values / num_total_values) < 0.5: # Only convert data types if at least half of values are duplicates
                converted_df.loc[:,col] = df[col].astype('category') # Store these columns as dtype "category"
            else: 
                converted_df.loc[:,col] = df[col]
        else:    
            converted_df.loc[:,col] = df[col]
                      
    converted_df.select_dtypes(include=['float']).apply(pd.to_numeric,downcast='float')
    converted_df.select_dtypes(include=['int']).apply(pd.to_numeric,downcast='signed')
    
    return converted_df


# Load data:

schooldf = pd.read_csv(counts_file, sep=",", low_memory=False, encoding="utf-8", na_values={"TITLEI":["M","N"]})
schooldf["PCTFRL"] = schooldf["TOTFRL"]/schooldf["MEMBER"] # Percent receiving free/ reduced-price lunch
schooldf["IDLEAN"] = schooldf["ess_strength"] - schooldf["prog_strength"]
schooldf = convert_df(schooldf)


# Draw and save plots:

with sns.axes_style("white"):
    #sns_hexplot = sns.jointplot(x="ess_strength", y="prog_strength", data=schooldf, marginal_kws=dict(bins=50), joint_kws=dict(bins=100), kind="hex", color="k", xlim=(0.0, 0.4), ylim=(0.0, 0.4)).set_axis_labels("Strength of traditionalist ideology", "Strength of progressivist ideology")
    #sns_jointplot = sns.jointplot(x="IDLEAN", y="PCTETH", data=schooldf, color="purple", marginal_kws=dict(bins=50), scatter_kws={"s": 10}, x_jitter=0.2, robust=True, kind="reg", xlim=(-0.4, 0.4), ylim=(0.0, 1.0)).set_axis_labels("Progressive ideology < > Traditional ideology", "Percent nonwhite students")
    #sns_lmplot1 = sns.lmplot(x="IDLEAN", y="PCTETH", col="PLACE", col_wrap=2, size=5, data=schooldf, x_jitter=0.2, robust=True, scatter_kws={"s": 10, "color": "brown"})
    #sns_lmplot2 = sns.lmplot(x="IDLEAN", y="PCTETH", hue="TITLEI", data=schooldf, x_jitter=0.2, robust=True, scatter_kws={"s": 10})
    
#sns_lmplot1.set(xlim=(-0.4, 0.4), ylim=(0.0, 1.0), xlabel="Progressive ideology < > Traditional ideology", ylabel="Percent nonwhite students")
#sns_lmplot2.set(xlim=(-0.4, 0.4), ylim=(0.0, 1.0), xlabel="Progressive ideology < > Traditional ideology", ylabel="Percent nonwhite students")
#sns_lmplot.set(xlim=(0.0, 1.0))
#lmaxes = sns_lmplot.axes
#lmaxes[0,0].set_xlim(-0.4, 0.4)
#lmaxes[0,0].set_ylim(0.0, 1.0)
#hexaxes = sns_hexplot.axes
#hexaxes[0,0].set_xlim(0.0, 0.5)
#hexaxes[0,0].set_ylim(0.0, 0.5)
#sns_hexplot.set(ylim=(0.0, 0.5))
#sns_hexplot.set(xlim=(0.0, 0.5))

#sns_hexplot.savefig(dir_prefix + "Charter-school-identities/data/graphs/ideology_hexplot_03-15.png")
#sns_jointplot.savefig(dir_prefix + "Charter-school-identities/data/graphs/ideology_jointplot_03-15.png")
#sns_lmplot1.savefig(dir_prefix + "Charter-school-identities/data/graphs/ideology_lmplot1_03-15.png")
#sns_lmplot2.savefig(dir_prefix + "Charter-school-identities/data/graphs/ideology_lmplot2_03-15.png")