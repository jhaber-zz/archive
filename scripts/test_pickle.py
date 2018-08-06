import pandas as pd
import os

pickle_path = "/vol_b/data/scrapy_cluster_data/processed_df.pkl"
df = pd.read_pickle(pickle_path)

# Load text from all pages into a single list
data = []
for school in df["data"]:
    for page in school:
        text = page[-1]
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\t", " ")
        data.append(text)
