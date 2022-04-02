import numpy as np
import pandas as pd
import re

unprocessed_df = pd.read_csv("ice_cream_reviews.csv") # preserve original df

def process_df(df):
    
    # drop useless columns
    df.drop(["author", "date", "helpful_yes", "helpful_no"], inplace=True, axis=1)
    
    # drop messed up rows 
    # some are incorrectly formatted. The ones that dont start with a key are a part of the review above it.
    # would like to fix all of them, but there's over 8k entries.
    df = df[df["key"].str.contains(r"\d+_bj", regex=True)] 
    
    # convert ratings: rating < 3 == 0 (negative), >= 3 == 1 (positive)
    ratings = pd.Series([0 if n < 3 else 1 for n in df["stars"]])

    # combine title and text columns (make sense because titles can have sentiment words in them)
    text = df["title"] + " " + df["text"]
    
    # putting it all together
    cleaned_df = pd.DataFrame({"rating": ratings,
                               "review": text
                              })
    
    return cleaned_df
