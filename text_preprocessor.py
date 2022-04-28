from nltk.corpus import stopwords
import pandas as pd
from re import sub, findall


#DATASET:  https://www.kaggle.com/datasets/tysonpo/ice-cream-dataset


# words that dont have any sentiment
s_words = stopwords.words("english")
s_words.extend(["ive", "much","ice", "cream", "im", "dont", 
                "try", "people", "cookie", "ill", "get", ]) 


def textcleaner(text):
    """
    Normalizes text (removes punctuation and stopwords, strips, lowercase)
    
    Args:
        text (str): string to be cleaned
    Returns:
        Cleaned string
    
    """
    text = sub(r"[^\w\s]", "", text).lower().strip().split()
    return " ".join([word for word in text if word not in s_words])

def tokenizer(text):
    """
    Tokenizes a string
    
    Args:
        text (str): string to be tokenized
    Returns:
        Tokenized string 
    """
    return findall(r"\w+", text.lower().strip())

def process_df(fname, to_drop):
    """
    Processes a csv to be used in a classifier:
        - reads in a csv as a dataframe
        - drops useless columns
        - converts star ratings into 0s and 1s
        - combines everything back into one dataframe and drops na values
        - cleans each text review
    
    Args:
        fname (str): file to be read
        to_drop (list): list of columns to drop
    Returns:
        Cleaned dataframe
    """

    df = pd.read_csv(fname)
    df.drop(to_drop, inplace=True, axis=1)
    
    sentiment = pd.Series([0 if n <=2 else 1 for n in df["stars"]]) # convert ratings: rating < 3 == 0 (negative), >= 3 == 1 (positive)

    text = df["title"] + " " + df["text"] # combine title and text columns into a Series (makes sense because titles can have sentiment words in them)
    
    cleaned_df = pd.DataFrame({"sentiment": sentiment, 
                               "review": text})
    cleaned_df.dropna(inplace=True)
    
    cleaned_df = pd.DataFrame({"sentiment": cleaned_df["sentiment"],
                               "review":list(map(lambda x: textcleaner(x), list(cleaned_df["review"])))}) # clean each text review and put everything together
    
    return cleaned_df


