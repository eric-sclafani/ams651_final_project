from nltk.corpus import stopwords
import pandas as pd
from re import sub, findall

# words that dont have any sentiment
s_words = stopwords.words("english")
s_words.extend(["ive", "much","ice", "cream", "im", "dont", "try", "people", "cookie", "ill", "get"]) # add more stopwords to list


def textcleaner(text):
    # clean text and remove stop words (useless words with no meaning)
    text = sub(r"[^\w\s]", "", text).lower().strip().split()
    print(text)
    return " ".join([word for word in text if word not in s_words])

def tokenizer(text):
    return findall(r"\w+", text.lower().strip())

def process_df(fname):

    df = pd.read_csv(fname)
    df.drop(["author", "date", "helpful_yes", "helpful_no"], inplace=True, axis=1) # drop useless columns
    
    sentiment = pd.Series([0 if n <=2 else 1 for n in df["stars"]]) # convert ratings: rating < 3 == 0 (negative), >= 3 == 1 (positive)
    
    conversion = pd.DataFrame({"stars": df["stars"], # debugging purposes:shows correspondance between stars (1-5) and sentiment ratings (0,1)
                               "sentiment": sentiment})
    # conversion.to_csv("conversion_ratio.csv")

    text = df["title"] + " " + df["text"] # combine title and text columns into a Series (makes sense because titles can have sentiment words in them)
    
    cleaned_df = pd.DataFrame({"sentiment": sentiment, # combine into one df
                               "review": text})
    cleaned_df.dropna(inplace=True)
    
    cleaned_df = pd.DataFrame({"sentiment": cleaned_df["sentiment"],
                               "review":list(map(lambda x: textcleaner(x), list(cleaned_df["review"])))}) # clean each text review and put everything together
    
    return cleaned_df



# DATAFRAME DEBUGGING CODE

# df = process_df("ice_cream_reviews.csv")
# print(df.head(10))
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)