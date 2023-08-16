import pandas as pd
import numpy as np
import nltk
import unicodedata
import re
from nltk.corpus import stopwords
from requests import get
from bs4 import BeautifulSoup
import os


######################################### PREPARE #########################################


def basic_clean(text):
    '''This function takes in a string and makes it lowercase, normalizes unicode characters, and
        replaces anything that is not a letter, number, whitespace or a single quote.'''
    
    #lowercase
    text = text.lower()
    
    #normalize unicode characters
    text = unicodedata.normalize('NKFD', text).encode('ascii', 'ignore').decode('utf-8')
    
    #replace anything not a letter, number, whitespace, or single quote
    text = re.sub(r'[^a-z0-9\s]','', text)
    
    return text


def tokenize(text):
    '''This function takes in a string and returns it tokenized.'''
    
    #create the tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    
    #tokenize the text
    text = tokenize.tokenize(text)
    
    return text


def stem(text):
    '''This function takes in a string and returns it after applying the Porter Stemmer.'''
    
    #create the stemmer
    ps = nltk.porter.PorterStemmer()
    
    #apply the stemmer to all words in the string
    text = [ps.stem(word) for word in text]
    
    return text


def lemmatize(text):
    '''This function takes in a string and returns it after applying WordNet Lemmatizer.'''
    
    #create the lemmatizer
    wnl = ntlk.stem.WordNetLemmatizer()
    
    #apply the lemmatizer
    text = [wnl.lemmatize(word) for word in text]
    
    return text


def remove_stopwords(text, extra_words=None, exclude_words=None):
    '''This function takes in a string and returns the text after removing all stopwords using nltk stopwords list.
    There are two optional arguments: extra_words add any additional stop words to the list,
                                      exclude_words remove words from the stop list so they will not be removed. '''
    
    stopwords = stopwords.words('english')
    
    if extra_words != None:
        
        stopwords = stopwords.append(extra_words)
        
    if exclude_words != None:
        
        stopwords = stopwords.remove(exclude_words)
        
    text = [word for word in text if word not in stopwords]
    
    return text