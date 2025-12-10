import convokit 
import lftk 
import matplotlib.pyplot as plt
import numpy as np

def load_and_filter_corpus(path=str, desired_posts=None):
    #get corpus loaded, and check if it needs to be cut filtered to desired post size.
    corpus = convokit.Corpus(filename=path)
    text = ''
    if desired_posts is None:
        text = 'No filtering applied to corpus.'
    
    if desired_posts is not None:
        filtered_corpus = corpus.get_utterance_dataframe() #pull off top 1000 posts
        #TODO make an option to get random posts, not first 1000
        text = f'Filtered corpus to first {desired_posts} posts.'
        
        
        
    return text, filtered_corpus
