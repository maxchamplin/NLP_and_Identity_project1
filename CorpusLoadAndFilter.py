import convokit 
import lftk 
import matplotlib.pyplot as plt
import numpy as np

def load_and_filter_corpus(path=str, desired_posts=None):
    #get corpus loaded, and check if it needs to be cut filtered to desired post size.
    corpus = convokit.Corpus(filename=path)
