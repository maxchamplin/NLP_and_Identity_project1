import convokit
import random
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


def analyze_differences(corpus1, corpus2):
    #analyze differences between two corpora, return most significant differences
    analysis_results = {}
    #TODO implement analysis logic here
    return analysis_results


def visualize_differences(analysis_results, corpus1 ,corpus2, differnces=dict):
    #visualize the differences found in analysis_results
    #TODO implement visualization logic here
    for diff in differnces:
        #TODO create visualizations for each difference
        try: 
            plt.boxplot([corpus1[diff], corpus2[diff]], labels=['Corpus 1', 'Corpus 2'],)
            
        except:
            print(f"Could not visualize difference for {diff} using boxplot.")
            continue
    pass
    