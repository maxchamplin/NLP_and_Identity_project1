import convokit # type: ignore
from convokit import Corpus, download # type: ignore




import random
import nltk # type: ignore
nltk.download('punkt_tab')
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

from convokit.text_processing import TextParser # type: ignore

def load_and_filter_corpus(path=str, desired_posts=None):
    
    
    
    parser = TextParser(input_field='clean_text', verbosity=50,mode='tokenize')


    filtered_corpus = parser.transform(convokit.Corpus(filename=path))
    
    
    print('Corpus loaded and tokenized.')
        
        
    return  filtered_corpus



def load_and_filter_corpus_dataframe(path=str, desired_posts=None):
    #get corpus loaded, and check if it needs to be cut filtered to desired post size.
    corpus = convokit.Corpus(filename=path)
    text = ''
    if desired_posts is None:
        text = 'No filtering applied to corpus.'
        return corpus.get_utterances_dataframe()
    if desired_posts is not None:
        
        filtered_corpus = corpus.get_utterances_dataframe().head(desired_posts) #pull off top X posts
        #TODO make an option to get random posts, not first X
        text = f'Filtered corpus to first {desired_posts} posts.'
    
    print(text)
        
        
    return  filtered_corpus

def analyze_differences(corpus1, corpus2):
    #analyze differences between two corpora, return most significant differences
    analysis_results = {'empty': 'to be implemented'}
    meansAndDevs = {'empty': 'to be implemented'}
    summary_results = {'empty': 'to be implemented'}
    #TODO figure out how to get the features into corpus file, look at socio part 1 or 2 
    #TODO create ratio of differences betweeen all features
    #TODO analysis results returns a dictionary of the score, with the name as the key, and a list like this [median of X value for corpus1, median corpus2, ratio between them]
    #TODO print a thing with the largest 3 differences
    
    print(f'largeest 3 differences between corpora: {summary_results}')
    
    
    return analysis_results, meansAndDevs


def visualize_differences(title, corpus1 ,corpus2, differnces=dict):
    #visualize the differences found in analysis_results
    #TODO implement visualization logic here
    for diff in differnces:
        #TODO create visualizations for each difference
        try: 
            
            plt.boxplot([corpus1[diff], corpus2[diff]], labels=['Corpus 1', 'Corpus 2'], title=title,)
            
        except:
            print(f"Could not visualize difference for {diff} using boxplot.")
            continue
    pass
    


def tokenize(corpus):
    tokenizer= TextParser(input_field='clean_text', verbosity=50,mode='tokenize')
    tokenitzed_corpus = tokenizer.transform(corpus)
    return tokenitzed_corpus


print('Corpus ready to be analyszed, file loaded correctly')