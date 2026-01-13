import convokit # type: ignore
from convokit import Corpus, download # type: ignore
import lftk
from convokit.text_processing import TextParser
from convokit.text_processing import TextProcessor
import spacy 
import random
import pandas as PD
import nltk # type: ignore
nltk.download('punkt_tab')
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import statistics
from convokit.text_processing import TextParser # type: ignore
pos_features = lftk.search_features(domain='syntax', family="partofspeech", language="general",
return_format="list_dict")
pos_features = [f['key'] for f in pos_features]
additional_features = ["a_word_ps", "a_bry_ps", "corr_ttr"]
features_to_extract = pos_features + additional_features
    
    
    
def load_and_filter_corpus(path=str, desired_posts=None):
    
    
    #TODO add in the filtering for the number of desired posts
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

def features_to_csv(corpus,csv):
    #fill provided csv file with data. 
    pos_features = lftk.search_features(domain='syntax', family="partofspeech", language="general",
                                    return_format="list_dict")
    pos_features = [f['key'] for f in pos_features]
    additional_features = ["a_word_ps", "a_bry_ps", "corr_ttr"]
    features_to_extract = pos_features + additional_features
    
    utterances = corpus.get_utterances_dataframe()['text'].tolist()
    nlp = spacy.load("en_core_web_sm")
    # process utterances with spacy pipe
    processed_utterances = list(nlp.pipe(utterances))
    LFTK = lftk.Extractor(docs=processed_utterances)
    #initialized
    features_extracted = LFTK.extract(features=features_to_extract)
    #extracted
    
    dataframe = PD.DataFrame(features_extracted)
    dataframe.to_csv(csv)
    
    

def transform_corpus(corpus, csv:str):
    features = PD.read_csv(csv)

    features.index = corpus.get_utterances_dataframe().index


    for utt in corpus.iter_utterances():
        utt_id = utt.id
        features_utterance = features.loc[utt_id]
    
        for feature in features.columns:
            value = features_utterance[feature]
            # normalize if needed
            if feature.startswith("n_") and not feature.startswith("n_u"):
                value = value / utt.meta['num_tokens']
            elif feature.startswith("n_u"):
                corresponding_n_feature = feature.replace("n_u", "n_")
                n_value = features_utterance[corresponding_n_feature]
                if n_value > 0:
                    value = value / n_value
                else:
                    value = 0.0
        utt.meta[feature] = value
    return corpus
    


def analyze_differences(corpus1, corpus2):
    #analyze differences between two corpora, return most significant differences
    analysis_results = {'empty': 'to be implemented'}
    meansAndDevs = {'empty': 'to be implemented'}
    summary_results = {'empty': []}
    
    
    for feature in features_to_extract:
      corpus1_features = list(corpus1.get_utterances_dataframe()[f'meta.'+ feature])
    print(statistics.median(corpus1_features))
    #TODO figure out how to get the features into corpus file, look at socio part 1 or 2 
    #TODO create ratio of differences betweeen all features
    #TODO analysis results returns a dictionary of the score, with the name as the key, and a list like this [median of X value for corpus1, median corpus2, ratio between them]
    #TODO print a thing with the largest 3 differences
    
    print(f'largest 3 differences between corpora: {summary_results}')
    
    
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
    def countTokens(text):
        return len(text.split())

    numtokens = TextProcessor(proc_fn=countTokens, output_field='num_tokens')
    tokenized_corpus = numtokens.transform(corpus)
    return tokenized_corpus



print('Corpus ready to be analyszed, file loaded correctly')