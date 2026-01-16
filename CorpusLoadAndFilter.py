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
    
    
    
def load_and_filter_corpus(path:str, desired_posts:None):
    
    
    #TODO add in the filtering for the number of desired posts
    parser = TextParser(input_field='clean_text', verbosity=50,mode='tokenize')

    if desired_posts != None:
        prefilter_num = desired_posts*2
        pre_filtered_corpus = Corpus(filename=path, utterance_start_index=0, utterance_end_index=prefilter_num)
        tokened = tokenize(pre_filtered_corpus)
        
        filtered_corpus = Corpus(filename=path, utterance_start_index=0, utterance_end_index=desired_posts)
    
    else: 
        
        
        filtered_corpus = Corpus(filename=path)
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
    additional_features = ["a_word_ps", "a_bry_ps", "corr_ttr","fkgl","fkre"]
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
    
    


def analyze_differences(corpus1, corpus2):
    features = list(corpus1.get_utterances_dataframe().columns)
    for feature in features:
        
        
        if 'meta.n' in feature or 'meta.f' in feature:

            test_feature = []
            
            test_feature.append(list(corpus1.get_utterances_dataframe()[feature]))

            for x in range(len(test_feature)):
                Corpus1_median = test_feature[x]
                Corpus1_median = [float(val) for val in Corpus1_median]
                
                
            Corpus1_median_val = statistics.median(Corpus1_median)

            test_feature2 = []
            test_feature2.append(list(corpus2.get_utterances_dataframe()[feature]))

            for x in range(len(test_feature2)):
                Corpus2_median = test_feature2[x]
                Corpus2_median = [float(val) for val in Corpus2_median]
                
                
            Corpus2_median_val = statistics.median(Corpus2_median)
            
            
            if Corpus1_median_val == 0  or Corpus2_median_val == 0 or Corpus1_median_val == 'nan'  or Corpus2_median_val == 'nan' :
                pass
            else:
                try:
                    print('___________','\n',f'item is {feature}','\n','Corpus 1 median value is:',Corpus1_median_val,'\n','Corpus 2 median is:',Corpus2_median_val,'\n','ratio between them is:',Corpus1_median_val/Corpus2_median_val)
                except:
                    print('###########','\n',f'attempted to divide by 0 for: {feature} median')
        else:
            pass

    
    


def visualize_differences(corpus1 ,corpus2, feature:str,corpus1_name:str,corpus2_name:str):
    features = list(corpus1.get_utterances_dataframe().columns)
    if feature in features: 
        pass
    else:
        raise Exception("input feature not in data")
    
            
    corpus1_values = corpus1.get_utterances_dataframe()[feature].astype(float)
    
            
    corpus2_values = corpus2.get_utterances_dataframe()[feature].astype(float)

    corpus1_mean = statistics.mean(corpus1_values)
    corpus2_mean = statistics.mean(corpus2_values)
    plt.figure(figsize=(8,6))
    plt.violinplot([corpus1_values,corpus2_values],showmeans=True)
    plt.xticks([1, 2], [corpus1_name, corpus2_name])
    plt.ylabel(feature)
    plt.title(f'Violin Plot of {feature}  for {corpus1_name} (mean: {corpus1_mean}) and {corpus2_name} (mean: {corpus2_mean})')
    plt.show()
    
    
def tokenize(corpus):
    def countTokens(text):
        return len(text.split())

    numtokens = TextProcessor(proc_fn=countTokens, output_field='num_tokens')
    tokenized_corpus = numtokens.transform(corpus)
    return tokenized_corpus



print('Corpus ready to be analyszed, file loaded correctly')