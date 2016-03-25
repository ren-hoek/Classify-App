from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.tokenize import word_tokenize
from nltk.metrics import ConfusionMatrix
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import bigrams
import math
import re
import pandas as pd
import numpy as np
from random import randint
from random import shuffle

class OccDict:
    def __init__(self, occ_data):
        self.no_records = len(occ_data.value_counts())
        distinct_occ = occ_data.value_counts().to_frame(name = 'count').sort_index()
        self.count = distinct_occ
        distinct_occ = distinct_occ.drop('count', 1)
        distinct_occ['occ'] = distinct_occ.index
        distinct_occ['id'] = distinct_occ['occ'].rank() - 1
        distinct_occ = distinct_occ.set_index(['id'])
        self.dict = distinct_occ.to_dict()['occ']


class BaggingClassifier:
    def __init__(self,train_set, test_set, class_dict, no_of_boots):
        bagged_classifier = []
        train_len = len(train_set)
        test_len = len(test_set)
        for i in xrange(no_of_boots):
            if no_of_boots != 1:
                train_boot = (
                    [train_set[randint(0, train_len - 1)]
                    for x in xrange(train_len)]
                )
            else:
                train_boot = train_set
            classifier = NaiveBayesClassifier.train(train_boot)
            bagged_classifier.append(classifier)
            test_classify = pd.DataFrame(
                [(test_set[x][1], classifier.classify(test_set[x][0]))
                for x in xrange(test_len)]
            )
            test_classify['ind'] = test_classify.index

            if i == 0:
                boot_samples = test_classify.values.tolist()
            else:
                [boot_samples.append(x) for x in test_classify.values.tolist()]

        boot_data = pd.DataFrame(boot_samples)
        boot_data.columns = ['act', 'pred', 'test_index']

        boot_data['cnt'] = 1
        boot_result = boot_data.groupby(['test_index', 'act', 'pred']).count()
        boot_result.reset_index(inplace=True)
        boot_result = boot_result.sort(
                ['test_index', 'cnt'], ascending=[1,0]
        ).groupby('test_index').first()
        self.model = bagged_classifier
        self.data = boot_result
        self.confusion_matrix = ConMatrix(
                boot_result['act'], boot_result['pred'], class_dict)


class ConMatrix:
    def __init__(self, actual, predict, group_dict):
        self.cm = ConfusionMatrix(actual.tolist(), predict.tolist())
        con_matrix_np = np.empty([group_dict.no_records, group_dict.no_records])
        for x in xrange(group_dict.no_records):
            for y in xrange(group_dict.no_records):
                con_matrix_np[x, y] = self.cm[group_dict.dict[x], group_dict.dict[y]]

        column_sum = np.sum(con_matrix_np, axis = 0)

        self.cm_np = con_matrix_np
        self.accuracy = np.trace(con_matrix_np)/np.sum(con_matrix_np)
        self.cm_column = con_matrix_np / column_sum[:, None]

def restrict_occ(occ, percent):
    occ_group_by = occ.value_counts().to_frame(name = 'cnt')
    occ_group_by['cum_sum'] = occ_group_by.cnt.cumsum()
    occ_group_by['cum_perc'] = occ_group_by.cum_sum / occ_group_by.cnt.sum()
    occ_restrict = occ_group_by[occ_group_by.cum_perc < percent]
    selected_occ = []
    [selected_occ.append(occ_restrict.index[x]) for x in xrange(len(occ_restrict))]
    return occ.map(lambda d: d if (d in selected_occ) else 9999)

def clean_occ(text):
    clean_text = text.lower()
    clean_text = re.sub('[^0-9a-zA-Z //]+', '', clean_text)
    clean_text = re.sub('/', ' ', clean_text)
    clean_text = re.sub('n k', 'nk', clean_text)
    clean_text = re.sub(' +', ' ', clean_text)
    return clean_text.strip()

def bag_of_words(text, use_bigram=True):
    token_text = text.split()
    token_dictionary = dict((word, True) for word in token_text)
    dictionary = token_dictionary.copy()
    if use_bigram is True:
        bigram_dictionary = dict((bigram, True) for bigram in bigrams(token_text))
        dictionary.update(bigram_dictionary)
    return dictionary

# Functions to produce agent, process and soc lookup dictionaries

def agent_fix(agent_code):
    agent_text = str(agent_code)
    if len(agent_text) == 5:
        return agent_text
    elif len(agent_text) == 3:
        return '0' + agent_text + '0'
    elif agent_code < 10 and len(agent_text) == 4:
        return '0' + agent_text
    else:
        return agent_text + '0'

def agent_lk_dict(lookup_file='agent.txt'):
    lookup_data = pd.read_csv(lookup_file)
    lookup_data.agent = lookup_data.AGENT * 100
    lookup_data.agent = lookup_data.agent.round()/100
    lookup_data.agent = lookup_data.agent.apply(lambda d: agent_fix(d))
    return dict(zip(lookup_data.agent, lookup_data.REF_DESC))

def process_lk_dict(lookup_file='process.txt'):
    lookup_data = pd.read_csv(lookup_file)
    return dict(zip(lookup_data.PROC_ENV, lookup_data.REF_DESC))

def occ_lk_dict(lookup_file='occ.txt'):
    lookup_data = pd.read_csv(lookup_file)
    return dict(zip(lookup_data.EM_SOC2000, lookup_data.REF_DESC))

# functions to lemmatize and bag of words the accident description

def pos_lookup(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def bag_of_words_pos(text, stop_words):

    token_text = word_tokenize(text)
    wordnet_lemmatizer = WordNetLemmatizer()
    pos = pos_tag(token_text)
    pos_ex =  [w for w in pos if w[0] not in stop_words]
    pos_word = (
        [wordnet_lemmatizer.lemmatize(y[0], pos_lookup(y[1])) for y in pos_ex]
    )
    pos_word_dist = list(set(pos_word))
    token_dictionary = dict((word, True) for word in pos_word_dist)
    return token_dictionary
