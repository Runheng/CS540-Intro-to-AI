import os
from numpy import log

def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    file = open(filepath, 'r', encoding='utf-8')
    line = file.read().splitlines()
    words = []
    for word in line:
        words.append(word)
    for word in words:
        if word in bow:
            bow[word] += 1
        elif word in vocab:
            bow[word] = 1
        elif None in bow:
            bow[None] += 1
        else:
            bow[None] = 1
    file.close()
    return bow

def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []
    file_in_16 = os.listdir(directory + '2016/')
    file_in_20 = os.listdir(directory + '2020/')    
    
    for i in range(0, len(file_in_20)):  
        bow = {}
        bow['label'] = '2020'
        bow['bow'] = create_bow(vocab, directory + '2020/' + file_in_20[i])
        dataset.append(bow)

    for i in range(0, len(file_in_16)):
        bow = {}
        bow['label'] = '2016'
        bow['bow'] = create_bow(vocab, directory + '2016/' + file_in_16[i])
        dataset.append(bow)
    return dataset

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    vocab = []
    file_in_16 = os.listdir(directory+'2016/')
    file_in_20 = os.listdir(directory+'2020/')
    words = []
    # read each line in each file, and save each word to words[]
    for i in range(0, len(file_in_16)):
        cur = open(directory+ '2016/' + file_in_16[i], 'r', encoding='utf-8')
        line = cur.read().splitlines()
        for word in line:
            words.append(word)
        cur.close()
    for i in range(0, len(file_in_20)):
        cur = open(directory + '2020/' + file_in_20[i], 'r', encoding='utf-8')
        line = cur.read().splitlines()
        for word in line:
            words.append(word)
        cur.close()
    # sort all the words and add those pass cutoff to vocab list
    words = sorted(words)    
    i = 0
    while i < len(words) + 1 - cutoff:
        if words[i] == words[i + cutoff - 1] and words[i] not in vocab:
            vocab.append(words[i])
            i = i + cutoff
        else:
            i += 1
    return vocab

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    label_16 = 0
    label_20 = 0
    # count total num of files with label 2016 or 2020
    for data in training_data:
        if data['label'] == '2016':
            label_16 = label_16 + 1
        elif data['label'] == '2020':
            label_20 = label_20 + 1
    # compute the possiblity, P = (N+1)/(total + 2), then take the natural log
    P_16 = (label_16 + 1)/(label_16 + label_20 + 2)
    P_20 = (label_20 + 1)/(label_16 + label_20 + 2)
    logprob['2020'] = log(P_20)
    logprob['2016'] = log(P_16)
    return logprob

def p_label_helper(cur_bow, vocab_dict, total):
    '''This function severs as the helper to the function p_word_given_label
      It updates the number of each word showing up in training data
      and increment the total number of words in the training data as well'''
    for word, n_w in cur_bow.items():
        if word in vocab_dict:
            vocab_dict[word] = vocab_dict[word] + n_w
        else:
            vocab_dict[None] = vocab_dict[None] + n_w
        total = total + n_w
    return vocab_dict, total

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    
    ''' the formula for computing prob of a word with given label is found on piazza '''
    smooth = 1 # smoothing factor
    word_prob = {}
    vocab_with_label = {}
    # initialize all counts to zero
    for word in vocab:
        vocab_with_label[word] = 0 
    vocab_with_label[None] = 0
    # count num of words with given label
    n = 0
    for data in training_data:
        if data['label'] == label:
            vocab_with_label, n = p_label_helper(data['bow'], vocab_with_label, n)
    for word, n_w in vocab_with_label.items():
        p = (n_w + smooth)/(n + smooth*(len(vocab)+1))
        word_prob[word] = log(p)
    return word_prob

    
##################################################################################
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    vocab = create_vocabulary(training_directory, cutoff)
    retval['vocabulary'] = vocab
    training_data = load_training_data(vocab, training_directory)
    log_prob = prior(training_data, ['2020','2016'])
    retval['log prior'] = log_prob
    word_prob_16 = p_word_given_label(vocab, training_data, '2016')
    word_prob_20 = p_word_given_label(vocab, training_data, '2020')
    retval['log p(w|y=2016)'] = word_prob_16
    retval['log p(w|y=2020)'] = word_prob_20
    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>, 
             'log p(y=2016|x)': <log probability of 2016 label for the document>, 
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    p_label16 = model['log prior']['2016']
    p_label20 = model['log prior']['2020']
    # Go through the given .txt file word by word
    cur_bow = create_bow(model['vocabulary'], filepath)
    # Add the conditional probability of that word to the prior probability
    for word, n_w in cur_bow.items():
        p_label16 += model['log p(w|y=2016)'][word] * n_w
        p_label20 += model['log p(w|y=2020)'][word] * n_w
    # The total sum for each year is then what is used to make the prediction.
    if p_label16 > p_label20:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'
    retval['log p(y=2016|x)'] = p_label16
    retval['log p(y=2020|x)'] = p_label20
    return retval
