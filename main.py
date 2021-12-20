from nltk.corpus import treebank
import time


# data pre-processing - split the raw data into training and testing data.
# replace the words of which the frequency is less than or equal to 1 in the training data with 'NONE'
# replace the words that have not appeared in the training data with 'NONE', and so do the tags
def pre_processing(raw_tagged_sents):
    tagged_sents = []
    for raw_sent in raw_tagged_sents:
        sent = []
        for w_t in raw_sent:
            sent.append((w_t[0].lower(), w_t[1]))
        tagged_sents.append(sent)

    train_rate = 0.8
    train_idx = round(train_rate * len(tagged_sents))
    train_sents = tagged_sents[0:train_idx]
    test_sents = tagged_sents[train_idx:]

    words_set = []
    for s in train_sents:
        for w_t in s:
            words_set.append(w_t[0])
    freq_dict = {}
    for w in words_set:
        freq_dict[w] = freq_dict.get(w, 0) + 1

    train_data, train_words = [], []
    for s in train_sents:
        sent = []
        for w_t in s:
            if freq_dict[w_t[0]] <= 1:
                sent.append(('UNK', w_t[1]))
                train_words.append('UNK')
            else:
                sent.append(w_t)
                train_words.append(w_t[0])
        train_data.append(sent)
    train_words = set(train_words)

    test_data = []
    for s in test_sents:
        sent = []
        for w_t in s:
            if w_t[0] not in train_words:
                sent.append(('UNK', w_t[1]))
            else:
                sent.append(w_t)
        test_data.append(sent)

    return train_data, test_data


# get all the training tags (no repetition)
def get_all_tags(train_data):
    tags = []
    for s in train_data:
        for w_t in s:
            if w_t[1] not in tags:
                tags.append(w_t[1])
    return tags


# get the initial vector {tag: prob} of the tags based on the maximum likelihood
def get_init_vector(train_data, tags):
    init_matrix = {key: 0 for key in tags}
    for sent in train_data:
        init_matrix[sent[0][1]] += 1
    sents_num = len(train_data)
    for tag, freq in init_matrix.items():
        init_matrix[tag] /= sents_num
    return init_matrix


# get the transition matrix of tags {tag_1: {tag_2: prob}} based on the maximum likelihood
def get_trans_matrix(train_data, tags):
    trans_matrix = {key: {k: 0 for k in tags} for key in tags}
    for sent in train_data:
        for idx in range(len(sent)):
            if idx != len(sent) - 1:
                cur_tag = sent[idx][1]
                next_tag = sent[idx + 1][1]
                trans_matrix[cur_tag][next_tag] += 1
    for key, value in trans_matrix.items():
        s = sum(value.values())
        if s:
            for k, v in trans_matrix[key].items():
                trans_matrix[key][k] /= s
    return trans_matrix


# get the emission matrix {tag: {word: prob}} based on the maximum likelihood
def get_emission_matrix(train_data, tags):
    words = []
    for s in train_data:
        for w in s:
            if w[0] not in words:
                words.append(w[0])
    emission_matrix = {t: {w: 0 for w in words} for t in tags}
    for s in train_data:
        for w_t in s:
            emission_matrix[w_t[1]][w_t[0]] += 1

    for key, value in emission_matrix.items():
        s = sum(emission_matrix[key].values())
        if s:
            for k, v in emission_matrix[key].items():
                emission_matrix[key][k] /= s
    return emission_matrix


# find the best path using viterbi algorithm
def viterbi(test_data, init_vector, trans_matrix, emission_matrix):
    pred = []
    for sent in test_data:
        # forward
        prob_matrix = [{tag: 0 for tag, value in init_vector.items()} for x in sent]  # for each word: {tag: prob}
        for idx, word in enumerate(sent):
            # get the emission vector of the word
            emission_vector = {}
            for tag, em_prob in emission_matrix.items():
                emission_vector[tag] = em_prob[word]
            if idx == 0:  # initialization
                prob_matrix[idx] = {t: init_vector[t] * emission_vector[t] for t, v in init_vector.items()}
            else:
                for t, v in prob_matrix[idx].items():
                    value_set = []
                    for t_, v_ in prob_matrix[idx - 1].items():
                        a = v_
                        b = trans_matrix[t_][t]
                        c = emission_vector[t]
                        value_set.append(a * b * c)
                    prob_matrix[idx][t] = max(value_set)
        # backward traverse
        p = []
        for prob in prob_matrix:
            p.append(max(prob, key=prob.get))
        pred.append(p)
    return pred


# get the accuracy of every testing sentence
def cal_accuracy(test_labels, pred):
    acc = []
    for idx in range(len(test_labels)):
        t, p = test_labels[idx], pred[idx]
        acc_per_sent = 0
        for i in range(len(t)):
            if t[i] == p[i]:
                acc_per_sent += 1
        acc_per_sent /= len(t)
        acc.append(acc_per_sent)
    return acc


if __name__ == '__main__':
    # retrieve the raw data
    raw_tagged_sents = treebank.tagged_sents()
    train_data, test_data = pre_processing(raw_tagged_sents)
    tags = get_all_tags(train_data)
    test_labels = get_all_tags(test_data)

    # split testing data into sentences and labels
    test_data, test_labels = [[x[0] for x in s] for s in test_data], [[x[1] for x in s] for s in test_data]
    tags = get_all_tags(train_data)

    # training
    start = time.time()
    init_vector = get_init_vector(train_data, tags)
    trans_matrix = get_trans_matrix(train_data, tags)
    emission_matrix = get_emission_matrix(train_data, tags)
    end = time.time()
    print(f'Training ends. Takes {end-start:.2f}s.')

    # testing
    start = time.time()
    pred_viterbi = viterbi(test_data, init_vector, trans_matrix, emission_matrix)
    end = time.time()
    print(f'Testing ends. Takes {end - start:.2f}s.')

    # evaluation
    acc = cal_accuracy(test_labels, pred_viterbi)
    print(f'The testing accuracy is {sum(acc) / len(acc):.4f}.')
