import numpy as np
import pandas as pd
import pickle

with open('/nfs/private/cas/dataset_0_N/week/test_features.pkl','rb') as f:
    test_features = pickle.load(f)
with open('/nfs/private/cas/dataset_0_N/week/test_labels.pkl','rb') as f:
    test_labels = pickle.load(f)

def gen_adjacent(element, num=5):
    '''
        Util Function to generate rank5 results
    '''
    if element < 0:
        raise ValueError
    elif element < (num/2):
        return list(range(num))
    else:
        return list(range(element-num/2,element+num/2+1))


def get_top_k(array, k=5):
    '''
        Util Function to get top k elements in array
        If length of unique elements do not satisfy k length,
        pad max+1 until results' length meet k
    '''
    values, counts = np.unique(array,return_counts=True)
    ids = counts.argsort()[-5:][::-1]
    results = values[ids]
    while len(results) < 5:
        results = np.append(results, np.max(results)+1)
    return results

# def AVG_Freq(features):
#     return np.round(np.mean(features, axis=1))

# def All_0(features):
#     return np.zeros(features.shape[0])

# def Random_Guess(features):
#     return np.random.randint(np.max(features), size=features.shape[0])

def Random_Guess(features):
    gen_arrays = np.repeat([range(int(np.max(features)))], features.shape[0], axis=0)
    shuffle = np.vectorize(np.random.permutation, signature='(n)->(n)')
    predict_results = shuffle(gen_arrays)
    return predict_results[:,:5]

def AVG_Freq(features):
    gen_arrays = np.round(np.mean(features, axis=1))
    gen_arrays = gen_arrays.astype(int)
    return np.array([gen_adjacent(x) for x in gen_arrays])  

def TOP_Freq(features):
    sort_features = np.sort(features)
    return np.array([get_top_k(x) for x in sort_features])

def All_0(features):
    return np.repeat([range(5)], features.shape[0], axis=0)

baseline_models = [AVG_Freq, All_0, Random_Guess, TOP_Freq]

for baseline in baseline_models:
    predict_labels = baseline(test_features)
    predict_labels_top = predict_labels[:,0]
    acc = np.mean(np.equal(predict_labels_top, np.squeeze(test_labels)))
    diff = np.mean(np.abs(predict_labels_top - np.squeeze(test_labels)))
    acc_rank5 = np.sum(np.equal(predict_labels, test_labels)) / float(len(test_labels))
    print "=======%s Baseline Model=======" % baseline.__name__
    print "       Accurancy(rank@1): %f" % acc
    print "       Accurancy(rank@5): %f" % acc_rank5
    print "       Different: %f" % diff

# rg_results = Random_Guess_rank5(test_features)
# acc_rank5 = np.sum(np.equal(rg_results,test_labels)) / float(len(test_labels))
# print "=======Random Guess Baseline Model======="
# print "       Accurancy(rank@5): %f" % acc_rank5