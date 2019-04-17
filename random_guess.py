import numpy as np
import pickle

def get_metrics(TP,FP,TN,FN):
    assert (TP+FP+TN+FN) == n_labels
    acc = (TP + TN) / float(n_labels)
    print "Acc: ", acc
    print "True Positive: ", TP
    print "False Positive: ", FP
    print "True Negative: ", TN
    print "False Negative: ", FN
    recall = TP / float(TP + FN)
    print "Recall:", recall
    F1 = acc * recall * 2 / (acc + recall)
    print "F1:", F1

with open("/home/luban/cas/test_labels.pkl","rb") as f:
    test_labels = pickle.load(f)
test_labels = test_labels.flatten()
n_labels = len(test_labels)
results = np.random.randint(2, size=n_labels)

# Metrics
TP = np.sum(results + test_labels == 2)
TN = np.sum(results + test_labels == 0)
FP = np.sum(results - test_labels == 1)
FN = np.sum(results - test_labels == -1)

get_metrics(TP,FP,TN,FN)


# All_0 Metrics
print "==============All 0 Model=============="
TP = 0
TN = len(test_labels[test_labels == 0])
FP = 0
FN = n_labels - TN
get_metrics(TP,FP,TN,FN)

# All_1 Metrics
print "==============All 1 Model=============="
TP = len(test_labels[test_labels == 1])
TN = 0
FP = n_labels - TP
FN = 0
get_metrics(TP,FP,TN,FN)