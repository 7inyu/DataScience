__author__ = 'Qinyu'

from math import log
import pprint

# Attributes set
Attributes = [i for i in range(9)]

# train_set
train_file = open("/Users/Qinyu/Box Sync/Cornell/2014FALL/Courses/Enrolled/CS 5780/Assignment 2/data/problem_2/bcan.train")
train_set = []

for line in train_file:
    temp = line.split(' ')
    entry = []
    for j in range(9):
        entry.append(int(temp[j+1].split(':')[1]))
    entry.append(temp[0])
    train_set.append(entry)

# validate set
validate_file = open("/Users/Qinyu/Box Sync/Cornell/2014FALL/Courses/Enrolled/CS 5780/Assignment 2/data/problem_2/bcan.validate")
validate_set = []

for line in validate_file:
    temp = line.split(' ')
    entry = []
    for j in range(9):
        entry.append(int(temp[j+1].split(':')[1]))
    entry.append(temp[0])
    validate_set.append(entry)

# test_set
test_file = open("/Users/Qinyu/Box Sync/Cornell/2014FALL/Courses/Enrolled/CS 5780/Assignment 2/data/problem_2/bcan.test")
test_set = []

for line in test_file:
    temp = line.split(' ')
    entry = []
    for j in range(9):
        entry.append(int(temp[j+1].split(':')[1]))
    entry.append(temp[0])
    test_set.append(entry)

# count label M and B of a dataset
def count(dataset):

    labels = [entry[-1] for entry in dataset]
    M_count = 0

    for label in labels:
        if label == 'M':
            M_count += 1

    B_count = len(dataset) - M_count

    return M_count, B_count

# calculate entropy of a dataset
def entropy(dataset):

    M_count, B_count = count(dataset)

    pM = float(M_count)/len(dataset)
    pB = 1 - pM

    if pM == 1 or pB == 1:
        entropy = 0
    else:
        entropy = -pM*log(pM) - pB*log(pB)

    return entropy


# divide dataset according to i and j
def divData(dataset, i, j):

    small_set = []
    big_set = []

    for entry in dataset:
        if entry[i] <= j:
            small_set.append(entry)
        else:
            big_set.append(entry)

    return small_set, big_set


# calculate info gain on dataset dividing by Ai<=j
def infoGain(dataset, i, j):

    small_set, big_set = divData(dataset, i, j)

    s_len = len(small_set)

    ps = float(s_len)/len(dataset)
    pb = 1 - ps

    if ps == 0 or pb == 0:
        info_gain = 0
    else:
        info_gain = entropy(dataset) - ps*entropy(small_set) - pb*entropy(big_set)

    return info_gain


'''try all possible values j for every attribute i and pick the most suitable attribute-value combination
according to the splitting criterion. If there are multiple attribute-value pairs that are equally good,
pick the attribute with the lowest i value and then the value with the smallest j among those combinations.'''


def getBest(dataset, Attributes):

    max_gain = 0

    for i in Attributes:
        for j in range(1, 10):
            info_gain = infoGain(dataset, i, j)
            if info_gain > max_gain:
                best_a = i
                best_j = j
                max_gain = info_gain

    return best_a, best_j


'''build tree using ID3 algorithm.
For impure leaf nodes, use the majority class as a node label.
If there is a tie, say that the node is malignant (M).'''


def ID3(dataset, Attributes):

    M_count, B_count = count(dataset)

    if M_count == len(dataset):
        return 'M'

    if B_count == len(dataset):
        return 'B'

    if len(Attributes) == 0:
        if M_count >= B_count:
            return 'M'
        else:
            return 'B'

    best_a, best_j = getBest(dataset, Attributes)

    tree = {best_a+1: {}}

    small_set, big_set = divData(dataset, best_a, best_j)

    s_attr = []
    for i in Attributes:
        if i != best_a:
            s_attr.append(i)
    tree[best_a+1]['<='+str(best_j)] = ID3(small_set, s_attr)

    b_attr = []
    for i in Attributes:
        if i != best_a:
            b_attr.append(i)
    tree[best_a+1]['>'+str(best_j)] = ID3(big_set, b_attr)

    return tree


# return depth of a tree
def depth(tree):
    d = 1
    root = tree.keys()[0]
    children = []
    for key in tree[root]:
        if tree[root][key] != 'M' and tree[root][key] != 'B':
            children.append(key)
    if len(children) != 0:
        d = d + max([depth(tree[root][key]) for key in children])
    else:
        return 2

    return d


def numOfNodes(tree):
    n = 1
    root = tree.keys()[0]
    children = []
    for key in tree[root]:
        if tree[root][key] == 'M' or tree[root][key] == 'B':
            n += 1
        else:
            children.append(key)
    if len(children) != 0:
        n = n + sum([numOfNodes(tree[root][key]) for key in children])

    return n


def numOfLeaf(tree):
    l = 0
    root = tree.keys()[0]
    children = []
    for key in tree[root]:
        if tree[root][key] == 'M' or tree[root][key] == 'B':
            l += 1
        else:
            children.append(key)
    if len(children) != 0:
        l = l + sum([numOfLeaf(tree[root][key]) for key in children])

    return l


printer = pprint.PrettyPrinter()

printer.pprint(ID3(train_set, Attributes))


