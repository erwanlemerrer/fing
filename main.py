import networkx as nx
import numpy as np
import copy
import bisect
 
import sys
 
''' Returns the list of sorted unique positive labels of G
'''
def unique_labels(G):
    def sum_incident_edges_weights(G, nodes):
        s = 0
        for i in nodes:
            for j in G.neighbors(i):
                s += G[i][j]['weight']
        return s
 
    incidence = {}
    for n in G.nodes():
        incidence[n] = sum_incident_edges_weights(G, [n])
 
    from collections import Counter
    counts = Counter(incidence.values())
 
    nx.set_node_attributes(G, incidence, "label")
    V_hat = np.sort([x for x in list(set(list(incidence.values())))
                      if ~np.isnan(x)])
    # positive weights are a condition of the original factor-r problem
    V_hat = list(filter(lambda x: x > 0, V_hat))
 
    return G, V_hat, counts
 
 
def find(S, delta, j):
    # Labels MUST be sorted
    assert all(S[i] <= S[i+1] for i in range(len(S)-1))
 
    left = bisect.bisect_left(S, j/delta)
    right = bisect.bisect_right(S, j*delta)
 
    if len(S[left:right]) > 0:
        elt = S[left:right][0]
        S.remove(elt)
        return elt, S
    else:
        return None, S
 
 
 
def well_spaced(labels, delta):
    # Labels MUST be sorted
    assert all(labels[i] <= labels[i+1] for i in range(len(labels)-1))
 
    elt = labels[0]
    labels_ = [elt]  # init
    while any(labels > elt * delta):
        next = np.argmax(labels > elt * delta)
        labels_.append(labels[next])
        elt = labels[next]
    return labels_
 
 
def fingerprint(labels, K):
    p = K/(2*len(labels))
    t1 = 0.0
    t2 = 0.0
    while t1 == 0.0 or t2 == 0.0:
        S1 = []
        S2 = []
        for i in range(len(labels)):
            if (np.random.uniform(low=0.0, high=1.0) < p):
                S1.append(labels[i])
            if (np.random.uniform(low=0.0, high=1.0) < p):
                S2.append(labels[i])
 
        t1 = np.sum(S1)
        t2 = np.sum(S2)
 
    return t1/t2, [S1, S2]
 
 
def extraction(labels, delta, L1, scaling=False):
    def scan(V_s,alpha):
        S1_hat = []
        S2_hat = []
        V_s = list(alpha * np.array(labels))
        for j in L1[0]:
            f, V_s = find(V_s, delta, j)
            if f == None:
                break
            S1_hat.append(f)
 
        for j in L1[1]:
            f, V_s = find(V_s, delta, j)
            if f == None:
                break
            S2_hat.append(f)
 
        if len(L1[0]) == len(S1_hat) and len(L1[1]) == len(S2_hat):
            return True 
        return False
 
    if scaling:
        for elt in labels:
            if elt != 0:
                alpha = L1[0][0] / elt # scaling
            V_s = list(alpha * np.array(labels))
            if scan(V_s,alpha):
                return True
        return False
    else:
        return scan(labels,alpha=1)
 
 
if __name__ == '__main__':
 
    import pandas as pd
 
    # example run on a wikipedia clickstream graph
    df = pd.read_csv("../datasets/clickstream-fawiki-2020-08.tsv",
                     delimiter="\t", usecols=[0, 1, 3], comment="%",
                     header=None)
    df.columns = ['from', 'to', 'weight']
    G = nx.from_pandas_edgelist(df, source='from',
                                target='to', edge_attr='weight')
 
    delta = 1
    K = 10
 
    G, labels, counts = unique_labels(G)
    l = well_spaced(labels, delta)
    L2, L1 = fingerprint(l, K)
    print(L1, L2, len(labels))
    # assert extraction works on untouched G
    assert extraction(labels, delta, L1)
    print("Fingerprint successfully extracted")
