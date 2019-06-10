from node2vec_v2 import Node2Vec
import numpy as np


def norm_hic(m):
    average = [np.mean(np.diag(m[i:, :len(m)-i])) for i in range(len(m))]
    for i in range(len(m)):
        if average[i] == 0: continue

        for j in range(len(m)-i):
            m[j][j+i] = m[j][j+i] / average[i]
            if i != 0:
                m[j+i][j] = m[j+i][j] / average[i]
    return m


def dynnode2vec(graphs, epochs):
    tot_c = len(graphs) * epochs
    cnt = 0
    for j in range(epochs):
        for i, graph in enumerate(graphs):
            cnt += 1
            print(f'Training node2vec model: {cnt} / {tot_c}')

            n2v = Node2Vec(graph, dimensions=16, walk_length=80, num_walks=15, p=1, q=1, workers=8)
            if i == 0 and j == 0:
                m = n2v.fit(save_wv_file='./temp/wv_0.emb', min_count=1, window=3)
            elif i == 0:
                m = n2v.fit(initial_wv_file=f'./temp/wv_{len(graphs) - 1}.emb', save_wv_file=f'./temp/wv_{i}.emb', min_count=1,
                            window=2)
            else:
                m = n2v.fit(initial_wv_file=f'./temp/wv_{i - 1}.emb', save_wv_file=f'./temp/wv_{i}.emb', min_count=1,
                            window=2)


data = np.zeros((23, 461, 461))
orig_data = np.load('1171_sc_TAD_chr1.npy')
for i in range(23):
    s = np.sum(orig_data[50 * i: 50 * (i+1)], axis=0)
    data[i, :, :] = norm_hic(s)


dynnode2vec(data, 5)

