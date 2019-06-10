import numpy as np
from node2vec_v2 import Node2Vec
import matplotlib.pyplot as plt
import seaborn as sns


def spectral(w):
    sm = np.sum(w, axis=1)
    zeros = []
    for i in range(len(sm)):
        if sm[i] == 0:
            zeros.append(i)
    w = np.delete(w, zeros, axis=0)
    w = np.delete(w, zeros, axis=1)

    sm = np.sum(w, axis=1)
    d = np.diag(sm)
    d_ = np.diag(1 / np.sqrt(sm))
    l = d - w
    l = d_.dot(l).dot(d_)
    eigenvals, eigenvecs = np.linalg.eig(l)
    st = np.argsort(eigenvals)
    min_ = st[1]
    eigen = eigenvecs[:, min_].T
    # eigen = np.where(eigen > 0, 1, -1)
    for zero in zeros:
        eigen = np.insert(eigen, zero, 0)
    # print(eigenvals[min_])
    return eigen


def norm_hic(m):
    average = [np.mean(np.diag(m[i:, :len(m)-i])) for i in range(len(m))]
    for i in range(len(m)):
        if average[i] == 0: continue

        for j in range(len(m)-i):
            m[j][j+i] = m[j][j+i] / average[i]
            if i != 0:
                m[j+i][j] = m[j+i][j] / average[i]
    return m


maps = np.load('ch1.npy')
print(maps.shape)
sm_map = norm_hic(np.log(np.sum(maps, axis=0) + 1))
# plt.figure()
# sns.heatmap(sm_map, square=True, vmax=4, vmin=0)
# plt.show()

# e = spectral(norm_hic(sm_map))
# plt.fill_between(range(461), e, 0)
# plt.show()

model = Node2Vec(sm_map, dimensions=8, walk_length=80, num_walks=20, p=1, q=1, workers=8)
# for walk in model.walks[:50]:
#     print(walk)
model.fit(initial_wv_file=None, save_wv_file='./temp/wv_agg.emb', min_count=1, window=3)

