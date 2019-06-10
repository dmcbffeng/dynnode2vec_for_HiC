# dynnode2vec
Implemented dynamic node2vec

node2vec_v2.py: implemented node2vec with numpy.array as input (more convenient for HiC contact map):

example:
model = Node2Vec(contact_map, dimensions=8, walk_length=80, num_walks=20, p=1, q=1, workers=8)
model.fit(initial_wv_file=None, save_wv_file='./temp/wv_agg.emb', min_count=1, window=3)



