import numpy as np
import gensim
import random


class Node2Vec:
    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1,
                 workers=1):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :type graph: numpy.array
        :param dimensions: Embedding dimensions (default: 128)
        :type dimensions: int
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        """
        self.graph = graph
        for i in range(len(self.graph)):
            self.graph[i, i] = 0
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.first_walk_probability = np.zeros(graph.shape)  # [current, destination]
        self.walk_probability = np.zeros((len(graph), len(graph), len(graph)))  # [last, current, destination]

        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """
        # First walk
        print(' Computing prob for first walk...')
        sum_weight = np.sum(self.graph, axis=1)
        sum_weight = np.where(sum_weight == 0, 1, sum_weight)
        normalized_weight = self.graph / np.tile(sum_weight, (len(self.graph), 1)).T
        self.first_walk_probability = normalized_weight

        # Afterwards
        print(' Computing prob for second walk...')
        for last in range(len(self.graph)):
            for current in range(len(self.graph)):
                if self.graph[last, current] == 0:
                    continue
                unbiased_unnormalized_weight = self.graph[current, :]
                for destination in range(len(self.graph)):
                    if destination == last:
                        unbiased_unnormalized_weight[last] = unbiased_unnormalized_weight[last] / self.p
                    elif self.graph[last, destination] == 0:
                        unbiased_unnormalized_weight[last] = unbiased_unnormalized_weight[last] / self.q
                biased_normalized_weight = unbiased_unnormalized_weight / np.sum(unbiased_unnormalized_weight)
                self.walk_probability[last, current, :] = biased_normalized_weight

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """
        all_walks = []
        for n_walk in range(self.num_walks):
            n_th_walks = []
            for source in range(len(self.graph)):
                walk = [source]
                # first walk
                #if n_walk == 0:
                #    print(source, np.sum(self.first_walk_probability[source, :]))

                if np.sum(self.first_walk_probability[source, :]) != 0:
                    first_dest = np.random.choice(np.arange(len(self.graph)), 1, p=self.first_walk_probability[source, :])[0]
                    walk.append(first_dest)
                else:
                    walk = [str(source)]
                    n_th_walks.append(walk)
                    continue

                while len(walk) < self.walk_length:
                    if np.sum(self.walk_probability[walk[-2], walk[-1], :]) != 0:
                        dest = np.random.choice(np.arange(len(self.graph)), 1, p=self.walk_probability[walk[-2], walk[-1], :])[0]
                        walk.append(dest)
                    else:
                        break
                walk = [str(elm) for elm in walk]
                n_th_walks.append(walk)
                # if n_walk == 0:
                #     print(walk)

            random.shuffle(n_th_walks)
            all_walks += n_th_walks
        return all_walks

    def fit(self, initial_wv_file=None, save_wv_file=None, **skip_gram_params):
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :param initial_wv_file: use pre-trained wv file
        :return: A gensim word2vec model
        """
        print(' Training skip-gram model...')
        model = gensim.models.Word2Vec(self.walks, workers=self.workers, size=self.dimensions, sg=1, **skip_gram_params)
        # min_count, window, sg, negative, alpha
        if initial_wv_file:
            last_model = gensim.models.KeyedVectors.load_word2vec_format(initial_wv_file, binary=False)
            model.build_vocab([list(last_model.vocab.keys())], update=True)
            model.intersect_word2vec_format(initial_wv_file, binary=False, lockf=1.0)  # lockf=1.0 allows updating of wv
            model.train(self.walks, total_examples=model.corpus_count, epochs=100)  # epochs = 10?
        else:
            model.train(self.walks, total_examples=model.corpus_count, epochs=100)  # epochs = 10?
        if save_wv_file:
            model.wv.save_word2vec_format(save_wv_file)

        return model

