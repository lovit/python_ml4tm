import time
import numpy as np
from sklearn.metrics import pairwise_distances

from .utils import most_closest_points

class NetworkBasedNeighbors:
    """Network based nearest neighbor search index

    .. code-block:: python

        >>> index = NetworkBasedNeighbors(
        >>>     x,                     # reference data. numpy.ndarray or scipy.sparse.csr_matrix
        >>>     n_nearest_neighbors=5, # number of nearest neighbors
        >>>     n_random_neighbors=5,  # number of random neighbors
        >>>     batch_size=500,        # indexing batch size
        >>>     metric='euclidean',    # metric, possible all metric defined in scipy
        >>>     verbose=True           # verbose mode if True
        >>> )
        >>> dist, idxs = index.search_neighbors(query, k=k) # usage mode
        >>> dist, idxs, infos, process_time = index._search_neighbors_dev(query, k=k) # debug mode

    :param int a: An integer to be computed
    :param int b: An integer to be computed

    :returns: The sum of two integer values
    """
    def __init__(self, X=None, n_nearest_neighbors=5,
        n_random_neighbors=5, batch_size=500,
        metric='euclidean', verbose=True):

        self.n_nearest_neighbors = n_nearest_neighbors
        self.n_random_neighbors = n_random_neighbors
        self.batch_size = batch_size
        self.verbose = verbose
        self.metric = metric
        self.buffer_factor = 3

        if X is not None:
            self.index(X)

    def index(self, X):
        """The function for indexing.

        :param Union[scipy.sparse.csr_matrix, numpy.ndarray] x: matrix
        :returns: None
        """
        n_data = X.shape[0]
        num_nn = self.n_nearest_neighbors
        num_rn = self.n_random_neighbors

        # set reference data
        self.X = X
        self.n_data = n_data

        if self.verbose:
            print('Indexing ...')

        # nearest neighbor indexing
        _, self.nn = most_closest_points(
            X, topk=num_nn+1, batch_size=self.batch_size,
            metric=self.metric, verbose=self.verbose)
        self.nn = self.nn[:,1:]

        # random neighbor indexing
        self.rn = np.random.randint(n_data, size=(n_data, num_rn))

        if self.verbose:
            print('Indexing was done')

    def search_neighbors(self, query, k=5, max_steps=10, converge=0.000001):
        """Search function.

        :param Union[sparse.csr_matrix, numpy.ndarray] query: query vector
        :param int k: the number of nearest neighbors
        :param int max_steps: the maximal number of updates
        :param float converge: threshold for convergence check
        :returns: tuple of nearest neighbor distances and corresponding indices
        """

        dist, idxs, infos, process_time = self._search_neighbors_dev(
            query, k, max_steps, converge)
        return dist, idxs

    def _search_neighbors_dev(self, query, k=5, max_steps=10, converge=0.000001):
        """Search function.

        :param Union[sparse.csr_matrix, numpy.ndarray] query: query vector
        :param int k: the number of nearest neighbors
        :param int max_steps: the maximal number of updates
        :param float converge: threshold for convergence check
        :returns: distances, indices, update logs, process time
            update logs is list consisting
            (step index, average distance, distance difference with previous step)
        """

        buffer_size = self.buffer_factor * k
        dist, idxs = self._initialize(query, buffer_size)
        dist_avg = dist.sum() / dist.shape[0]

        infos = []
        process_time = time.time()

        for step in range(max_steps):
            candi_idxs = np.unique(
                np.concatenate([idxs, self._get_neighbors(idxs)])
            )

            candi_dist = pairwise_distances(
                query, self.X[candi_idxs], metric=self.metric).reshape(-1)

            args = candi_dist.argsort()[:buffer_size]
            idxs_ = candi_idxs[args]
            dist_ = candi_dist[args]
            dist_avg_ = dist_.sum() / dist_.shape[0]

            diff = dist_avg - dist_avg_
            infos.append((step, dist_avg_, diff))

            if diff <= converge:
                break

            dist = dist_
            idxs = idxs_
            dist_avg = dist_avg_

        process_time = time.time() - process_time

        idxs_ = dist.argsort()[:k]
        idxs = idxs[idxs_]
        dist = dist[idxs_]

        return dist, idxs, infos, process_time

    def _initialize(self, query, k):
        idxs = np.random.randint(self.n_data, size=k)
        refx = self.X[idxs]
        dist = pairwise_distances(refx, query)
        return dist, idxs

    def _get_neighbors(self, base):
        neighbor_idxs = np.concatenate(
            [self.nn[base].reshape(-1),
             self.rn[base].reshape(-1)]
        )
        return neighbor_idxs