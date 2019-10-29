import numpy as np
import time
from sklearn.metrics import pairwise_distances

def most_closest_points(X, Y=None, metric='euclidean',
    topk=5, batch_size=500, verbose=False):

    if Y is None:
        Y = X

    n_data = X.shape[0]
    num_batch = int(np.ceil(n_data/batch_size))

    idxs = np.zeros((n_data, topk), dtype=np.int)
    dist = np.zeros((n_data, topk))

    base_time = time.time()

    for batch_idx in range(num_batch):
        b = batch_idx * batch_size
        e = (batch_idx + 1) * batch_size
        dist_ = pairwise_distances(X[b:e], Y, metric=metric)
        idxs_ = dist_.copy().argsort(axis=1)[:,:topk]
        idxs[b:e] = idxs_
        dist_.sort(axis=1)
        dist[b:e] = dist_[:,:topk]

        if verbose:
            remain_time = (
                (time.time() - base_time) * (num_batch - batch_idx)
                / (batch_idx + 1)
            )
            print('\rbatch {} / {}, remain = {:f} sec.'.format(
                batch_idx+1, num_batch, remain_time), end='', flush=True)

    if verbose:
        process_time = time.time() - base_time
        print('\rbatch {} / {} done. computation time = {:f} sec.'.format(
            num_batch, num_batch, process_time))

    return dist, idxs