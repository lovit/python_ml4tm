## (Approximiated) Nearest Neighbor Search

이웃 탐색 문제는 그 자체로 밀도 추정이나 classification, 혹은 regression 의 역할을 합니다. 혹은 ISOMAP 이나 k-means 학습 과정에서 한 점을 기준으로 가장 가까운 centroid 를 탐색하는 것처럼 한 알고리즘의 부분 문제로 포함되어 있기도 합니다. 그런데 한 점 (query point) 을 기준으로 가장 가까운 k 개의 점을 찾거나, 거리가 r 이하인 점들을 찾는 검색 비용은 reference data 의 크기, n 에 비례합니다. 이러한 검색 비용을 완화하기 위하여 n 개의 모든 reference data 와의 거리를 계산하는 것이 아니라, 최인접이웃이 될 가능성이 높은 몇 개의 점들만 거리를 계산하는 방법들이 연구되었습니다.

가장 널리 알려진 방법은 Locality Sensitive Hashing (LSH) 입니다. 이전에 Scikit-learn 에서는 LSHForest 라는 클래스를 제공하였습니다. 이 구현체는 original LSH 에 가까운 구현체이며, 사실 LSH 는 데이터의 특징에 맞춰 튜닝을 해야 하는 것들이 많습니다. 그렇기 때문에 Scikit-learn >= 0.21.3 부터는 성능이 개선되지 않은 이 패키지를 deprecated 하였습니다. LSH 의 기본 버전은 구현하는데 큰 어려움이 없습니다. 이에 대한 과정을 `day8_develop_lsh` 에 넣어뒀습니다. LSHForest 와는 조금 다른 방식으로 구현하였지만, 큰 차이는 없습니다.

LSH 외에도 다양한 최인접 검색을 위한 인덱서들이 제안되었습니다. NN-descent 는 미리 구축된 최인접이웃 정보를 이용하여 점진적으로 최인접이웃을 개선하는 방법입니다. PyNNDescent 알고리즘을 이용하는 방법을 `day8_pynndescent` 에서 살펴볼 수 있습니다. 같은 개념을 다르게 구현하면 random neighbor graph 와 nearest neighbor graph 를 혼합하여 최인접이웃을 탐색할 수도 있습니다. 이에 대하여 `day8_network_based_similarity_search` 에서 살펴볼 수 있습니다. 또한 이전에 Facebook Research 에서 Bag-of-Words model 과 같은 sparse matrix 용으로 hierarchical clusters tree 기반 인덱서인 PySparNN 을 제공하였습니다. 이에 대해서도 `day8_pysparnn` 에서 살펴볼 수 있습니다.

## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [PyNNDescent](https://github.com/lmcinnes/pynndescent)

```
pip install pynndescent
```

2. [Network based nearest neighbor search](https://github.com/lovit/network_based_nearest_neighbors)

```
git clone https://github.com/lovit/network_based_nearest_neighbors
```

3. [PySparNN](https://github.com/facebookresearch/pysparnn)

```
git clone https://github.com/facebookresearch/pysparnn
```