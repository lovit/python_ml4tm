# IPython notebook 에서는 verbose message 가 출력되지 않을 수 있습니다.
# 메시지를 확인하기 위하여 terminal 환경에서 실행할 수 있는 py 파일을 만듭니다.
# >>> python lightgbm_test.py 를 실행하여 결과를 확인합니다.

import config
import lightgbm as lgb
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from navermovie_comments import load_sentiment_dataset

texts, x, y, idx_to_vocab = load_sentiment_dataset(data_name='small', tokenize='komoran')
x = scipy.sparse.csr_matrix(x, dtype=np.float32)
y[np.where(y == -1)[0]] = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
dtrain = lgb.Dataset(data=x_train, label=y_train)

param = {
    'num_leaves':31,
    # 'max_depth':8,
    'min_data_in_leaf': 10,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': 2
}

num_round = 10
bst = lgb.train(param, dtrain, num_round)
