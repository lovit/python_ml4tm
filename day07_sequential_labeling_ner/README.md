## Sequential labeling & NER

자연어처리 과업에는 sequential labeling 를 이용하는 과업들이 자주 있습니다. 객체명인식 (Named Entity Recognition) 이나 품사 판별 (Part of Speech Tagging) 이 대표적입니다. Sparse vector 형식의 학습데이터를 이용하는 접근법에서는 Conditional Random Field 나 이와 유사한 transition based tagger 가 좋은 성능을 보여줍니다.

python-crfsuite 은 Crfsuite 패키지를 파이썬에서 이용할 수 있도록 도와줍니다. `day7_0_pycrfsuite_CoNLL2002_NER` 에서는 객체명인식 competiton 인 CoNLL 2002 데이터를 이용한 객체명인식 모델을 구현해 봅니다. 이는 python-crfsuite 에 주석을 추가하고, 앞 뒤에 등장한 단어 정보만을 이용하였을 때에도 성능이 그리 떨어지지 않음을 확인합니다. 그리고 이 튜토리얼에서 `python-crfsuite` 패키지의 활용법을 익히실 수 있습니다.

더하여 `day7_a_pycrfsuite_spacing` 에서는 python-crfsuite 를 이용하여 한국어 띄어쓰기 교정기를 만들어봅니다.

그런데 객체명인식 과업에 반드시 sequential labeling 알고리즘을 이용해야 하는 것은 아닙니다. 한 문장 내에서 하나의 단어가 객체명인 경우들이 많으며, CRF 라 하더라도 features 로 이용하는 정보는 결국 앞, 뒤에 등장하는 단어입니다. 단어의 위치 정보까지 포함하여 features 를 만들면 window classification 으로도 객체명을 추출하거나 인식할 수 있습니다. `day7_a_logistic_window_classification_ner` 에서는 Logistic regression 을 이용하여 영화평 데이터에서 배우, 캐릭터 이름을 인식하는 객체명인식 모델을 만듭니다.

`day7_a_feedforward_window_classification_ner` 에서는 하나의 hidden layer 를 이용하는 feed-forward neural network 기반 객체명인식 모델을 만듭니다. 입력값은 Word2Vec 으로 학습한 단어 벡터입니다. 이 튜토리얼에서는 scikit-learn 을 이용하여 minibatch style 로 모델을 학습하는 방법에 대해서도 연습합니다. 그리고 딥러닝 모델을 구현하는 언어들에 익숙하시다면 Convolutional Neural Network 기반으로 모델을 설계해도 잘 작동합니다.


## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/lovit/textmining_dataset
```

2. [python_crfsuite](https://github.com/scrapinghub/python-crfsuite)

```
pip install python-crfsuite
```

3. [Scikit-learn >= 0.20.3](https://scikit-learn.org/)

```
pip install scikit-learn
```
