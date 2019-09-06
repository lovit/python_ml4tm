# Day2 Classifiers

Scikit-learn 에서 제공하는 classifiers 및 XGBoost 를 학습하는 실습 코드입니다. 특히 scikit-learn 에서는 학습된 모델들에 저장된 각각의 attributes 에 대해서도 알아봅니다. 또한 classifiers 간의 성능 비교를 통하여 전반적인 모델의 성능도 알아봅니다.

튜토리얼은 아래의 순서대로 보는 것이 좋습니다.

## day2_logistic_regression_and_regularization_for_sentence_classification.ipynb

Logistic regression 에 L1, L2 regularization 을 부여하여 문장의 긍/부정을 분류하는 모델을 학습합니다.

## day2_other_classifiers.ipynb

(1) Logistic regression 외에도 (2) Multilayer feed forward neural network, (3) Support Vector Classifier (RBF, Linear), (4) Naive Bayes, (5) Decision Tree, (6) Random Forest 의 사용법을 알아보고, 학습된 패러매터들의 값을 확인합니다.

그 외 Extra Tree 나 Gradient Boosting Tree 와 같은 다른 모델들은 Random Forest 와 사용법이 비슷합니다.

## day2_xgboost_classifier.ipynb

XGBoost 패키지를 이용하여 모델을 학습합니다.

## day2_classification_performance_comparison.ipynb

작은 크기의 데이터에 대하여 (1) Decision Tree, (2) Lasso regression, (3) Ridge regression, (4) Random Forest, (5) Gradient boosting tree, (6) XGBoost 의 성능을 비교합니다. 이를 위하여 10 fold cross validation 을 이용하였으며, scikit-learn 패키지가 아닌 XGBoost 의 경우, 직접 cross validation 과정을 구현합니다.
