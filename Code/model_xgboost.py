import numpy as np
import pandas as pd
import data_processing
from sklearn import metrics
from sklearn.model_selection import train_test_split

data = data_processing.load_data(download=False)
new_data = data_processing.convert2onehot(data)

# prepare training data
new_data = new_data.values.astype(np.float32)     # change to numpy array and float32
X_train, X_test, y_train, y_test = train_test_split(new_data[:,:21], new_data[:,21:], test_size=0.2, shuffle=True)  # 划分数据集


# **********  原生接口  ************* #
import xgboost as xgb

# data convert
train_data = xgb.DMatrix(X_train, label=np.argmax(y_train, axis=1))
X_test2 = xgb.DMatrix(X_test)

# data training
num_round = 20
param = {
    'n_estimators' :20,
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth': 5,
    'silent': 0,
    'objective': 'multi:softprob',
    'num_class': 4
}
model = xgb.train(param, train_data, num_round)

# make prediction
y_predict = model.predict(X_test2)
# print(y_predict)
# print(y_test)
print("XGBoost_自带接口    AUC Score : %f" % metrics.roc_auc_score(y_test, y_predict))



# *********  sklearn接口  ************* #
from xgboost.sklearn import XGBClassifier

clf = XGBClassifier(
    n_estimators=20,  # 树的个数
    learning_rate=0.3,
    max_depth=5,
    min_child_weight=1,
    gamma=0.2, # 叶子节点的正则化系数
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob', #输出类型
    num_class=4,
    nthread=2,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)
model2 = clf.fit(X_train, np.argmax(y_train, axis=1))
y_predict2 = model2.predict(X_test)
y_test = np.argmax(y_test, axis=1)
# print(y_predict2)
# print(y_test)
print("XGBoost_自带接口    ACC Score : %f" % metrics.accuracy_score(y_test, y_predict2))
