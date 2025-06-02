import re
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials
from xgboost import XGBClassifier
from sklearn.svm import SVC,LinearSVC



#加载数据并对数据进行初步分析
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#合并训练、测试集数据
alldata = pd.concat([train_data, test_data], axis = 0, ignore_index = True)



print(f"训练集有{len(train_data)}个样本数据，测试集有{len(test_data)}样本数据")
print(f"总共有{len(alldata)}个样本数据")



#提取乘客称谓到title中
alldata['title'] = alldata.Name.apply(lambda x: re.search(r',\s(.+?)\.', x).group(1))
#统计title称谓计数
alldata.title.value_counts()


#整合称谓信息
alldata.loc[alldata.title.isin(['Ms', 'Mlle']), 'title'] = 'Miss'
alldata.loc[alldata.title.isin(['Mme']), 'title'] = 'Mrs'
rare = ['Major', 'Lady', 'Sir', 'Don', 'Capt', 'the Countess', 'Jonkheer', 'Dona', 'Dr', 'Rev', 'Col']
alldata.loc[alldata.title.isin(rare), 'title'] = 'rare'
alldata.title.value_counts()

#提取新特征
alldata['family_size'] = alldata.SibSp + alldata.Parch + 1
alldata['ticket_group_count'] = alldata.groupby('Ticket')['Ticket'].transform('count')
alldata['group_size'] = alldata[['family_size', 'ticket_group_count']].max(axis = 1)
alldata['is_alone'] = alldata.group_size.apply(lambda x: 1 if x == 1 else 0)

#提取新特征，并填充缺失值
alldata['fare_p'] = alldata.Fare / alldata.ticket_group_count
alldata.loc[alldata[alldata.fare_p.isna()].index, 'fare_p'] = alldata.groupby('Pclass')['fare_p'].median()[3]


alldata.drop('Fare', axis = 1, inplace = True)
alldata.drop('Ticket', axis = 1, inplace = True)

alldata[alldata.Embarked.isnull()]
alldata.loc[alldata.Embarked.isnull(), 'Embarked'] = 'S'

preprocessing_dummies = pd.get_dummies(alldata[['Pclass', 'Sex', 'Embarked', 'title']],
               columns = ['Pclass', 'Sex', 'Embarked', 'title'],
               prefix = ['pclass', 'sex', 'embarked', 'title'],
               drop_first= False
              )
alldata = pd.concat([alldata, preprocessing_dummies], axis = 1)

alldata.drop(['Pclass', 'Sex', 'Embarked', 'title'], axis = 1, inplace = True)

imputer = KNNImputer(n_neighbors=4)
features = ['SibSp', 'Parch', 'Age',
       'family_size', 'ticket_group_count', 'group_size', 'is_alone',
       'fare_p', 'pclass_1', 'pclass_2', 'pclass_3',
       'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S',
       'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_rare']
all_data_filled = pd.DataFrame(imputer.fit_transform(alldata[features]), columns=features)
alldata['Age'] = all_data_filled['Age']

#k均值聚类
kmeans = KMeans(n_clusters = 4, random_state = 41)
labels_pred = kmeans.fit_predict(alldata[['Age']])

# 获取聚类中心，返回每段年龄平均值。
kmeans.cluster_centers_.flatten()

np.argsort(kmeans.cluster_centers_.flatten())

label_dict = {label: v for v, label in enumerate(np.argsort(kmeans.cluster_centers_.flatten()))}

labels = [label_dict[label] for label in labels_pred]
alldata['Age_category'] = labels

alldata.drop(['PassengerId', 'Cabin', 'Age'], axis = 1, inplace = True)

train_clean = alldata.loc[alldata.Survived.notnull()].copy()
test_clean = alldata.loc[alldata.Survived.isnull()].drop('Survived', axis = 1).copy()

X = train_clean.drop('Survived', axis = 1)
y = train_clean.Survived

features = ['family_size', 'group_size', 'is_alone', 'Age_category','ticket_group_count',
       'pclass_1', 'pclass_2', 'pclass_3', 'sex_female', 'sex_male',
       'embarked_C', 'embarked_Q', 'embarked_S', 'title_Master', 'title_Miss',
       'title_Mr', 'title_Mrs', 'title_rare', 'fare_p']
X = X[features].copy()

#划分训练集，验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41) 

def objective(params):
    svc_params = {k: v for k, v in params.items() if k in ['C', 'kernel', 'gamma', 'degree', 'probability']}
    model = SVC(random_state=41, **svc_params)
    # 进行交叉验证，计算 AUC
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
    return cv_results.mean()
 
# 定义搜索空间
space = {
    'n_estimators': hp.randint('n_estimators', 100, 201),
    'max_depth': hp.randint('max_depth', 4, 7),
    'learning_rate': hp.uniform('learning_rate', 0.1, 1),
    'gamma': hp.uniform('gamma', 0, 5),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
}
# 运行贝叶斯优化
trials = Trials()
#最佳参数
best_params = fmin(objective, space, algo=tpe.suggest, max_evals=200, trials=trials)
print(best_params)

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_val)  # 使用测试集进行预测
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

test_pred = best_model.predict(test_clean[features])
pd.DataFrame({
     'PassengerId': test_data.PassengerId,
     'Survived' : test_pred
 }).to_csv('submission_svm1.csv', index = False)

