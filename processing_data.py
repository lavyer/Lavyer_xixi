import numpy as np
import pandas as pd
import re
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

#数据的载入
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
#合并数据
alldata=pd.concat([train_data,test_data],axis=0,ignore_index=True)
#alldata.info()
"""
 0   PassengerId  1309 non-null   int64
 1   Survived     891 non-null    float64
 2   Pclass       1309 non-null   int64
 3   Name         1309 non-null   object
 4   Sex          1309 non-null   object
 5   Age          1046 non-null   float64
 6   SibSp        1309 non-null   int64
 7   Parch        1309 non-null   int64
 8   Ticket       1309 non-null   object
 9   Fare         1308 non-null   float64
 10  Cabin        295 non-null    object
 11  Embarked     1307 non-null   object
dtypes: float64(3), int64(4), object(5)
memory usage: 122.8+ KB
"""

# print(f"训练集有{len(train_data)}个训练数据，测试集有{len(test_data)}个测试数据")
# print(f"总共有{len(alldata)}个样本数据")
"""
训练集有891个训练数据，测试集有418个测试数据
总共有1309个样本数据
"""

print(round(train_data.describe(percentiles=[.5,.6,.7,.8]),2))
print(train_data.describe(include=['O']))
"""
通过生成数据可以看到：train_data:Age只有714个值，存在NaN值，并且std(Age)=14.53,年纪差大，80%有41岁
Cabin只有204个值，NaN;
male>female,male=577;
embarked=3，只有三种登陆的港口
"""

#查看三种票级的生存率
# print(train_data[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Survived',ascending=False))
# #SibSp
# print(train_data[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived',ascending=False))
# #Parch
# print(train_data[['Parch','Survived']].groupby(['Parch']).mean().sort_values(by='Survived',ascending=False))
# #Embarked
# print(train_data[['Embarked','Survived']].groupby(['Embarked']).mean().sort_values(by='Survived',ascending=False))
# #Sex
#print(train_data[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Survived',ascending=False))
"""
        Survived
Pclass
1       0.629630
2       0.472826
3       0.242363

SibSp
1      0.535885
2      0.464286
0      0.345395
3      0.250000
4      0.166667
5      0.000000
8      0.000000

Parch
3      0.600000
1      0.550847
2      0.500000
0      0.343658
5      0.200000
4      0.000000
6      0.000000

          Survived
Embarked
C         0.553571
Q         0.389610
S         0.336957

        Survived
Sex
female  0.742038
male    0.188908
票级和性别与生存率有关联：0.6：0.4：0.2;0.7:0.1
"""

#猜测年纪和生存也有关系，可能是妇女和孩子先走？？？，通过名字称呼来提取新特征
alldata['title']=alldata.Name.apply(lambda x:re.search(r',\s(.+?)\.',x).group(1))
# print(alldata.title.value_counts())
"""
title
Mr              757
Miss            260
Mrs             197
Master           61
Rev               8
Dr                8
Col               4
Major             2
Mlle              2

男性，未婚女性，已婚女性，小孩
"""


#整合称谓信息
alldata.loc[alldata.title.isin(['Ms', 'Mlle']), 'title'] = 'Miss'
alldata.loc[alldata.title.isin(['Mme']), 'title'] = 'Mrs'
rare = ['Major', 'Lady', 'Sir', 'Don', 'Capt', 'the Countess', 'Jonkheer', 'Dona', 'Dr', 'Rev', 'Col']
alldata.loc[alldata.title.isin(rare), 'title'] = 'rare'
alldata.title.value_counts()

alldata.drop(['Name'],axis=1,inplace=True)

#提取新特征
alldata['family_size'] = alldata.SibSp + alldata.Parch + 1   #没父母没姐妹也是有一个人
alldata['ticket_group_count'] = alldata.groupby('Ticket')['Ticket'].transform('count')
alldata['group_size'] = alldata[['family_size', 'ticket_group_count']].max(axis = 1)
alldata['is_alone'] = alldata.group_size.apply(lambda x: 1 if x == 1 else 0)

#提取新特征，并填充缺失值
alldata['fare_p'] = alldata.Fare / alldata.ticket_group_count
alldata.loc[alldata[alldata.fare_p.isna()].index, 'fare_p'] = alldata.groupby('Pclass')['fare_p'].median()[3]

alldata.drop('Fare', axis = 1, inplace = True)
alldata.drop('Ticket', axis = 1, inplace = True)

#print(alldata.Embarked.isnull()):::61和829
alldata.loc[alldata.Embarked.isnull(),'Embarked']='S'

#对分类变量进行独热编码（One-Hot Encoding），并将生成的新特征合并到原始数据中
preprocessing_dummies = pd.get_dummies(
    alldata[['Pclass', 'Sex', 'Embarked', 'title']],  # 选择需要编码的列
    columns=['Pclass', 'Sex', 'Embarked', 'title'],   # 指定编码的列名
    prefix=['pclass', 'sex', 'embarked', 'title'],    # 新列名的前缀
    drop_first=False                                  # 保留所有类别列
)
alldata.drop(['Pclass','Sex','Embarked','title'],axis=1,inplace=True)

#对年龄进行处理，填充NaN值
imputer = KNNImputer(n_neighbors=4)
features = ['SibSp', 'Parch', 'Age',
       'family_size', 'ticket_group_count', 'group_size', 'is_alone',
       'fare_p', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'embarked_C', 'embarked_Q', 'embarked_S',
       'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_rare']
all_data_filled = pd.DataFrame(imputer.fit_transform(alldata[features]), columns=features)
alldata['Age'] = all_data_filled['Age']

#k均值聚类
kmeans = KMeans(n_clusters = 4, random_state = 41)
labels_pred = kmeans.fit_predict(alldata[['Age']])
np.argsort(kmeans.cluster_centers_.flatten())

label_dict={label:v for v,label in enumerate(np.argsort(kmeans.cluster_centers_.flatten()))}
labels=[label_dict[label] for label in labels_pred]
alldata['Age_category']=labels
alldata.drop(['PassengerId','Cabin','Age'],axis=1,inplace=True)
alldata.info()