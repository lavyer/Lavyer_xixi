import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import joblib

warnings.filterwarnings('ignore')

# 1. 特征工程
def advanced_feature_engineering(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['FareCluster'] = kmeans.fit_predict(data[['Fare']])
    return data

# 2. 数据加载和预处理
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data = advanced_feature_engineering(train_data)
test_data = advanced_feature_engineering(test_data)

X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# 3. 预处理管道
numeric_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'IsAlone', 'FareCluster']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. 超参数优化
def objective(params):
    # 转换choice参数为实际值
    kernel_options = ['linear', 'rbf']
    params['kernel'] = kernel_options[params['kernel']]
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(**params, probability=True))
    ])
    
    score = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}

# 使用索引作为choice值，在objective中转换
space = {
    'C': hp.loguniform('C', np.log(0.1), np.log(10)),
    'kernel': hp.choice('kernel', [0, 1]),  # 0=linear, 1=rbf
    'gamma': hp.loguniform('gamma', np.log(0.01), np.log(1)),
    'degree': hp.choice('degree', [2, 3])  # 2或3
}

def optimize_hyperparameters(X, y, preprocessor, max_evals=50):
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    
    # 转换最佳参数
    kernel_map = {0: 'linear', 1: 'rbf'}
    degree_map = {0: 2, 1: 3}
    
    best_params = {
        'C': best['C'],
        'kernel': kernel_map[best['kernel']],
        'gamma': best['gamma'],
        'degree': degree_map[best['degree']],
        'probability': True
    }
    
    return best_params, trials

best_params, trials = optimize_hyperparameters(X, y, preprocessor, max_evals=50)

# 5. 最终模型训练和评估
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(**best_params))
])


# 训练最终模型
final_pipeline.fit(X, y)
train_pred = final_pipeline.predict(X)
train_accuracy = accuracy_score(y, train_pred)
print(f"验证集准确率: {train_accuracy:.4f}")

# 6. 预测和保存结果
test_pred = final_pipeline.predict(test_data.drop('PassengerId', axis=1))

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('submission_svm.csv', index=False)
joblib.dump(final_pipeline, 'titanic_svm_model.pkl')
print("模型训练和保存完成")