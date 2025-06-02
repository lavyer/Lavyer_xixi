# Lavyer_xixi
参加的比赛为Kaggle上的泰塔尼克号幸存者预测，链接如下：https://www.kaggle.com/competitions/titanic/submissions
我的环境设计放在requirements.txt里面，可以直接通过pip install -r requirements.txt安装所需的虚拟环境。
submission_svm.csv文件是我所上传的最佳结果，可以达到0.79425,但是submission_optimized.csv也是我跑出来的结果可以达到0.78708，效果也还可以，其他的效果不大好没有上传。
train.csv文件是题目所给出的训练集，test.csv是测试集，genger_submission.csv是官方给出的提交样例。
processing_data.py是最初对数据集的一些了解工作和处理。
train.py和val.py都是可以直接运行生成submission.csv的文件，并且效果差不多，但是val.py的代码更流畅。
