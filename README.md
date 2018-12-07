# text-classification
“达观杯”文本智能处理挑战赛python代码示例，逻辑回归方法

# 达观杯介绍：

类似kaggle的数据科学比赛，任何人可以参加

网址：http://www.dcjingsai.com/

可以用支付宝实名注册

 

# 项目名称：

“达观杯”文本智能处理挑战赛

安装Python，运行如下代码。可以得到一个分数

 

# 项目代码说明：

在Python3中运行代码就可以

导入库函数

读取文件，并且删除无关东西

获取特征向量

进行训练，测试

保存文件为可以提交的CSV格式



# 输入说明

注意修改自己的路径

train_set.csv 1.5G, 普通电脑打开很吃力，随意阅读也吃力，谨慎打开；
第一行有：ID，article， Word_seg, class;
id:文章数量编号102277个文本； article：文章内容，是一些数字； Word_seg:也是一些数字； class:文本对应的类别从1到20

test_set.csv 1.38Gb,
第一行有：ID，article， Word_sequence,内容和训练集一样，只是没有了类别标签

result.csv 865 KB, 
第一行有ID， class；也就是预测每一个文档的类别

# 输出内容：

```
start
/usr/local/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
/usr/local/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/usr/local/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:459: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
/usr/local/lib/python3.7/site-packages/sklearn/svm/base.py:922: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
end
[Finished in 1315.7s]

```

# 提升成绩的方法
方法

得分：0.73

数据预处理：这里的数据比较完整，不用担心

特征工程
这里技巧很足，需要不断的积累

机器学习算法
不同算法都有对应的任务类型。但是xgboost很厉害
lightboard微软开发的工具，适合大部分的情况，属于西瓜书第八章的内容。

数据增强：
给了1万条数据，变成10万条数据。
