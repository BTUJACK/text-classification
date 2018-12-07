'''
作者：公众号：湾区人工智能
场景：达观杯 文本智能竞赛  http://www.pkbigdata.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html
时间：#2018-12-01 563610 December Saturday the 48 week, the 335 day SZ

'''

print('start')
import pandas as pd 
from sklearn.linear_model import LogisticRegression #导入线性回归库
from sklearn.feature_extraction.text import CountVectorizer #导入特征提取库

#读取文件，并且删除无关东西
df_train = pd.read_csv('/Users/apple/Documents/ST/python/competition/DataCastel/text_intelligence/new_data/train_set.csv')
df_test = pd.read_csv('/Users/apple/Documents/ST/python/competition/DataCastel/text_intelligence/new_data/test_set.csv')
df_train.drop(columns =['article', 'id'], inplace = True ) #问题1： 为什么要删除这两个列,id列没有意义，不需要用article，直接删除
df_test.drop(columns =['article'], inplace = True ) 




#获取特征向量
vectorizer = CountVectorizer(ngram_range = (1,2), min_df = 3, max_df = 0.9, max_features = 100000) #提取特征
vectorizer.fit(df_train['word_seg']) #问题2：为啥要训练这一列内容，要先学习整个数据集的词的DF（文档词频）
x_train = vectorizer.transform(df_train['word_seg']) #特征转为特征向量
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class'] - 1  #问题3：这里为啥要给所有的类别都减去1，减一是代码习惯问题，让class从0计数

lg = LogisticRegression(C = 4, dual = True) #逻辑回归初始化
lg.fit(x_train, y_train) #进行训练，模型保存在lg里面

y_test = lg.predict(x_test) #用模型进行测试

df_test['class'] = y_test.tolist() #测试结果转为列表，并且放入测试文档的类别里面。问题5：测试文档没有类别这个列。这行代码会自动给测试文档添加一个类别列。
df_test['class'] = df_test['class'] + 1  #问题4：为啥又要给所有类别分别加1
df_result = df_test.loc[:, ['id', 'class']]  #从测试集里面拿到'id', 'class']]列的内容
df_result.to_csv('/Users/apple/Documents/ST/python/competition/DataCastel/text_intelligence/new_data/result.csv', index = False) #测试结果转为提交的CSV格式

print('end')





'''
train_set.csv 1.5G, 普通电脑打开很吃力，随意阅读也吃力，谨慎打开；
第一行有：ID，article， Word_seg, class;
id:文章数量编号102277个文本； article：文章内容，是一些数字； Word_seg:也是一些数字； class:文本对应的类别从1到20

test_set.csv 1.38Gb,
第一行有：ID，article， Word_sequence,内容和训练集一样，只是没有了类别标签

result.csv 865 KB, 
第一行有ID， class；也就是预测每一个文档的类别

提交说明

1) 以csv格式提交，编码为UTF-8，第一行为表头； 
2) 内含两列，一列为id，另一列为class； 
3) id对应测试集中样本的id，class为参赛者的模型预测的文本标签。


y_train = df_train['class'] - 1 内容：
  import imp
0         13
1          2
2         11
3         12
4         11
5         12
102274    11
102275     3
102276    10
Name: class, Length: 102277, dtype: int64
[Finished in 25.6s]


y_train = df_train['class']内容：
  import imp
0         14
1          3
2         12
102274    12
102275     4
102276    11
Name: class, Length: 102277, dtype: int64
[Finished in 24.3s]

区别就是给每个类的类型都减去1；
'''

'''
得分：0.73

拿高分的方法：
数据预处理：这里的数据比较完整，不用担心

特征工程
这里技巧很足，需要不断的积累

机器学习算法
不同算法都有对应的任务类型。
lightboard微软开发的工具，适合大部分的情况，属于西瓜书第八章的内容。

数据增强：
给了1万条数据，变成10万条数据。



输出内容：
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

'''