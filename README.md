#  ActiveLearning4NER
通过NER任务进行主动学习,分别基于softmax和globalpointer实现  
# Requirements
- Python 3.7
- Tensorflow 1.14 
- bert4keras 0.11.3
- fairies 0.1.34
# 论文
Deep Active Learning for Named Entity Recognition 
# 博客
https://blog.csdn.net/HGlyh/article/details/118524845?spm=1001.2014.3001.5501
# 使用
运行softmax_active_learning.py 通过softmax实现NER任务并进行主动学习(需要较多轮次的训练)  
运行globalpointer_active_learning.py 通过globalpointer实现NER任务并进行主动学习  
# 效果
data/test.json中的数据未参与训练,用于测试主动学习效果    
example/中的数据是主动学习结果样例  
对于softmax的NER任务 文本的entry_MNLP_confidence分数越低,越值得被标注  
对于globalPoiner的NER任务 文本的entry_MNLP_confidence分数越高,越值得被标注
(实现原理相同,分数计算方式不同)  





