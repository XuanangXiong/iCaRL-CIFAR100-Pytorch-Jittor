from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import torch

numclass=10 # 初始类数量（0-9个类）
task_size=10 # 每个任务新增类数量（10-19,...,100-109）

feature_extractor=resnet18_cbam()
img_size=32

batch_size=128
memory_size=2000 # 代表样本记忆缓冲区容量
'''
随任务增加，每个类别的样本数减少：
任务1后: 2000/10 = 200个样本/类别
...
任务10后: 2000/100 = 20个样本/类别
'''
epochs=100
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

for i in range(10):
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
