# iCaRL-CIFAR100-Pytorch-Jittor
分别基于Pytorch框架和Jittor框架在CIFAR100数据集上实现了连续学习模型iCaRL

## 环境配置
### Pytorch框架环境
Python3.12.3
PyTorch2.5.1 + cu124

### Jittor框架环境
Python3.8
Jittor1.3.1 + cu113

### 其他库
PIL
matplotlib
seaborn
numpy

## 运行指令

```shell
python -u main.py
```

## 项目结构
### iCaRL_Pytorch & iCaRL_Jittor
- main.py 训练执行脚本
- iCaRL.py iCaRL模型核心算法实现，包括均值原型分类器和蒸馏等
- Network.py 网络架构
- ResNet.py - 特征提取器
- iCIFAR100.py - 数据集处理
- results 准确率曲线，遗忘矩阵与实验日志

### Result_Compute
- Pytorch_vs_Jittor_log.py 框架性能比对与Loss曲线计算

## 结果
### Pytorch框架结果
#### 当前任务准确率与累计任务准确率变化曲线
<img width="4491" height="1468" alt="accuracy_curves_task_9" src="https://github.com/user-attachments/assets/d16392c5-6989-45a0-96c9-0e4dd6b5c083" />

#### 遗忘矩阵

<img style="width: 50%;" alt="forgetting_matrix_task_9" src="https://github.com/user-attachments/assets/a838dd39-64a9-4cc5-872c-ec310b77aa3c" />

### Jittor框架结果
#### 当前任务准确率与累计任务准确率变化曲线
<img width="3600" height="1200" alt="accuracy_curves_task_9" src="https://github.com/user-attachments/assets/69db025b-5656-43e8-aa81-8cbe09a5b98c" />

#### 遗忘矩阵

<img style="width: 50%;" alt="forgetting_matrix_task_9" src="https://github.com/user-attachments/assets/d5da73ff-57d7-49df-b2b7-b7678b5639de" />


### Pytorch vs Jittor 对比结果

#### 任务准确率对比
<img width="4470" height="2966" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/a1e80e97-9611-4156-9c36-95a3ba09d23e" />

#### 平均遗忘与总遗忘对比
<img width="4470" height="1466" alt="forgetting_comparison" src="https://github.com/user-attachments/assets/6244d6dc-5d13-42b4-af93-8f932384e1e6" />

#### Loss变化曲线对比
<img width="4468" height="2365" alt="loss_comparison" src="https://github.com/user-attachments/assets/a3026c4e-9069-4cec-abc1-6d7f7cf9bccb" />

#### 对比总结
#### ![252f75d25d808f199c55b8238a3a8df2](https://github.com/user-attachments/assets/2ac30ca8-233e-4f9f-a692-7e5a5ee2c66f)

## 说明
- Learning Rate在本项目中对结果影响巨大，当lr较大时(例如lr=2)，参数在新任务上快速下降损失，短期内新类精度高，但特征空间剧烈漂移，因此旧类的原型均值与特征提取器不再对齐，导致旧类决策边界错位，增大了遗忘率；当lr较小时(例如lr=0.1)，很明显遗忘率大幅度降低，有时甚至会出现遗忘矩阵中参数为负的情况，但参数更新缓慢，难以学习新任务导致正确率较低，以下是Pytorch框架的iCaRL在lr=0.1时的性能：
### 当前任务准确率与累计任务准确率变化曲线(Pytorch,lr=0.1)
<img width="4491" height="1468" alt="accuracy_curves_task_9" src="https://github.com/user-attachments/assets/6c0fd4aa-c5a5-4347-a088-e56773a242b3" />

#### 遗忘矩阵(Pytorch,lr=0.1)
<img style="width: 50%" alt="forgetting_matrix_task_9" src="https://github.com/user-attachments/assets/aff5b066-48c7-4e9b-acd2-9764151eda2b" />

- lr的值对正确率-遗忘率的影响正是连续学习中很经典的稳定-可塑权衡，由于暂时缺少计算资源，计划今后为项目添加lr快速1调参脚本以寻找稳定-可塑平衡点。
- Jittor框架在本项目中性能明显低于Pytorch脚本，可能有以下原因：
  * 项目源于Pytorch，迁移时各类函数性能差异与迁移产生的Bug导致性能下降
  * 数据增强部分Jittor的RandomCrop缺少Padding参数，手动实现导致性能差异
  * 损失函数计算时scatter的具体操作在两个框架中的区别导致性能差异
  * Jittor手动初始化参数可能破坏了网络收敛性
- 本项目在Autodl服务器部署并使用一张A800显卡进行训练，Jittor框架的训练时间约为1.5h，Pytorch框架的训练时间约为2.5h，Jittor的训练速度明显大于Pytorch框架。








