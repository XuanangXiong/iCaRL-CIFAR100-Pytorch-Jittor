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

<img style="width: 50%;" alt="forgetting_matrix_task_9" src="https://github.com/user-attachments/assets/58580c33-bb04-4157-8ca9-afe6b3815abc" />

### Pytorch vs Jittor 对比结果

#### 任务准确率对比
<img width="4470" height="2966" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/a1e80e97-9611-4156-9c36-95a3ba09d23e" />

#### 平均遗忘与总遗忘对比
<img width="4470" height="1466" alt="forgetting_comparison" src="https://github.com/user-attachments/assets/6244d6dc-5d13-42b4-af93-8f932384e1e6" />

#### Loss变化曲线对比
<img width="4468" height="2365" alt="loss_comparison" src="https://github.com/user-attachments/assets/a3026c4e-9069-4cec-abc1-6d7f7cf9bccb" />

#### ![252f75d25d808f199c55b8238a3a8df2](https://github.com/user-attachments/assets/2ac30ca8-233e-4f9f-a692-7e5a5ee2c66f)






