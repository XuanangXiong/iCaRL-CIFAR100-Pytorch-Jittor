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
