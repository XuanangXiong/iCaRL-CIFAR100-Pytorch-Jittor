
from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import jittor as jt

# 设置Jittor使用GPU（如果可用）
# Jittor文档中并未记录get_divice_count方法，但是有卡的时候可以返回布尔值1
# 使用jt.has_cuda可能会引发报错
if jt.get_device_count() > 0:
    jt.flags.use_cuda = 1
    print(f"Using CUDA device, device count: {jt.get_device_count()}")
else:
    print("CUDA not available, using CPU")

numclass = 10  # 初始类数量（0-9个类）
task_size = 10  # 每个任务新增类数量（10-19,...,100-109）

feature_extractor = resnet18_cbam()
img_size = 32

batch_size = 128
memory_size = 2000  # 代表样本记忆缓冲区容量
'''
随任务增加，每个类别的样本数减少：
任务1后: 2000/10 = 200个样本/类别
...
任务10后: 2000/100 = 20个样本/类别
'''
epochs = 100
learning_rate = 1.5

print("=" * 60)
print("iCaRL持续学习项目 - Jittor版本")
print("=" * 60)
print(f"配置信息:")
print(f"- 初始类数量: {numclass}")
print(f"- 每任务类数量: {task_size}")
print(f"- 批次大小: {batch_size}")
print(f"- 记忆容量: {memory_size}")
print(f"- 训练轮数: {epochs}")
print(f"- 学习率: {learning_rate}")
print("=" * 60)

model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)

print("\n开始训练10个连续学习任务...")
for i in range(10):
    print(f"\n{'=' * 30} 开始任务 {i + 1} {'=' * 30}")
    print(f"当前学习类别: {i * task_size}-{(i + 1) * task_size - 1}")
    print(f"累计学习类别: 0-{(i + 1) * task_size - 1}")

    model.beforeTrain()
    accuracy = model.train()
    model.afterTrain(accuracy)

    print(f"任务 {i + 1} 完成!")
    print(f"当前总类别数: {model.numclass}")
    print(f"历史样本集数量: {len(model.exemplar_set)}")

print("\n" + "=" * 60)
print("所有任务训练完成！")
print("结果文件位置：")
print("1. 模型文件: model/ 目录")
print("2. 可视化结果: results/ 目录")
print("3. 准确率曲线: results/accuracy_curves_task_*.png")
print("4. 遗忘矩阵: results/forgetting_matrix_task_*.png")
print("5. 详细数据: results/results_task_*.json")
print("=" * 60)