import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import os
from PIL import Image
import torch.optim as optim
from Network import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader

# 修复matplotlib的Qt问题
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import warnings

warnings.filterwarnings('ignore')  # 忽略警告信息

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate):

        super(iCaRLmodel, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass

        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.test_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                      # transforms.Resize(img_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                           (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100('dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.test_loader = None

        # 添加性能追踪变量
        self.current_task = 0
        self.performance_matrix = {}  # 存储每个任务在每个时间点的准确率
        self.forgetting_matrix = {}  # 遗忘矩阵
        self.current_accuracies = []  # 当前任务准确率
        self.cumulative_accuracies = []  # 累计准确率
        self.nms_current_accuracies = []  # NMS当前任务准确率
        self.nms_cumulative_accuracies = []  # NMS累计准确率
        self.loss_history = []  # 记录每个epoch的loss

        # 确保results目录存在
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created directory: {self.results_dir}")

    def beforeTrain(self):
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    def train(self):
        print(f"开始训练任务 {self.current_task}，将训练 {self.epochs} 个epochs...")
        accuracy = 0
        epoch_losses = []  # 记录当前任务的所有epoch loss
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=1.0 / 5, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 5
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass > self.task_size:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 25
                else:
                    opt = optim.SGD(self.model.parameters(), lr=1.0 / 25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                if self.numclass == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=1.0 / 125, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 125
                print("change learning rate:%.3f" % (self.learning_rate / 100))

            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            num_batches = 0

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()

                epoch_loss += loss_value.item()
                num_batches += 1

                if step % 10 == 0:  # 每10步打印一次，减少输出
                    print('  step:%d,loss:%.3f' % (step, loss_value.item()))

            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)

            accuracy = self._test(self.test_loader, 1)
            print('  epoch:%d,accuracy:%.3f' % (epoch, accuracy))

        self.loss_history.append(epoch_losses)
        print(f"任务 {self.current_task} 训练完成，最终准确率: {accuracy:.3f}%")
        return accuracy

    def _test(self, testloader, mode):
        if mode == 0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, indexs, imgs, target):
        output = self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

    def evaluate_all_tasks(self, mode=1):
        """评估所有已学习任务的性能"""
        task_accuracies = {}

        print(f"  评估所有 {self.current_task + 1} 个任务...")
        for task_id in range(self.current_task + 1):
            # 为每个任务创建测试数据
            start_class = task_id * self.task_size
            end_class = (task_id + 1) * self.task_size

            # 创建临时测试数据集
            temp_test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=False)
            temp_test_dataset.getTestData([start_class, end_class])
            temp_test_loader = DataLoader(dataset=temp_test_dataset, shuffle=False, batch_size=self.batchsize)

            # 测试该任务
            accuracy = self._test(temp_test_loader, mode)
            task_accuracies[task_id] = accuracy.item()
            print(f"    任务 {task_id}: {accuracy:.2f}%")

        return task_accuracies

    def afterTrain(self, accuracy):
        print("开始后处理...")
        self.model.eval()
        m = int(self.memory_size / self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass - self.task_size, self.numclass):
            print('construct class %s examplar:' % (i), end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m)
        self.numclass += self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy = self._test(self.test_loader, 0)
        print("NMS accuracy：" + str(KNN_accuracy.item()))

        # 保存模型
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        total_classes = self.numclass  # 当前学到的总类别数
        filename = f'model/accuracy_{accuracy:.3f}_KNN_accuracy_{KNN_accuracy:.3f}_increment_{total_classes}_net.pkl'
        torch.save(self.model, filename)
        print(f"Model saved to: {filename}")

        self.old_model = torch.load(filename, weights_only=False)
        self.old_model.to(device)
        self.old_model.eval()

        # 性能分析和可视化
        try:
            print("开始性能分析...")
            self._update_performance_tracking(accuracy, KNN_accuracy)
            print("开始生成可视化...")
            self._update_visualization()
            print("后处理完成！")
        except Exception as e:
            print(f"可视化过程出错，但不影响主流程: {e}")
            # 即使可视化失败，也要更新任务计数
            self.current_task += 1

    def _update_performance_tracking(self, fc_accuracy, nms_accuracy):
        """更新性能追踪数据"""

        # 评估所有任务的性能
        print("使用FC分类器评估所有任务...")
        fc_task_accuracies = self.evaluate_all_tasks(mode=1)

        print("使用NMS分类器评估所有任务...")
        nms_task_accuracies = self.evaluate_all_tasks(mode=0)

        # 更新性能矩阵
        for task_id, acc in fc_task_accuracies.items():
            if task_id not in self.performance_matrix:
                self.performance_matrix[task_id] = {}
            self.performance_matrix[task_id][self.current_task] = acc

        # 计算遗忘矩阵
        self._compute_forgetting_matrix()

        # 计算当前任务准确率（只针对新学的任务）
        current_fc_acc = fc_task_accuracies[self.current_task]
        current_nms_acc = nms_task_accuracies[self.current_task]

        # 计算累计准确率（所有学过任务的平均）
        cumulative_fc_acc = np.mean(list(fc_task_accuracies.values()))
        cumulative_nms_acc = np.mean(list(nms_task_accuracies.values()))

        # 存储结果
        self.current_accuracies.append(current_fc_acc)
        self.cumulative_accuracies.append(cumulative_fc_acc)
        self.nms_current_accuracies.append(current_nms_acc)
        self.nms_cumulative_accuracies.append(cumulative_nms_acc)

        print(f"任务 {self.current_task}: FC 当前={current_fc_acc:.3f}%, FC 累计={cumulative_fc_acc:.3f}%")
        print(f"任务 {self.current_task}: NMS 当前={current_nms_acc:.3f}%, NMS 累计={cumulative_nms_acc:.3f}%")

        self.current_task += 1

    def _compute_forgetting_matrix(self):
        """计算遗忘矩阵"""
        for task_id in self.performance_matrix:
            if task_id not in self.forgetting_matrix:
                self.forgetting_matrix[task_id] = {}

            # 获取该任务刚学完时的性能（基准）
            if task_id in self.performance_matrix[task_id]:
                baseline_acc = self.performance_matrix[task_id][task_id]

                # 计算在每个时间点的遗忘程度
                for time_point, current_acc in self.performance_matrix[task_id].items():
                    if time_point >= task_id:  # 只考虑该任务学完之后的时间点
                        forgetting = baseline_acc - current_acc
                        self.forgetting_matrix[task_id][time_point] = forgetting

    def _update_visualization(self):
        """更新可视化图表"""
        print("  生成准确率曲线...")
        self._plot_accuracy_curves()
        print("  生成遗忘矩阵...")
        self._plot_forgetting_matrix()
        print("  保存结果数据...")
        self._save_results_to_json()

    def _plot_accuracy_curves(self):
        """绘制准确率曲线"""
        try:
            plt.figure(figsize=(15, 5))

            # 子图1：当前任务准确率对比
            plt.subplot(1, 3, 1)
            tasks = list(range(len(self.current_accuracies)))
            plt.plot(tasks, self.current_accuracies, 'b-o', label='FC Classifier', linewidth=2, markersize=6)
            plt.plot(tasks, self.nms_current_accuracies, 'r-s', label='NMS Classifier', linewidth=2, markersize=6)
            plt.xlabel('Task ID')
            plt.ylabel('Accuracy (%)')
            plt.title('Current Task Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 子图2：累计准确率对比
            plt.subplot(1, 3, 2)
            plt.plot(tasks, self.cumulative_accuracies, 'b-o', label='FC Classifier', linewidth=2, markersize=6)
            plt.plot(tasks, self.nms_cumulative_accuracies, 'r-s', label='NMS Classifier', linewidth=2, markersize=6)
            plt.xlabel('Task ID')
            plt.ylabel('Accuracy (%)')
            plt.title('Cumulative Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 子图3：性能对比表格
            plt.subplot(1, 3, 3)
            plt.axis('tight')
            plt.axis('off')

            # 创建表格数据
            table_data = []
            for i in range(len(self.current_accuracies)):
                table_data.append([
                    f'Task {i}',
                    f'{self.current_accuracies[i]:.2f}%',
                    f'{self.nms_current_accuracies[i]:.2f}%',
                    f'{self.cumulative_accuracies[i]:.2f}%',
                    f'{self.nms_cumulative_accuracies[i]:.2f}%'
                ])

            table = plt.table(cellText=table_data,
                              colLabels=['Task', 'FC Current', 'NMS Current', 'FC Cumulative', 'NMS Cumulative'],
                              cellLoc='center',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            plt.title('Detailed Results')

            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/accuracy_curves_task_{self.current_task - 1}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()
            print(f"    准确率曲线已保存")
        except Exception as e:
            print(f"    准确率曲线生成失败: {e}")

    def _plot_forgetting_matrix(self):
        """绘制遗忘矩阵"""
        try:
            if len(self.forgetting_matrix) == 0:
                return

            # 创建遗忘矩阵数组
            max_task = max(self.forgetting_matrix.keys())
            forgetting_array = np.full((max_task + 1, self.current_task), np.nan)

            for task_id in self.forgetting_matrix:
                for time_point, forgetting in self.forgetting_matrix[task_id].items():
                    if time_point < self.current_task:
                        forgetting_array[task_id][time_point] = forgetting

            # 绘制热力图
            plt.figure(figsize=(12, 8))

            # 使用mask来隐藏NaN值和对角线下方的值
            mask = np.isnan(forgetting_array)
            for i in range(forgetting_array.shape[0]):
                for j in range(forgetting_array.shape[1]):
                    if j < i:  # 对角线下方的值设为mask
                        mask[i][j] = True

            sns.heatmap(forgetting_array,
                        annot=True,
                        fmt='.2f',
                        cmap='RdYlBu_r',  # 红色表示遗忘严重，蓝色表示遗忘轻微
                        center=0,
                        mask=mask,
                        square=True,
                        linewidths=0.5,
                        cbar_kws={'label': 'Forgetting (%)'},
                        xticklabels=[f'After Task {i}' for i in range(self.current_task)],
                        yticklabels=[f'Task {i}' for i in range(max_task + 1)])

            plt.title('Forgetting Matrix\n(Positive values indicate forgetting, Negative values indicate improvement)')
            plt.xlabel('Time Point (After Learning Task)')
            plt.ylabel('Task Being Evaluated')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/forgetting_matrix_task_{self.current_task - 1}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()
            print(f"    遗忘矩阵已保存")

            # 输出遗忘统计信息
            self._print_forgetting_statistics()
        except Exception as e:
            print(f"    遗忘矩阵生成失败: {e}")

    def _print_forgetting_statistics(self):
        """输出遗忘统计信息"""
        try:
            print("\n" + "=" * 50)
            print("遗忘分析")
            print("=" * 50)

            total_forgetting = 0
            total_measurements = 0

            for task_id in self.forgetting_matrix:
                task_forgettings = []
                print(f"\n任务 {task_id}:")
                for time_point, forgetting in self.forgetting_matrix[task_id].items():
                    if time_point > task_id:  # 排除基准时间点
                        task_forgettings.append(forgetting)
                        print(f"  学完任务 {time_point} 后: {forgetting:.2f}% 遗忘")
                        total_forgetting += forgetting
                        total_measurements += 1

                if task_forgettings:
                    avg_forgetting = np.mean(task_forgettings)
                    max_forgetting = np.max(task_forgettings)
                    print(f"  平均遗忘: {avg_forgetting:.2f}%")
                    print(f"  最大遗忘: {max_forgetting:.2f}%")

            if total_measurements > 0:
                overall_avg_forgetting = total_forgetting / total_measurements
                print(f"\n总体平均遗忘: {overall_avg_forgetting:.2f}%")

            print("=" * 50)
        except Exception as e:
            print(f"遗忘统计输出失败: {e}")

    def _save_results_to_json(self):
        """保存结果到JSON文件"""
        try:
            results = {
                'current_task': self.current_task - 1,
                'fc_current_accuracies': self.current_accuracies,
                'fc_cumulative_accuracies': self.cumulative_accuracies,
                'nms_current_accuracies': self.nms_current_accuracies,
                'nms_cumulative_accuracies': self.nms_cumulative_accuracies,
                'performance_matrix': self.performance_matrix,
                'forgetting_matrix': self.forgetting_matrix,
                'loss_history': self.loss_history, # loss添加
                'hyperparameters': {
                    'task_size': self.task_size,
                    'memory_size': self.memory_size,
                    'batch_size': self.batchsize,
                    'epochs': self.epochs,
                    'learning_rate': self.learning_rate
                }
            }

            with open(f'{self.results_dir}/results_task_{self.current_task - 1}.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"    结果数据已保存")
        except Exception as e:
            print(f"    结果保存失败: {e}")

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))

        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        '''
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data
        '''
        for index in range(1, len(images)):
            data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data


    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s" % (str(index)))
            exemplar = self.exemplar_set[index]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean(exemplar, self.classify_transform)
            class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)