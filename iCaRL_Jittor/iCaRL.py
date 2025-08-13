import jittor as jt
from jittor import nn, Module, transform
import numpy as np
from PIL import Image
import os
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class PyTorchStyleRandomCrop:
    """
    Jittor中的RandomCrop不支持padding参数
    定义新函数实现与PyTorch RandomCrop((32, 32), padding=4) 完全一致的行为
    """
    def __init__(self, size, padding=4):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('Input should be PIL Image')

        # 添加padding
        if self.padding > 0:
            from PIL import ImageOps
            img = ImageOps.expand(img, border=self.padding, fill=0)

        # 随机裁剪
        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return img

        if w < tw or h < th:
            raise ValueError(f'图像尺寸 ({w}, {h}) 小于目标尺寸 {self.size}')

        import random
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return img.crop((j, i, j + tw, i + th))


class iCaRLmodel:

    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate):
        super(iCaRLmodel, self).__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.numclass = numclass
        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        from Network import network
        self.model = network(numclass, feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.old_model = None

        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.ImageNormalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        self.train_transform = transform.Compose([
            PyTorchStyleRandomCrop(32, padding=4),  # 使用新定义的RandomCrop
            transform.RandomHorizontalFlip(p=0.5),
            transform.ColorJitter(brightness=0.24705882352941178),
            transform.ToTensor(),
            transform.ImageNormalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        # 测试变换
        self.test_transform = transform.Compose([
            transform.ToTensor(),
            transform.ImageNormalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        # 分类变换（用于exemplar）
        self.classify_transform = transform.Compose([
            transform.RandomHorizontalFlip(p=1.0),
            transform.ToTensor(),
            transform.ImageNormalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        # 数据集
        from iCIFAR100 import iCIFAR100
        self.train_dataset = iCIFAR100('dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=True)

        # 数据加载器
        self.train_loader = None
        self.test_loader = None

        # 性能追踪
        self.current_task = 0
        self.performance_matrix = {}
        self.forgetting_matrix = {}
        self.current_accuracies = []
        self.cumulative_accuracies = []
        self.nms_current_accuracies = []
        self.nms_cumulative_accuracies = []
        self.loss_history = []  # 记录每个epoch的loss

        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)


    def create_optimizer(self, lr):
        return nn.SGD(self.model.parameters(), lr=lr, weight_decay=0.00001)


    def update_learning_rate(self, optimizer, new_lr):
        optimizer.lr = new_lr
        return optimizer


    def Image_transform_fixed(self, images, transform_func):
        # 将图像列表转换为Jittor-Var张量
        data = []
        for img in images:
            # 先将NumPy数组转换为PIL图像，再通过transform处理
            transformed = transform(Image.fromarray(img))
            # 确保输出是Var
            if not isinstance(transformed, jt.Var):
                transformed = jt.array(transformed)
            transformed = transformed.unsqueeze(0)
            data.append(transformed)

        # 合并
        data = jt.concat(data, dim=0)
        return data


    def get_one_hot_fixed(self, target, num_class):
        one_hot = jt.zeros([target.shape[0], num_class])
        one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1),
                                  src=jt.ones_like(target.float().view(-1, 1)))
        return one_hot

    def _compute_loss_fixed(self, indexs, imgs, target):
        output = self.model(imgs)
        target_onehot = self.get_one_hot_fixed(target, self.numclass)

        if self.old_model is None:
            loss = nn.binary_cross_entropy_with_logits(output, target_onehot)
        else:
            with jt.no_grad():
                old_output = self.old_model(imgs)
                old_target = jt.sigmoid(old_output)

            old_task_size = old_target.shape[1]

            target_onehot[:, :old_task_size] = old_target.detach()

            loss = nn.binary_cross_entropy_with_logits(output, target_onehot)

        return loss


    def compute_class_mean_fixed(self, images, transform_func):
        x = self.Image_transform_fixed(images, transform_func)


        # 获取特征并归一化
        features = self.model.feature_extractor(x).detach()

        # 使用与PyTorch一致的L2归一化（虽然测试显示一致，但保险起见手动实现）
        features_normalized = features / jt.norm(features, dim=1, keepdims=True)

        feature_extractor_output = features_normalized.numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)

        return class_mean, feature_extractor_output


    def _construct_exemplar_set_fixed(self, images, m):
        print(f'构建exemplar集合，目标数量: {m}')

        class_mean, feature_extractor_output = self.compute_class_mean_fixed(images, self.transform)

        exemplar = []
        now_class_mean = np.zeros((1, 512))

        # 改进的exemplar选择策略
        for i in range(m):
            # 计算到类中心的距离
            target_mean = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            distances = np.linalg.norm(target_mean, axis=1)

            # 选择距离最小的样本
            index = np.argmin(distances)

            # 更新累积均值
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

            # 调试信息
            if i < 3:
                print(f'  选择exemplar {i + 1}: 索引={index}, 距离={distances[index]:.6f}')

        print(f"exemplar集合大小: {len(exemplar)}")
        self.exemplar_set.append(exemplar)

        self._validate_exemplar_quality(exemplar, len(self.exemplar_set) - 1)


    def _validate_exemplar_quality(self, exemplar, class_id):
        """验证exemplar质量"""
        if len(exemplar) == 0:
            return

        _, exemplar_features = self.compute_class_mean_fixed(exemplar, self.transform)
        mean_feature = np.mean(exemplar_features, axis=0)
        std_feature = np.std(exemplar_features, axis=0)

        pairwise_distances = []
        for i in range(len(exemplar_features)):
            for j in range(i + 1, len(exemplar_features)):
                dist = np.linalg.norm(exemplar_features[i] - exemplar_features[j])
                pairwise_distances.append(dist)

        if pairwise_distances:
            avg_pairwise_dist = np.mean(pairwise_distances)
            print(f"类别 {class_id} exemplar质量:")
            print(f"  特征均值模长: {np.linalg.norm(mean_feature):.6f}")
            print(f"  特征标准差均值: {np.mean(std_feature):.6f}")
            print(f"  平均成对距离: {avg_pairwise_dist:.6f}")


    def beforeTrain(self):
        """训练前准备 - 修复版本"""
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader_fixed(classes)

        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)

        self.model.train()


    def _get_train_and_test_dataloader_fixed(self, classes):
        """修复后的数据加载器"""
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)

        # 使用Jittor的数据加载方式
        self.train_dataset.set_attrs(batch_size=self.batchsize, shuffle=True)
        self.test_dataset.set_attrs(batch_size=self.batchsize, shuffle=False)  # 测试不需要shuffle

        return self.train_dataset, self.test_dataset


    def train(self):
        print(f"开始训练任务 {self.current_task}，将训练 {self.epochs} 个epochs...")
        epoch_losses = []
        opt = self.create_optimizer(self.learning_rate)

        for epoch in range(self.epochs):
            if epoch == 48:
                new_lr = self.learning_rate / 5
                opt = self.update_learning_rate(opt, new_lr)
                print(f"学习率调整为: {new_lr:.6f}")
            elif epoch == 62:
                new_lr = self.learning_rate / 25
                opt = self.update_learning_rate(opt, new_lr)
                print(f"学习率调整为: {new_lr:.6f}")
            elif epoch == 80:
                new_lr = self.learning_rate / 125
                opt = self.update_learning_rate(opt, new_lr)
                print(f"学习率调整为: {new_lr:.6f}")

            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            num_batches = 0

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = jt.array(images), jt.array(target)
                loss_value = self._compute_loss_fixed(indexs, images, target)

                opt.zero_grad()
                opt.backward(loss_value)
                opt.step()

                epoch_loss += loss_value.item()
                num_batches += 1

                if step % 10 == 0:
                    print(f'  step:{step}, loss:{loss_value.item():.3f}')

            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)  # 记录平均loss

            accuracy = self._test_fixed(self.test_loader, 1)
            print(f'  epoch:{epoch}, avg_loss:{avg_loss:.3f}, accuracy:{accuracy:.3f}')

        final_accuracy = self._test_fixed(self.test_loader, 1)
        print(f"任务 {self.current_task} 训练完成，最终准确率: {final_accuracy:.3f}%")
        return final_accuracy


    def _test_fixed(self, testloader, mode):
        if mode == 0:
            print("使用NMS分类器测试")

        self.model.eval()
        correct, total = 0, 0

        with jt.no_grad():
            for step, (indexs, imgs, labels) in enumerate(testloader):
                imgs, labels = jt.array(imgs), jt.array(labels)

                if mode == 1:
                    outputs = self.model(imgs)
                    predicts = jt.argmax(outputs, dim=1)[0]
                else:
                    predicts = self.classify_fixed(imgs)

                correct += (predicts.numpy() == labels.numpy()).sum()
                total += len(labels)

        accuracy = 100 * correct / total
        self.model.train()
        return accuracy


    def classify_fixed(self, test):
        result = []
        features = self.model.feature_extractor(test).detach()
        test_features = (features / jt.norm(features, dim=1, keepdims=True)).numpy()

        class_mean_set = np.array(self.class_mean_set)

        for target in test_features:
            distances = np.linalg.norm(target - class_mean_set, ord=2, axis=1)
            predicted_class = np.argmin(distances)
            result.append(predicted_class)

        return jt.array(result)


    def afterTrain(self, accuracy):
        print("开始后处理...")
        self.model.eval()

        # Exemplar管理
        m = int(self.memory_size / self.numclass)
        self._reduce_exemplar_sets_fixed(m)

        for i in range(self.numclass - self.task_size, self.numclass):
            print(f'构建类别 {i} 的exemplar集合:', end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set_fixed(images, m)

        self.numclass += self.task_size
        self.compute_exemplar_class_mean_fixed()

        self.model.train()

        KNN_accuracy = self._test_fixed(self.test_loader, 0)
        print(f"NMS accuracy: {KNN_accuracy:.3f}%")

        self._save_model_fixed(accuracy, KNN_accuracy)

        self._update_performance_tracking_fixed(accuracy, KNN_accuracy)
        self._update_visualization()


    def _save_model_fixed(self, accuracy, knn_accuracy):
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = f'model/accuracy_{accuracy:.3f}_KNN_accuracy_{knn_accuracy:.3f}_increment_{self.numclass}_net.pkl'

        self.model.save(filename)
        print(f"模型已保存: {filename}")

        if self.old_model is not None:
            del self.old_model

        from Network import network
        self.old_model = network(self.numclass - self.task_size, self.model.feature)
        self.old_model.load(filename)
        self.old_model.eval()


    def _reduce_exemplar_sets_fixed(self, m):
        for index in range(len(self.exemplar_set)):
            old_size = len(self.exemplar_set[index])
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            new_size = len(self.exemplar_set[index])
            print(f'类别 {index} exemplar: {old_size} -> {new_size}')


    def compute_exemplar_class_mean_fixed(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print(f"计算类别 {index} 的类均值")
            exemplar = self.exemplar_set[index]

            class_mean, _ = self.compute_class_mean_fixed(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean_fixed(exemplar, self.classify_transform)

            class_mean = class_mean / np.linalg.norm(class_mean)
            class_mean_ = class_mean_ / np.linalg.norm(class_mean_)
            final_mean = (class_mean + class_mean_) / 2

            self.class_mean_set.append(final_mean)


    def evaluate_all_tasks(self, mode=1):
        task_accuracies = {}

        print(f"  评估所有 {self.current_task + 1} 个任务...")
        for task_id in range(self.current_task + 1):
            start_class = task_id * self.task_size
            end_class = (task_id + 1) * self.task_size

            from iCIFAR100 import iCIFAR100
            temp_test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=False)
            temp_test_dataset.getTestData([start_class, end_class])
            temp_test_dataset.set_attrs(batch_size=self.batchsize, shuffle=False)

            accuracy = self._test_fixed(temp_test_dataset, mode)
            task_accuracies[task_id] = float(accuracy)
            print(f"    任务 {task_id}: {accuracy:.2f}%")

        return task_accuracies


    def _update_performance_tracking_fixed(self, fc_accuracy, nms_accuracy):
        print("使用FC分类器评估所有任务...")
        fc_task_accuracies = self.evaluate_all_tasks(mode=1)

        print("使用NMS分类器评估所有任务...")
        nms_task_accuracies = self.evaluate_all_tasks(mode=0)

        for task_id, acc in fc_task_accuracies.items():
            if task_id not in self.performance_matrix:
                self.performance_matrix[task_id] = {}
            self.performance_matrix[task_id][self.current_task] = acc

        self._compute_forgetting_matrix()

        current_fc_acc = fc_task_accuracies[self.current_task]
        current_nms_acc = nms_task_accuracies[self.current_task]
        cumulative_fc_acc = np.mean(list(fc_task_accuracies.values()))
        cumulative_nms_acc = np.mean(list(nms_task_accuracies.values()))

        self.current_accuracies.append(current_fc_acc)
        self.cumulative_accuracies.append(cumulative_fc_acc)
        self.nms_current_accuracies.append(current_nms_acc)
        self.nms_cumulative_accuracies.append(cumulative_nms_acc)

        print(f"任务 {self.current_task}: FC 当前={current_fc_acc:.3f}%, FC 累计={cumulative_fc_acc:.3f}%")
        print(f"任务 {self.current_task}: NMS 当前={current_nms_acc:.3f}%, NMS 累计={cumulative_nms_acc:.3f}%")

        self.current_task += 1


    def _compute_forgetting_matrix(self):
        for task_id in self.performance_matrix:
            if task_id not in self.forgetting_matrix:
                self.forgetting_matrix[task_id] = {}

            if task_id in self.performance_matrix[task_id]:
                baseline_acc = self.performance_matrix[task_id][task_id]

                for time_point, current_acc in self.performance_matrix[task_id].items():
                    if time_point >= task_id:
                        forgetting = baseline_acc - current_acc
                        self.forgetting_matrix[task_id][time_point] = forgetting

    def _update_visualization(self):
        try:
            self._plot_accuracy_curves()
            self._plot_forgetting_matrix()
            self._save_results_to_json()
        except Exception as e:
            print(f"可视化过程出错: {e}")

    def _plot_accuracy_curves(self):
        try:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            tasks = list(range(len(self.current_accuracies)))
            plt.plot(tasks, self.current_accuracies, 'b-o', label='FC Classifier')
            plt.plot(tasks, self.nms_current_accuracies, 'r-s', label='NMS Classifier')
            plt.xlabel('Task ID')
            plt.ylabel('Accuracy (%)')
            plt.title('Current Task Accuracy')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(tasks, self.cumulative_accuracies, 'b-o', label='FC Classifier')
            plt.plot(tasks, self.nms_cumulative_accuracies, 'r-s', label='NMS Classifier')
            plt.xlabel('Task ID')
            plt.ylabel('Accuracy (%)')
            plt.title('Cumulative Accuracy')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/accuracy_curves_task_{self.current_task - 1}.png', dpi=300)
            plt.close()
            print(f"准确率曲线已保存")
        except Exception as e:
            print(f"准确率曲线生成失败: {e}")

    def _plot_forgetting_matrix(self):
        try:
            if len(self.forgetting_matrix) == 0:
                return

            max_task = max(self.forgetting_matrix.keys())
            forgetting_array = np.full((max_task + 1, self.current_task), np.nan)

            for task_id in self.forgetting_matrix:
                for time_point, forgetting in self.forgetting_matrix[task_id].items():
                    if time_point < self.current_task:
                        forgetting_array[task_id][time_point] = forgetting

            plt.figure(figsize=(10, 6))
            mask = np.isnan(forgetting_array)
            for i in range(forgetting_array.shape[0]):
                for j in range(forgetting_array.shape[1]):
                    if j < i:
                        mask[i][j] = True

            sns.heatmap(forgetting_array, annot=True, fmt='.2f', cmap='RdYlBu_r',
                        center=0, mask=mask, square=True, linewidths=0.5,
                        xticklabels=[f'After Task {i}' for i in range(self.current_task)],
                        yticklabels=[f'Task {i}' for i in range(max_task + 1)])

            plt.title('Forgetting Matrix (% Accuracy Drop)')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/forgetting_matrix_task_{self.current_task - 1}.png', dpi=300)
            plt.close()
            print(f"遗忘矩阵已保存")
        except Exception as e:
            print(f"遗忘矩阵生成失败: {e}")

    def _save_results_to_json(self):
        try:
            results = {
                'current_task': self.current_task - 1,
                'fc_current_accuracies': self.current_accuracies,
                'fc_cumulative_accuracies': self.cumulative_accuracies,
                'nms_current_accuracies': self.nms_current_accuracies,
                'nms_cumulative_accuracies': self.nms_cumulative_accuracies,
                'performance_matrix': self.performance_matrix,
                'forgetting_matrix': self.forgetting_matrix,
                'loss_history': self.loss_history,  # 新添加
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
            print(f"结果数据已保存")
        except Exception as e:
            print(f"结果保存失败: {e}")


def apply_fixes_to_existing_model(original_model):
    """
    将修复应用到现有iCaRL模型的函数
    """
    print("=" * 60)
    print("应用Jittor iCaRL性能修复")
    print("=" * 60)

    # 保存原模型的基本属性
    fixed_model = iCaRLmodel(
        numclass=original_model.numclass,
        feature_extractor=original_model.model.feature,
        batch_size=original_model.batchsize,
        task_size=original_model.task_size,
        memory_size=original_model.memory_size,
        epochs=original_model.epochs,
        learning_rate=original_model.learning_rate
    )

    # 如果原模型已有状态，复制过来
    if hasattr(original_model, 'exemplar_set'):
        fixed_model.exemplar_set = original_model.exemplar_set
    if hasattr(original_model, 'class_mean_set'):
        fixed_model.class_mean_set = original_model.class_mean_set
    if hasattr(original_model, 'current_task'):
        fixed_model.current_task = original_model.current_task

    print("✓ 修复已应用:")
    print("  - 数据增强：PyTorchStyleRandomCrop + ColorJitter")
    print("  - 图像变换：复现PyTorch的'有益bug'")
    print("  - 优化器：修复学习率调度")
    print("  - 损失计算：改进知识蒸馏")
    print("  - Exemplar：增强质量验证")
    print("  - 模型保存：修复加载机制")

    return fixed_model



