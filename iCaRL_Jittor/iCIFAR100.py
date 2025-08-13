
import numpy as np
from PIL import Image
import jittor as jt
from jittor.dataset import Dataset
import os
import pickle
import tarfile
from jittor_utils.misc import download_url_to_local

'''
CIFAR100：
60000张 32*32 像素的彩色图像，其中训练集50000张，测试集10000张，共100个类
'''

class iCIFAR100(Dataset):

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = ['train']
    test_list = ['test']
    meta_key = 'fine_label_names'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):

        super(iCIFAR100, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform

        if download:
            self.download()

        self._load_data()

        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        filepath = os.path.join(self.root, self.filename)
        if not os.path.exists(filepath):
            download_url_to_local(self.url, self.filename, self.root, self.tgz_md5)

        extract_path = os.path.join(self.root, "cifar-100-python")
        if not os.path.exists(extract_path):
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(self.root)

    # 加载CIFAR100数据
    def _load_data(self):
        base_folder = os.path.join(self.root, "cifar-100-python")

        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.targets = []

        for file_name in file_list:
            file_path = os.path.join(base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # 转换为HWC格式
        self.targets = np.array(self.targets)

    # 数据及标签合并
    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    # 训练数据准备
    def getTrainData(self, classes, exemplar_set):
        # classes: [start_class, end_class]

        # 历史样本集
        datas, labels = [], []
        if len(exemplar_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in range(len(exemplar_set))]

        # 新任务数据
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        # 合并
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    # 测试数据准备
    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)

        if isinstance(self.TestData, list) and len(self.TestData) == 0:
            self.TestData = datas
            self.TestLabels = labels
        elif isinstance(self.TestData, np.ndarray) and self.TestData.size == 0:
            self.TestData = datas
            self.TestLabels = labels
        else:
            self.TestData = np.concatenate((self.TestData, datas), axis=0)
            self.TestLabels = np.concatenate((self.TestLabels, labels), axis=0)

        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    # 获取训练数据
    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return index, img, target

    # 获取测试数据
    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        if self.target_test_transform:
            target = self.target_test_transform(target)

        return index, img, target

    # 统一数据访问接口
    def __getitem__(self, index):
        # 修复：使用更安全的判断方式
        if self._has_train_data():
            return self.getTrainItem(index)
        elif self._has_test_data():
            return self.getTestItem(index)
        else:
            raise RuntimeError("No data available. Please call getTrainData() or getTestData() first.")

    # 长度
    def __len__(self):
        # 修复：使用更安全的判断方式
        if self._has_train_data():
            return len(self.TrainData)
        elif self._has_test_data():
            return len(self.TestData)
        else:
            return 0

    # 辅助方法：检查是否有训练数据
    # 原代码：if self.TrainData != []:引发广播错误
    def _has_train_data(self):
        # 检查TrainData是否是numpy数组且不为空
        if isinstance(self.TrainData, list):
            return len(self.TrainData) > 0
        elif isinstance(self.TrainData, np.ndarray):
            return self.TrainData.size > 0
        else:
            return False

    # 辅助方法：检查是否有测试数据
    def _has_test_data(self):
        if isinstance(self.TestData, list):
            return len(self.TestData) > 0
        elif isinstance(self.TestData, np.ndarray):
            return self.TestData.size > 0
        else:
            return False

    # 根据标签获得class包含的img
    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]