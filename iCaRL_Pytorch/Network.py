import torch.nn as nn

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        # 特征提取器
        self.feature = feature_extractor
        # 获取特征提取器最后一层的输入维度作为线性分类器的输入
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        # (batch_size, 3, 32, 32) -> (batch_size, 512) -> (batch_size, numclass)
        x = self.feature(input)
        x = self.fc(x)
        return x

    # 动态扩展分类器
    def Incremental_learning(self, numclass):
        # 保存现有(旧)参数
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features # 输入特征维度（512）
        out_feature = self.fc.out_features # 当前输出类别数

        # 创建新分类器
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        '''
        输入维度保持不变（in_feature = 512）
        输出维度扩展到新的类别数（numclass）
        '''

        # 复制旧参数到前 out_feature 个权重
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.feature(inputs)
