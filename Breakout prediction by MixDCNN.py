import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torchvision import datasets
from torchvision.datasets import MNIST, CIFAR10
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, roc_curve, auc
import torch.nn.functional as F
# from torchviz import make_dot
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg   # mpimg 用于读取图片
import numpy as np
import pandas as pd
import os
import random
import time

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False


# hyperparameters
batch_size_train = 32
batch_size_validation = 16
batch_size_test = 16
Epochs = 5000
test_G_mean_benchmark = 0.99


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare dataset

#create dataset class
class Mydataset(Dataset):
    def __init__(self, train_sub_dataset = 0, test_sub_dataset = 0, transform = None):
        super(Mydataset, self).__init__()
        self.filename = ".\\Breakout Image Index.txt"
        # 使用names参数给数据增加表头
        images_indices = pd.read_csv(self.filename, names = ['index', 'plant', 'date', 'time', 'mold', 'type'],
                            encoding = 'gb2312', header = None, sep = "\t").values

        image_type = images_indices[:, -1]
        image_index = images_indices[:, 0]
        # 数据标签
        images_labels = np.array([1 if type == '漏钢' else 0 for type in image_type])

        # 分层采样划分训练集和测试集
        # n_splits为划分组数(每组对应不同的划分方式), test_size为测试集比例, random_state设置随机状态(每次划分结果相同)
        split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.20)

        for train_index, test_index in split.split(images_indices, images_labels):
            # 对训练集中的漏钢样本过采样
            index_breakout_in_train_index = np.array([tr for tr in train_index if images_labels[tr] == 1])      # train_index中的漏钢样本索引
            index_nonbreakout_in_train_index = np.array([tr for tr in train_index if images_labels[tr] == 0])   # train_index中的非漏钢样本索引

            train_breakout_num = len(index_breakout_in_train_index)
            train_nonbreakout_num = len(index_nonbreakout_in_train_index)

            imbalance_coeff = train_nonbreakout_num // train_breakout_num
            index_breakout_new = index_breakout_in_train_index.repeat(imbalance_coeff)
            train_index_new = np.concatenate((index_breakout_new, index_nonbreakout_in_train_index), axis=0)

            self.train_images_indices, self.test_images_indices = images_indices[train_index_new], images_indices[test_index]
            self.y_train, self.y_test = np.array([images_labels[tr] for tr in train_index_new]), np.array([images_labels[te] for te in test_index])
            self.index_train, self.index_test = np.array([image_index[tr] for tr in train_index_new]), np.array([image_index[te] for te in test_index])




        # 对过采样之后的训练集划分为三个无重叠的子样本集用于后续3个子网络模型的训练
        train_split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.33)

        for train_index_1_2, train_index_3 in train_split1.split(self.train_images_indices, self.y_train):
            self.train_images_1_2_indices, self.train_images_3_indices = self.train_images_indices[train_index_1_2], self.train_images_indices[train_index_3]
            self.y_images_1_2, self.y_images_3 = np.array([self.y_train[idx_1_2] for idx_1_2 in train_index_1_2]), np.array([self.y_train[idx_3] for idx_3 in train_index_3])


        train_split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50)

        for train_index_1, train_index_2 in train_split2.split(self.train_images_1_2_indices, self.y_images_1_2):
            self.train_images_1_indices, self.train_images_2_indices = self.train_images_1_2_indices[train_index_1], self.train_images_1_2_indices[train_index_2]
            self.y_images_1, self.y_images_2 = np.array([self.y_images_1_2[idx1] for idx1 in train_index_1]), np.array([self.y_images_1_2[idx2] for idx2 in train_index_2])



        # 对原数据集中的测试集划分为三个无重叠的子样本集用于后续3个子网络模型的测试输出
        test_split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.33)

        for test_index_1_2, test_index_3 in test_split1.split(self.test_images_indices, self.y_test):
            self.test_images_1_2_indices, self.test_images_3_indices = self.test_images_indices[test_index_1_2], self.test_images_indices[test_index_3]
            self.y_test_1_2, self.y_test_3 = np.array([self.y_test[idx_1_2] for idx_1_2 in test_index_1_2]), np.array([self.y_test[idx_3] for idx_3 in test_index_3])


        test_split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50)

        for test_index_1, test_index_2 in test_split2.split(self.test_images_1_2_indices, self.y_test_1_2):
            self.test_images_1_indices, self.test_images_2_indices = self.test_images_1_2_indices[test_index_1], self.test_images_1_2_indices[test_index_2]
            self.y_test_1, self.y_test_2 = np.array([self.y_test_1_2[idx1] for idx1 in test_index_1]), np.array([self.y_test_1_2[idx2] for idx2 in test_index_2])




        self.transform = transform

        if train_sub_dataset == 1:
            self.samples_indices = self.train_images_1_indices
            self.samples_labels = self.y_images_1
        elif train_sub_dataset == 2:
            self.samples_indices = self.train_images_2_indices
            self.samples_labels = self.y_images_2
        elif train_sub_dataset == 3:
            self.samples_indices = self.train_images_3_indices
            self.samples_labels = self.y_images_3
        else:
            pass


        if test_sub_dataset == 1:
            self.samples_indices = self.test_images_1_indices
            self.samples_labels = self.y_test_1
        elif test_sub_dataset == 2:
            self.samples_indices = self.test_images_2_indices
            self.samples_labels = self.y_test_2
        elif test_sub_dataset == 3:
            self.samples_indices = self.test_images_3_indices
            self.samples_labels = self.y_test_3
        else:
            pass



        # self.samples_index = self.index_train if is_train_set else self.index_test


    def __len__(self):
        return len(self.samples_indices)


    def __getitem__(self, idx):
        index, plant, date, time, mold, type = self.samples_indices[idx]
        plant_path = ".\\" + plant + "图像\\"
        abnormal_type = "漏钢样本\\"
        sample_type = type + "\\"
        datenew = '.'.join(date.split('/'))
        timenew = '.'.join(time.split(':'))
        image_info = datenew + "  " + timenew + "  " + mold + ".jpg"
        img_path = os.path.join(plant_path, abnormal_type, sample_type, image_info)
        image = mpimg.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.samples_labels[idx]


# 所以resize的根本原因有两点：1.可以使用更小的卷积核减小运算量；2.更小的输入能够使得模型复杂度不至于过高，降低过拟合风险。
transform_train = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([224, 224]),
                                transforms.RandomHorizontalFlip(),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
# transforms.Normalize((0.9781, 0.9267, 0.9266), (0.0503, 0.1843, 0.1844)),

transform_test = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([224, 224]),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
# transforms.Normalize((0.9781, 0.9267, 0.9266), (0.0503, 0.1843, 0.1844)),


# 只对训练集应用数据增强
train_subset1 = Mydataset(train_sub_dataset = 1, transform = transform_train)
train_data_loader1 = DataLoader(train_subset1, batch_size = len(train_subset1), shuffle=True)

train_subset2 = Mydataset(train_sub_dataset = 2, transform = transform_train)
train_data_loader2 = DataLoader(train_subset2, batch_size = len(train_subset2), shuffle=True)

train_subset3 = Mydataset(train_sub_dataset = 3, transform = transform_train)
train_data_loader3 = DataLoader(train_subset3, batch_size = len(train_subset3), shuffle=True)


# 测试集(分为3个子集)
test_subset1 = Mydataset(test_sub_dataset= 1, transform = transform_test)
test_data_loader1 = DataLoader(test_subset1, batch_size = len(test_subset1), shuffle=False)

test_subset2 = Mydataset(test_sub_dataset = 2, transform = transform_test)
test_data_loader2 = DataLoader(test_subset2, batch_size = len(test_subset2), shuffle=False)

test_subset3 = Mydataset(test_sub_dataset = 3, transform = transform_test)
test_data_loader3 = DataLoader(test_subset3, batch_size = len(test_subset3), shuffle=False)



# test_data = Mydataset(train_sub_dataset = 0, transform = transform_test)
# test_data_loader = DataLoader(test_data, batch_size = len(test_data), shuffle = False)

# BK_test_loader_rt = DataLoader(BK_test_data, batch_size = 1, shuffle = False)


# LC_train_data = MNIST(root ='../dataset/samples/', train=True, download=False, transform=transform)
# LC_train_loader = DataLoader(LC_train_data, shuffle=True, batch_size=batch_size)
# LC_test_data = MNIST(root = '../dataset/samples/', train=False, download=False, transform=transform)
# LC_test_loader_pic = DataLoader(LC_test_data, shuffle=False, batch_size=batch_size)




# transform1 = transforms.Compose([transforms.ToTensor()])
# train_dataset = CIFAR10(root='G:\\下载\\2021-2022 第二学期\\课题拷贝 2022.05.31\\dataset\\', train=True, download=True, transform=transform1)
# train_loader_CF = DataLoader(train_dataset, shuffle=True, batch_size=len(train_dataset))

# # 计算自定义数据集的均值和方差，normalize之后有助于网络收敛
# imgs = [item[0] for item in train_data_loader] # item[0] and item[1] are image and its label
# # imgs = torch.stack(imgs, dim=0).numpy()
# imgs = np.array(imgs[0])
# # calculate mean over each channel (r,g,b)
# mean_r = imgs[:, 0, :, :].mean()
# mean_g = imgs[:, 1, :, :].mean()
# mean_b = imgs[:, 2, :, :].mean()
# print(mean_r, mean_g, mean_b)   # 0.97810775 0.9266955 0.9266079
#
# # calculate std over each channel (r,g,b)
# std_r = imgs[:, 0, :, :].std()
# std_g = imgs[:, 1, :, :].std()
# std_b = imgs[:, 2, :, :].std()
# print(std_r, std_g, std_b)  # 0.0503033 0.18429644 0.18435793


# #实现softmax函数将线性层输出转化为概率值
# def softmax(x, axis=1):
#     # 计算每行的最大值
#     row_max = x.max(axis=axis)[0]
#
#     # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
#     #row_max = row_max.reshape(-1, 1)
#     x = x - row_max
#
#     # 计算e的指数次幂
#     x_exp = np.exp(x)
#     x_sum = torch.sum(x_exp)
#     softmax_pb = x_exp / x_sum
#     return softmax_pb



class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=0.5, gamma=2, logits = False, reduction = True):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.Tensor([alpha, 1-alpha]).to(DEVICE)  # alpha代表非漏钢样本所占权重
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))    #gather函数:a.gather(dim = x, y), 将y中数据索引的第x维替换为y值得到新的索引，用新的索引在a中取对应值
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        if self.reduction:
            return F_loss.mean()
        else:
            return F_loss.sum()




def bce_loss_w(input, target):
    # bce_loss = nn.BCELoss(size_average=True)
    weight = torch.zeros(len(target))
    weight = torch.fill_(weight, 0.1)
    weight[target == 1] = 1
    weight = weight.to(DEVICE)
    loss = nn.BCELoss(weight = weight)(input, target.float())
    return loss




# training cycle forward, backward, update
def train(device, images1, labels1, images2, labels2, images3, labels3, optimizer):
    model.train()   # 启用batch normalization和drop out
    training_loss = 0
    batch_size = batch_size_train
    num = 0
    y_true = np.array([])   # 用于存储真实图片标签值
    y_pred = np.array([])   # 用于存储模型预测标签值

    # 判断每次取一个batch的数据是否全部取完
    images1 = images1.to(device)
    labels1 = labels1.to(device)

    images2 = images2.to(device)
    labels2 = labels2.to(device)

    images3 = images3.to(device)
    labels3 = labels3.to(device)


    # numel()函数返回数组中元素的个数
    while (images1[num * batch_size : (num + 1) * batch_size]).numel() or \
        (images2[num * batch_size : (num + 1) * batch_size]).numel() or \
        (images3[num * batch_size : (num + 1) * batch_size]).numel():

        batch_images1 = images1[num * batch_size : (num + 1) * batch_size]
        batch_labels1 = labels1[num * batch_size : (num + 1) * batch_size]

        batch_images2 = images2[num * batch_size : (num + 1) * batch_size]
        batch_labels2 = labels2[num * batch_size : (num + 1) * batch_size]

        batch_images3 = images3[num * batch_size : (num + 1) * batch_size]
        batch_labels3 = labels3[num * batch_size : (num + 1) * batch_size]


        # To see if batch_images1, batch_images1 and batch_images1 have the same size.
        # (torch.exp(C1) + torch.exp(C2) + torch.exp(C3)) requires the above three tensors have the same size
        N1 = batch_images1.size()[0]
        N2 = batch_images2.size()[0]
        N3 = batch_images3.size()[0]

        N = max(N1, N2, N3)
        if N1 != N:
            N_supplement1 = N - N1
            batch_images1 = torch.cat((batch_images1, images1[: N_supplement1]), dim = 0)
            batch_labels1 = torch.cat((batch_labels1, labels1[: N_supplement1]), dim = 0)
        elif N2 != N:
            N_supplement2 = N - N2
            batch_images2 = torch.cat((batch_images2, images2[: N_supplement2]), dim = 0)
            batch_labels2 = torch.cat((batch_labels2, labels2[: N_supplement2]), dim = 0)
        elif N3 != N:
            N_supplement3 = N - N3
            batch_images3 = torch.cat((batch_images3, images3[: N_supplement3]), dim = 0)
            batch_labels3 = torch.cat((batch_labels3, labels3[: N_supplement3]), dim = 0)
        else:
            pass


        batch_labels = torch.cat((batch_labels1, batch_labels2, batch_labels3), dim = 0)


        optimizer.zero_grad()
        outputs = (model(batch_images1, batch_images2, batch_images3, batch_labels1, batch_labels2, batch_labels3)).view(-1).to(device)
        # loss = bce_loss_w(outputs, batch_labels)
        loss = criterion(outputs, batch_labels.float())
        training_loss += loss.item() * len(batch_labels)

        train_pred = torch.Tensor([1 if output > 0.5 else 0 for output in outputs]).to(device)
        y_true = np.concatenate((y_true, batch_labels.cpu().numpy()), axis = 0)
        y_pred = np.concatenate((y_pred, train_pred.cpu().numpy()), axis = 0)

        loss.backward()
        optimizer.step()
        num += 1


    cnf_matrix = confusion_matrix(y_true, y_pred)
    print("训练混淆矩阵为:", cnf_matrix, sep='\n')
    recall_neg = cnf_matrix[0][0] / (cnf_matrix[0][0] + cnf_matrix[0][1])
    recall_pos = cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1])
    training_G_mean = pow(recall_neg * recall_pos, 1 / 2)

    training_loss /= (len(images1) + len(images2) + len(images3))

    return training_G_mean, training_loss



def validate(device, images, labels, batch_size_validation):
    model.eval()    #评价时将BN层和Dropout层冻结，这两个操作不会对模型进行更改(验证时对单个样本进行处理，不需要BN和Dropout)
    validating_loss_fold = 0
    batch_size = batch_size_validation
    num = 0
    y_true = np.array([])   # 用于存储真实图片标签值
    y_pred = np.array([])   # 用于存储模型预测标签值


    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        # 判断每次取一个batch的数据是否全部取完
        while (images[num * batch_size : (num + 1) * batch_size]).numel():
            batch_images = images[num * batch_size : (num + 1) * batch_size]
            batch_labels = labels[num * batch_size : (num + 1) * batch_size]

            outputs = (model(batch_images)).view(-1).to(device)
            loss = criterion(outputs, batch_labels.float())
            validating_loss_fold += loss.item()

            validate_pred = torch.Tensor([1 if output > 0.5 else 0 for output in outputs]).to(device)
            y_true = np.concatenate((y_true, batch_labels.cpu().numpy()), axis = 0)
            y_pred = np.concatenate((y_pred, validate_pred.cpu().numpy()), axis = 0)

            num += 1

        cnf_matrix = confusion_matrix(y_true, y_pred)
        recall_neg = cnf_matrix[0][0] / (cnf_matrix[0][0] + cnf_matrix[0][1])
        recall_pos = cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1])
        validating_G_mean = pow(recall_neg * recall_pos, 1 / 2)

        validating_loss_fold /= len(images)

    return validating_G_mean, validating_loss_fold





def test(device, images1, labels1, images2, labels2, images3, labels3):
    model.eval()    #评价时将BN层和Dropout层冻结，这两个操作不会对模型进行更改(验证时对单个样本进行处理，不需要BN和Dropout)
    batch_size = batch_size_test
    y_true = np.array([])   # 用于存储真实图片标签值
    y_pred = np.array([])   # 用于存储模型预测标签值
    y_pred_proba = np.array([])
    testing_loss = 0
    num = 0

    images1 = images1.to(device)
    labels1 = labels1.to(device)

    images2 = images2.to(device)
    labels2 = labels2.to(device)

    images3 = images3.to(device)
    labels3 = labels3.to(device)

    with torch.no_grad():

        while (images1[num * batch_size: (num + 1) * batch_size]).numel() or \
                (images2[num * batch_size: (num + 1) * batch_size]).numel() or \
                (images3[num * batch_size: (num + 1) * batch_size]).numel():

            batch_images1 = images1[num * batch_size: (num + 1) * batch_size]
            batch_labels1 = labels1[num * batch_size: (num + 1) * batch_size]

            batch_images2 = images2[num * batch_size: (num + 1) * batch_size]
            batch_labels2 = labels2[num * batch_size: (num + 1) * batch_size]

            batch_images3 = images3[num * batch_size: (num + 1) * batch_size]
            batch_labels3 = labels3[num * batch_size: (num + 1) * batch_size]


            # To see if batch_images1, batch_images1 and batch_images1 have the same size.
            # (torch.exp(C1) + torch.exp(C2) + torch.exp(C3)) requires the above three tensors have the same size
            N1 = batch_images1.size()[0]
            N2 = batch_images2.size()[0]
            N3 = batch_images3.size()[0]

            N = max(N1, N2, N3)
            if N1 != N:
                N_supplement1 = N - N1
                batch_images1 = torch.cat((batch_images1, images1[: N_supplement1]), dim = 0)
                batch_labels1 = torch.cat((batch_labels1, labels1[: N_supplement1]), dim = 0)
            elif N2 != N:
                N_supplement2 = N - N2
                batch_images2 = torch.cat((batch_images2, images2[: N_supplement2]), dim = 0)
                batch_labels2 = torch.cat((batch_labels2, labels2[: N_supplement2]), dim = 0)
            elif N3 != N:
                N_supplement3 = N - N3
                batch_images3 = torch.cat((batch_images3, images3[: N_supplement3]), dim = 0)
                batch_labels3 = torch.cat((batch_labels3, labels3[: N_supplement3]), dim = 0)
            else:
                pass


            batch_labels = torch.cat((batch_labels1, batch_labels2, batch_labels3), dim = 0)

            outputs = (model(batch_images1, batch_images2, batch_images3, batch_labels1, batch_labels2, batch_labels3)).view(-1).to(device)
            # loss = bce_loss_w(outputs, batch_labels)
            loss = criterion(outputs, batch_labels.float())  # 计算出loss与预训练微调模型进行比较
            testing_loss += loss.item() * len(batch_labels)

            test_pred = torch.Tensor([1 if output > 0.5 else 0 for output in outputs]).to(device)
            y_true = np.concatenate((y_true, batch_labels.cpu().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, test_pred.cpu().numpy()), axis=0)
            y_pred_proba = np.concatenate((y_pred_proba, outputs.cpu().numpy()), axis=0)
            num += 1


        cnf_matrix = confusion_matrix(y_true, y_pred)
        print("测试混淆矩阵为:", cnf_matrix, sep='\n')
        recall_neg = cnf_matrix[0][0] / (cnf_matrix[0][0] + cnf_matrix[0][1])
        recall_pos = cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1])
        testing_G_mean = pow(recall_neg * recall_pos, 1 / 2)

        testing_loss /= (len(images1) + len(images2) + len(images3))

        if testing_G_mean > test_G_mean_benchmark:
            fpr, tpr, threshold = roc_curve(y_true, y_pred_proba)  # 依次将每个样本预测为正例绘制ROC曲线,threshold即为将样本依次划分为正例的阈值列表
            roc_auc = auc(fpr, tpr)
            with open("./MixDCNN/模型AUC值.txt", 'a') as f:
                f.write(str(cnf_matrix[0][0]))
                f.write("\t")
                f.write(str(cnf_matrix[0][1]))
                f.write("\t")
                f.write(str(cnf_matrix[1][0]))
                f.write("\t")
                f.write(str(cnf_matrix[1][1]))
                f.write("\t")
                f.write(str(format(roc_auc, ".4f")))
                f.write("\n")

    return testing_loss, testing_G_mean, cnf_matrix




# 搭建网络层
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size = 3)
        self.conv1_bn = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 3, padding = 1, stride = 2)
        self.conv2_bn = torch.nn.BatchNorm2d(20)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(320, 1)
        # self.fc2 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(0.8)

    def forward(self, x):

        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是980
        x = self.dropout(x)
        x = self.fc1(x)
        # x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(x)
        return x


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size = 3, padding = 2, stride = 4)
        self.conv1_bn = torch.nn.BatchNorm2d(10)
        # self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 3, padding = 1, stride = 2)
        # self.conv2_bn = torch.nn.BatchNorm2d(20)
        self.pooling = torch.nn.MaxPool2d(3, 2, 1)
        self.fc = torch.nn.Linear(250, 1)
        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, x):

        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1_bn(self.conv1(x))))
        # x = F.relu(self.pooling(self.conv2_bn(self.conv2(x))))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是980
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))

        return x



class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size = 11, padding = 2, stride = 4)
        self.conv1_bn = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5, padding = 0, stride = 2)
        self.conv2_bn = torch.nn.BatchNorm2d(20)
        self.pooling1 = torch.nn.MaxPool2d(3, 2)
        self.pooling2 = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(720, 1)
        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, x):

        batch_size = x.size(0)
        x = F.relu(self.pooling1(self.conv1_bn(self.conv1(x))))
        x = F.relu(self.pooling2(self.conv2_bn(self.conv2(x))))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))

        return x



class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size = 11, padding = 2, stride = 4)
        self.conv1_bn = torch.nn.BatchNorm2d(10)
        # self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5, padding = 0, stride = 2)
        # self.conv2_bn = torch.nn.BatchNorm2d(20)
        self.pooling1 = torch.nn.MaxPool2d(3, 2)
        # self.pooling2 = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(7290, 1)
        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, x):

        batch_size = x.size(0)
        x = F.relu(self.pooling1(self.conv1_bn(self.conv1(x))))
        # x = F.relu(self.pooling2(self.conv2_bn(self.conv2(x))))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))

        return x

# model = (Net()).to(DEVICE)



#######################################################################################################################
# VGG16 as pretrained model
features = 4096  # the feature dimension of last but one layer output of vgg16_bn

class MixDCNN(torch.nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)

        # vgg16_bn pretrained model as component
        vgg16_bn = models.vgg16_bn(pretrained=True)
        # remove the last classifier layer of vgg16_bn
        vgg16_bn.classifier = nn.Sequential(*list(vgg16_bn.classifier.children())[:-1])
        # Freeze model weights
        for param in vgg16_bn.parameters():
            param.requires_grad = False

        self.component = vgg16_bn

        # LDA降维(n_components cannot be larger than min(n_features, n_classes - 1).)
        # self.lda = LDA(n_components = lda_feature)

        # Linear classifier.
        self.fc = torch.nn.Linear(features, 1)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val = 0)

    def forward(self, x1, x2, x3, y1, y2, y3):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*224*224.
        Returns:
            Score, torch.autograd.Variable of shape 3N*1.
        """
        N1 = x1.size()[0]
        N2 = x2.size()[0]
        N3 = x3.size()[0]

        x1 = self.component(x1)     # x1: N1 * 4096
        x2 = self.component(x2)     # x2: N2 * 4096
        x3 = self.component(x3)     # x3: N3 * 4096

        # x1 = self.lda.fit_transform(x1, y1)     # x1: N1 * 128
        # x2 = self.lda.fit_transform(x2, y2)     # x2: N2 * 128
        # x3 = self.lda.fit_transform(x3, y3)     # x3: N3 * 128

        # Calculate Ck(best classification result) for each component
        C1 = torch.max(x1, dim = 1, keepdim = True)[0]  # [0]return the max value, [1] return the max value index
        C2 = torch.max(x2, dim = 1, keepdim = True)[0]
        C3 = torch.max(x3, dim = 1, keepdim = True)[0]

        C_sum = torch.exp(C1) + torch.exp(C2) + torch.exp(C3)
        a1 = torch.exp(C1) / C_sum
        a2 = torch.exp(C2) / C_sum
        a3 = torch.exp(C3) / C_sum

        zn = torch.cat((torch.mul(a1, x1), torch.mul(a2, x2), torch.mul(a3, x3)), dim = 0)    # zn: (N1 + N2 + N3) * 128

        z = torch.sigmoid(self.fc(zn))

        return z


model = MixDCNN().to(DEVICE)


# criterion = torch.nn.BCELoss()
criterion = WeightedFocalLoss(alpha = 0.3)

# optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr = 1e-5, weight_decay = 1e-9)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)


train_all_G_mean_epoch = []
train_all_loss_epoch = []
test_all_G_mean_epoch = []
test_all_loss_epoch = []


start_time = time.time()

for epoch in range(1, Epochs + 1):
    torch.cuda.empty_cache()  # 清理缓存
    print("当前正在运行第{}轮".format(epoch))

    for data in train_data_loader1:
        train_images1, train_labels1 = data

    for data in train_data_loader2:
        train_images2, train_labels2 = data

    for data in train_data_loader3:
        train_images3, train_labels3 = data

        # 在全部训练集上训练模型
        train_all_G_mean, train_all_loss = train(DEVICE, train_images1, train_labels1, train_images2, train_labels2, train_images3, train_labels3, optimizer)
        train_all_G_mean_epoch.append(train_all_G_mean)
        train_all_loss_epoch.append(train_all_loss)
        print("在全部训练集上得到的G-mean = {}, Loss = {}".format(train_all_G_mean, train_all_loss))




    # 测试模型泛化性能
    for data in test_data_loader1:
        test_images1, test_labels1 = data

    for data in test_data_loader2:
        test_images2, test_labels2 = data

    for data in test_data_loader3:
        test_images3, test_labels3 = data


        test_all_loss, test_all_G_mean, test_cnf_matrix = test(DEVICE, test_images1, test_labels1, test_images2, test_labels2, test_images3, test_labels3)
        test_all_G_mean_epoch.append(test_all_G_mean)
        test_all_loss_epoch.append(test_all_loss)
        print("在全部测试集上得到的G-mean = {}, Loss = {}".format(test_all_G_mean, test_all_loss))

        # # 保存模型状态字典 #保存路径 "H:/Fine-grained image classification/MixDCNN模型权值/"  #测试路径"F:/测试/MixDCNN模型权值/"
        # if test_all_G_mean > test_G_mean_benchmark:
        #     filepath = "H:/Fine-grained image classification/MixDCNN模型权值/"
        #     save_location = "Epoch = " + str(epoch) + "  " + "test_cnf_matrix = " + "[[" + str(test_cnf_matrix[0][0]) + "  " + str(test_cnf_matrix[0][1]) + "]" + \
        #     " " + "[" + str(test_cnf_matrix[1][0]) + "  " + str(test_cnf_matrix[1][1]) + "]]" + ".pth"
        #     save_location = filepath + save_location
        #
        #     torch.save({'state_dict': model.state_dict()}, save_location)


            # 加载模型
            # model = models.vgg16_bn(pretrained=True)
            # model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
            # model.classifier.add_module('7', nn.Sigmoid())
            #
            # model.load_state_dict(torch.load(save_location)['state_dict'])
            # model = model.to(DEVICE)



end_time = time.time()
print("程序训练用时 : %.2f s" % (end_time - start_time))


# 绘制模型在全部训练样本和测试样本上的性能曲线
plt.figure()
plt.plot(np.arange(1, Epochs + 1), train_all_loss_epoch, color='yellow')
plt.plot(np.arange(1, Epochs + 1), test_all_loss_epoch, color='deepskyblue')
plt.legend(['Train_All_Loss', 'Test_All_Loss'], loc='upper right')
plt.title('Fine-tuning fully connected layer loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(np.arange(1, Epochs + 1), train_all_G_mean_epoch, color='purple')
plt.plot(np.arange(1, Epochs + 1), test_all_G_mean_epoch, color='orange')
plt.legend(['Train_All_G_mean', 'Test_All_G_mean'], loc='lower right')
plt.title('Fine-tuning fully connected layer G-mean')
plt.xlabel('Epochs')
plt.ylabel('G-mean')
plt.show()

# #保存路径"H:/Fine-grained image classification/训练及测试过程数据.csv"   #测试路径 "F:/测试/训练及测试过程数据.csv"
# with open("H:/Fine-grained image classification/训练及测试过程数据.csv", 'a') as f:
#     f.write("train_loss")
#     f.write(",")
#     f.write("test_loss")
#     f.write(",")
#     f.write("train_G_mean")
#     f.write(",")
#     f.write("test_G_mean")
#     f.write("\n")
#     for i in range(len(train_all_loss_epoch)):
#         f.write(str(format(train_all_loss_epoch[i], '.4f')))
#         f.write(",")
#         f.write(str(format(test_all_loss_epoch[i], '.4f')))
#         f.write(",")
#         f.write(str(format(train_all_G_mean_epoch[i], '.4f')))
#         f.write(",")
#         f.write(str(format(test_all_G_mean_epoch[i], '.4f')))
#         f.write("\n")

#######################################################################################################################






#######################################################################################################################
# # design model using pretrained model
# model = models.vgg16_bn(pretrained=True)
#
# print("VGG16模型原始结构:", model, sep = '\n')
#
# #Freeze model weights
# for param in model.parameters():
#     param.requires_grad = False
#
#
# # for i in range(len(model.features)):
# #     for param in model.features[i].parameters():
# #         print(param.requires_grad)
# #
# # for i in range(len(model.classifier)):
# #     for param in model.classifier[i].parameters():
# #         print(param.requires_grad)
#
#
# model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
# model.classifier.add_module('7', nn.Sigmoid())
#
# model = model.to(DEVICE)
# print("VGG16模型微调结构:", model, sep = '\n')
#
#
# # criterion = torch.nn.BCELoss()
# criterion = WeightedFocalLoss(alpha = 0.02)
#
# optimizer1 = torch.optim.Adam(model.classifier[6].parameters(), lr = 1e-5, weight_decay = 1e-9)
# # optimizer2 = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay = 1e-7)
#
#
# train_all_G_mean_epoch = []
# train_all_loss_epoch = []
# test_all_G_mean_epoch = []
# test_all_loss_epoch = []
#
# start_time = time.time()
# print("第一阶段微调全连接层")
# for epoch in range(1, Epochs + 1):
#     torch.cuda.empty_cache()  # 清理缓存
#     print("当前正在运行第一阶段第{}轮".format(epoch))
#
#     for data in train_data_loader:
#         images, labels = data
#
#         # 在全部训练集上训练模型
#         train_all_G_mean, train_all_loss = train(DEVICE, images, labels, batch_size_train, optimizer1)
#         train_all_G_mean_epoch.append(train_all_G_mean)
#         train_all_loss_epoch.append(train_all_loss)
#         print("在全部训练集上得到的G-mean = {}, Loss = {}".format(train_all_G_mean, train_all_loss))
#
#
#     # 测试模型泛化性能
#     for data in test_data_loader:
#         test_all_loss, test_all_G_mean, test_cnf_matrix = test(DEVICE, data[0], data[1])
#         test_all_G_mean_epoch.append(test_all_G_mean)
#         test_all_loss_epoch.append(test_all_loss)
#         print("在全部测试集上得到的G-mean = {}, Loss = {}".format(test_all_G_mean, test_all_loss))
#
#         # 保存模型状态字典
#         if test_all_G_mean > test_G_mean_benchmark:
#             filepath = "./预训练+微调模型/VGG16预训练模型权值/"
#             save_location = "Epoch = " + str(epoch) + "  " + "test_cnf_matrix = " + "[[" + str(test_cnf_matrix[0][0]) + "  " + str(test_cnf_matrix[0][1]) + "]" + \
#             " " + "[" + str(test_cnf_matrix[1][0]) + "  " + str(test_cnf_matrix[1][1]) + "]]" + ".pth"
#             save_location = filepath + save_location
#
#             torch.save({'state_dict': model.state_dict()}, save_location)
#
#
#             # 加载模型
#             # model = models.vgg16_bn(pretrained=True)
#             # model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
#             # model.classifier.add_module('7', nn.Sigmoid())
#             #
#             # model.load_state_dict(torch.load(save_location)['state_dict'])
#             # model = model.to(DEVICE)
#
#
#
# end_time = time.time()
# print("程序训练用时 : %.2f s" % (end_time - start_time))
#
#
# # 绘制模型在全部训练样本和测试样本上的性能曲线
# plt.figure()
# plt.plot(np.arange(1, Epochs + 1), train_all_loss_epoch, color='yellow')
# plt.plot(np.arange(1, Epochs + 1), test_all_loss_epoch, color='deepskyblue')
# plt.legend(['Train_All_Loss', 'Test_All_Loss'], loc='upper right')
# plt.title('Fine-tuning fully connected layer loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
#
# plt.figure()
# plt.plot(np.arange(1, Epochs + 1), train_all_G_mean_epoch, color='purple')
# plt.plot(np.arange(1, Epochs + 1), test_all_G_mean_epoch, color='orange')
# plt.legend(['Train_All_G_mean', 'Test_All_G_mean'], loc='lower right')
# plt.title('Fine-tuning fully connected layer G-mean')
# plt.xlabel('Epochs')
# plt.ylabel('G-mean')
# plt.show()
#
#
# with open("./预训练+微调模型/训练及测试过程数据.csv", 'a') as f:
#     f.write("train_loss")
#     f.write(",")
#     f.write("test_loss")
#     f.write(",")
#     f.write("train_G_mean")
#     f.write(",")
#     f.write("test_G_mean")
#     f.write("\n")
#     for i in range(len(train_all_loss_epoch)):
#         f.write(str(format(train_all_loss_epoch[i], '.4f')))
#         f.write(",")
#         f.write(str(format(test_all_loss_epoch[i], '.4f')))
#         f.write(",")
#         f.write(str(format(train_all_G_mean_epoch[i], '.4f')))
#         f.write(",")
#         f.write(str(format(test_all_G_mean_epoch[i], '.4f')))
#         f.write("\n")
#
#
#
#
#
#
# # # 微调部分卷积层((40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
# # for param in model.features[40].parameters():
# #     param.requires_grad = True
# #
# # optimizer2 = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay = 1e-7)
# #
# # train_all_G_mean_epoch = []
# # train_all_loss_epoch = []
# # test_all_G_mean_epoch = []
# # test_all_loss_epoch = []
# #
# # start_time = time.time()
# # print("第二阶段微调部分卷积层和全连接层")
# # for epoch in range(1, Epochs + 1):
# #     torch.cuda.empty_cache()  # 清理缓存
# #     print("当前正在运行第二阶段第{}轮".format(epoch))
# #
# #     for data in train_data_loader:
# #         images, labels = data
# #
# #         # 在全部训练集上训练模型
# #         train_all_G_mean, train_all_loss = train(DEVICE, images, labels, batch_size_train, optimizer2)
# #         train_all_G_mean_epoch.append(train_all_G_mean)
# #         train_all_loss_epoch.append(train_all_loss)
# #         print("在全部训练集上得到的G-mean = {}, Loss = {}".format(train_all_G_mean, train_all_loss))
# #
# #
# #     # 测试模型泛化性能
# #     for data in test_data_loader:
# #         test_all_loss, test_all_G_mean = test(DEVICE, data[0], data[1])
# #         test_all_G_mean_epoch.append(test_all_G_mean)
# #         test_all_loss_epoch.append(test_all_loss)
# #         print("在全部测试集上得到的G-mean = {}, Loss = {}".format(test_all_G_mean, test_all_loss))
# #
# #
# # end_time = time.time()
# # print("程序训练用时 : %.2f s" % (end_time - start_time))
# #
# #
# # # 绘制模型在全部训练样本和测试样本上的性能曲线
# # plt.figure()
# # plt.plot(np.arange(1, Epochs + 1), train_all_loss_epoch, color='yellow')
# # plt.plot(np.arange(1, Epochs + 1), test_all_loss_epoch, color='deepskyblue')
# # plt.legend(['Train_All_Loss', 'Test_All_Loss'], loc='upper right')
# # plt.title('Fine-tuning fully connected and conv layer loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# #
# # plt.figure()
# # plt.plot(np.arange(1, Epochs + 1), train_all_G_mean_epoch, color='purple')
# # plt.plot(np.arange(1, Epochs + 1), test_all_G_mean_epoch, color='orange')
# # plt.legend(['Train_All_G_mean', 'Test_All_G_mean'], loc='lower right')
# # plt.title('Fine-tuning fully connected and conv layer G-mean')
# # plt.xlabel('Epochs')
# # plt.ylabel('G-mean')
# # plt.show()

#######################################################################################################################




# # print(model)
# # for i, j in list(enumerate(model.state_dict().items())):
# #     if i == len(model.state_dict()) - 1:
# #         model.state_dict()["abc"] = model.state_dict().pop(j[0])
# #         # print(model.state_dict()["abc"].shape)
# #
# #
# # for i, j in list(enumerate(model.state_dict())):
# #     print(i,j,sep = '  ', end = '\n')
#





# #将部分卷积层重新训练
# for i in range(len(model.features)):
#     if i >= 40:
#         for param in model.features[i].parameters():
#             param.requires_grad = True
#
# for j in range(len(model.classifier)):
#     for param in model.classifier[j].parameters():
#         param.requires_grad = True





# for param in model.classifier[6].parameters():
#     param.requires_grad = True
# for i in range(len(model.features)):
#         for param in model.features[i].parameters():
#             print(param.requires_grad)
#
# for j in range(len(model.classifier)):
#     for param in model.classifier[j].parameters():
#         print(param.requires_grad)
# for i, j in list(enumerate(model.state_dict())):
# #     print(i,j,sep = '  ', end = '\n')
# # #查看模型各层参数
# # for name, parameters in model.named_parameters():
# #     print(name, ':', parameters.size())




# #可视化网络层结构
# x = torch.randn(32, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
# y = model(x)    # 获取网络的预测值 ​
# #vis_graph = make_dot(y, params=dict(model.named_parameters()))
# vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# #vis_graph.format = "png"
# # 指定文件生成的文件夹
# vis_graph.directory = "data"
# # 生成文件
# #vis_graph.render('espnet_model', view=False)
# vis_graph.view()

# construct loss and optimizer
#定义新的损失函数FocalLoss
# class FocalLoss(nn.Module):
#     def __init__(self, alpha = 1, gamma = 2, logits = True, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')    #input不需要经过sigmoid函数激活
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')    #input需要经过sigmoid函数激活
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss






# 损失函数及优化器
#criterion = torch.nn.MSELoss()
# weight = torch.FloatTensor([0.2, 1])
# criterion = torch.nn.CrossEntropyLoss(weight = weight)
# criterion = WeightedFocalLoss(alpha = 0.05)
# criterion = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)




# if __name__ == '__main__':
#
#     # train_G_mean_epoch = []
#     # train_loss_epoch = []
#     # validate_G_mean_epoch = []
#     # validate_loss_epoch = []
#
#     train_all_G_mean_epoch = []
#     train_all_loss_epoch = []
#     test_all_G_mean_epoch = []
#     test_all_loss_epoch = []
#
#     start_time = time.time()
#     for epoch in range(1, Epochs + 1):
#         print("当前正在运行第{}轮".format(epoch))
#         # train_G_mean = 0
#         # train_loss = 0
#         # validate_G_mean = 0
#         # validate_loss = 0
#         #
#         # # 交叉验证划分训练集和验证集
#         for data in train_data_loader:
#             images, labels = data
#         #     k = 4
#         #     val_num = len(images) // k
#         #
#         #     for fold in range(k):
#         #
#         #         evaluate_images = images[val_num * fold : val_num * (fold + 1)]
#         #         evaluate_labels = labels[val_num * fold : val_num * (fold + 1)]
#         #
#         #         train_images = torch.cat((images[: val_num * fold], images[val_num * (fold + 1) :]), dim = 0)
#         #         train_labels = torch.cat((labels[: val_num * fold], labels[val_num * (fold + 1) :]), dim = 0)
#         #
#         #         train_G_mean_fold, train_loss_fold = train(DEVICE, train_images, train_labels, batch_size_train)
#         #         train_G_mean += train_G_mean_fold
#         #         train_loss += train_loss_fold
#         #
#         #         validate_G_mean_fold, validate_loss_fold = validate(DEVICE, evaluate_images, evaluate_labels, batch_size_validation)
#         #         validate_G_mean += validate_G_mean_fold
#         #         validate_loss += validate_loss_fold
#         #
#         #
#         #     train_G_mean_epoch.append(train_G_mean / k)
#         #     train_loss_epoch.append(train_loss / k)
#         #     validate_G_mean_epoch.append(validate_G_mean / k)
#         #     validate_loss_epoch.append(validate_loss / k)
#         #
#         #
#         # print("第{}轮的Train Loss = {}, Validation Loss = {}, Train G-mean = {}, Validation G-mean = {}"
#         # .format(epoch, train_loss_epoch[epoch - 1], validate_loss_epoch[epoch - 1], train_G_mean_epoch[epoch - 1], validate_G_mean_epoch[epoch - 1]))
#
#         torch.cuda.empty_cache()  # 清理缓存
#         # 在全部训练集上训练模型
#         train_all_G_mean, train_all_loss = train(DEVICE, images, labels, batch_size_train)
#         train_all_G_mean_epoch.append(train_all_G_mean)
#         train_all_loss_epoch.append(train_all_loss)
#         print("在全部训练集上得到的G-mean = {}, Loss = {}".format(train_all_G_mean, train_all_loss))
#
#
#         # 测试模型泛化性能
#         for data in test_data_loader:
#             test_all_loss, test_all_G_mean = test(DEVICE, data[0], data[1])
#             test_all_G_mean_epoch.append(test_all_G_mean)
#             test_all_loss_epoch.append(test_all_loss)
#             print("在全部测试集上得到的G-mean = {}, Loss = {}".format(test_all_G_mean, test_all_loss))
#
#
#
#
#
#
#     end_time = time.time()
#     print("程序训练用时 : %.2f s" % (end_time - start_time))
#
#
#     # # 在全部训练集上训练模型
#     # train_all_G_mean, train_all_loss = train(DEVICE, images, labels, batch_size_train)
#     # print("在全部训练集上得到的G-mean = {}, Loss = {}".format(train_all_G_mean, train_all_loss))
#     #
#     #
#     # # 测试模型泛化性能
#     # for data in test_data_loader:
#     #     test_all_loss, test_all_G_mean = test(DEVICE, data[0], data[1])
#     #     print("Test loss:", test_all_loss)
#     #     print("Test G-mean:", test_all_G_mean)
#
#
#     # # 绘制模型训练和验证曲线
#     # plt.figure()
#     # plt.plot(np.arange(1, Epochs + 1), train_loss_epoch, color = 'blue')
#     # plt.plot(np.arange(1, Epochs + 1), validate_loss_epoch, color = 'red')
#     # plt.legend(['Train_Loss', 'Validation_Loss'], loc = 'upper right')
#     # plt.xlabel('Epochs')
#     # plt.ylabel('Loss')
#     #
#     #
#     # plt.figure()
#     # plt.plot(np.arange(1, Epochs + 1), train_G_mean_epoch, color = 'green')
#     # plt.plot(np.arange(1, Epochs + 1), validate_G_mean_epoch, color = 'cyan')
#     # plt.legend(['Train_G_mean', 'Validation_G_mean'], loc = 'lower right')
#     # plt.xlabel('Epochs')
#     # plt.ylabel('G-mean')
#     # # plt.show()
#
#     # 绘制模型在全部训练样本和测试样本上的性能曲线
#     plt.figure()
#     plt.plot(np.arange(1, Epochs + 1), train_all_loss_epoch, color = 'yellow')
#     plt.plot(np.arange(1, Epochs + 1), test_all_loss_epoch, color = 'deepskyblue')
#     plt.legend(['Train_All_Loss', 'Test_All_Loss'], loc = 'upper right')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#
#
#     plt.figure()
#     plt.plot(np.arange(1, Epochs + 1), train_all_G_mean_epoch, color = 'purple')
#     plt.plot(np.arange(1, Epochs + 1), test_all_G_mean_epoch, color = 'orange')
#     plt.legend(['Train_All_G_mean', 'Test_All_G_mean'], loc = 'lower right')
#     plt.xlabel('Epochs')
#     plt.ylabel('G-mean')
#     plt.show()
#
#


# def test_real_time(epoch, device):
#     with torch.no_grad():
#         for batch_idx, data in enumerate(LC_test_loader_rt, 1):
#             images, labels = data
#             images = torch.cat((images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]), axis=2).to(device)
#             labels = labels.to(device)
#
#             for i in range(sequence_length):
#                 rt_input = images[:, 0:(i+1) ,:]
#                 rt_output = model(rt_input)
#                 rt_pb_classes = softmax(rt_output)
#                 class_pred_pb = rt_pb_classes[0][0]
#                 #class_pred = rt_pb_classes.max(1)[1]
#                 print('有{:.2f}的概率预测为纵裂'.format(class_pred_pb.item()))
#
#             print("实际情况为: ", LC_prediction[labels.item()])


#开始训练
# if __name__ == '__main__':
#     start_time = time.time()
#     for epoch in range(1, Epochs + 1):      #每轮中128个训练样本，64个测试样本
#         # train(epoch, DEVICE)
#         test(epoch, DEVICE)
#         #test_real_time(epoch, DEVICE)
#         torch.cuda.empty_cache()  # 清理缓存
#         #保存模型网络层权重
#         if epoch % 10 == 0:
#             save_path = "G:\\课题\\python\\demo\\pytorch教程\\南钢漏钢识别结果\\第四次实验结果\\模型网络层权重\\Epoch = " + str(epoch) + ".pth"
#             torch.save(model.state_dict(), save_path)  # 一个状态字典就是一个简单的Python的字典，其键值对是每个网络层和其对应的参数张量。
#
#     end_time = time.time()
#     print("程序用时 : %.2f s" %(end_time - start_time))







    #保存和加载模型
    #保存模型
    # save_path = "G:\\课题\\python\demo\\pytorch教程\\漏钢识别结果\\漏钢图像识别.pth"
    # torch.save(model.state_dict(), save_path)   #一个状态字典就是一个简单的Python的字典，其键值对是每个网络层和其对应的参数张量。
    #加载模型
    # model = models.vgg16_bn(pretrained=True)
    # model.classifier[6] = nn.Sequential(
    #     nn.Linear(4096, 2))
    # model.load_state_dict(torch.load(save_path))
    # model.eval()
    # model.to(DEVICE)
    # #用训练好的模型进行预测
    # test_pic(0, DEVICE)

# #绘制训练及测试损失曲线图
# plt.figure()
# plt.plot(np.arange(1, Epochs + 1), train_losses, color='blue')
# plt.plot(np.arange(1, Epochs + 1), test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
#
#
# #绘制训练及测试分类准确率曲线图
# plt.figure()
# plt.plot(np.arange(1, Epochs + 1), train_accuracy, color='blue')
# plt.plot(np.arange(1, Epochs + 1), test_accuracy, color='red')
# plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
#
# #绘制训练和测试漏报和误报曲线图
# plt.figure()
# plt.plot(np.arange(1, Epochs + 1), train_underreports, color='blue')
# plt.plot(np.arange(1, Epochs + 1), train_false_alarms, color='red')
# plt.plot(np.arange(1, Epochs + 1), test_underreports, color='green')
# plt.plot(np.arange(1, Epochs + 1), test_false_alarms, color='cyan')
# plt.legend(['Train_Underreports', 'Train_False_Alarms', 'Test_Underreports', 'Test_False_Alarms'], loc='upper right')
# plt.xlabel('Epochs')
# plt.ylabel('Errors')
# plt.show()
#
#
# #将结果导出至文件
# with open("G:\\课题\\python\\demo\\pytorch教程\\南钢漏钢识别结果\\第四次实验结果\\漏钢识别结果.txt", "a") as f:
#     f.write("train_losses")
#     f.write("\t")
#     f.write("test_losses")
#     f.write("\t")
#     f.write("train_accuracy")
#     f.write("\t")
#     f.write("test_accuracy")
#     f.write("\t")
#     f.write("train_underreports")
#     f.write("\t")
#     f.write("train_false_alarms")
#     f.write("\t")
#     f.write("test_underreports")
#     f.write("\t")
#     f.write("test_false_alarms")
#     f.write("\n")
#     for i in range(Epochs):
#         f.write(str(train_losses[i]))
#         f.write("\t")
#         f.write(str(test_losses[i]))
#         f.write("\t")
#         f.write(str(train_accuracy[i]))
#         f.write("\t")
#         f.write(str(test_accuracy[i]))
#         f.write("\t")
#         f.write(str(train_underreports[i]))
#         f.write("\t")
#         f.write(str(train_false_alarms[i]))
#         f.write("\t")
#         f.write(str(test_underreports[i]))
#         f.write("\t")
#         f.write(str(test_false_alarms[i]))
#         f.write("\n")
#
# f.close()
