import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
# from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# 设置随机种子
np.random.seed(42)


# 从flist特征列表中随机选取n个,返回所有可能的无重复的组合结果
def feature_selection(flist, n, combin = [], result = []):
    for i in range(len(flist)):
        combin.append(flist[i])
        if len(combin) == n:
            result.append(combin.copy())
            combin.pop()
        else:
            result = feature_selection(flist[i + 1 :], n, combin, result)
            combin.pop()

    return result


# result = feature_selection([1,2,3,4,5,6], 3)


# 加载数据(注意先划分训练、测试集再进行数据归一化，测试样本用的是训练集得出的归一化参数)
def load_data(feature_list):
    data_name = ".\\黏结区域特征向量(全部样本添加新特征).csv"
    # 读取csv文件中的特定列并指定数据类型
    data = pd.read_csv(data_name, usecols = feature_list, dtype = np.float64, encoding = 'gb2312').values

    data_labels = pd.read_csv(data_name, usecols = ['类型'], dtype = str, encoding = 'gb2312').values

    data_index = pd.read_csv(data_name, usecols=['编号'], dtype=int, encoding='gb2312').values

    return data, data_labels, data_index


# 绘制ROC曲线
def plot_AUC(y_test, y_pred_proba):

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)  # 依次将每个样本预测为正例绘制ROC曲线,threshold即为将样本依次划分为正例的阈值列表
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


G_mean_max = 0              # 最优G-mean分数
best_feature_combin = []    # 最佳特征组合
feature_result = []         # 所有不重复的特征组合结果
features = ['Vc', 'Height', 'Width', 'Area', 'Gave', 'Fourier', 'Sticking_Expansion']
feature_list = ['Vx', 'Vy', 'Vc', 'Height', 'Width', 'Area','Gave','Fourier','Sticking_Expansion']
with open("最佳特征选择-LR(去掉Edgenum).csv",'a') as f:
    # f.write("特征个数")
    # f.write(",")
    # f.write("G_mean_max")
    # f.write("\n")
    for num_remove in range(1):
        # f.write(str(len(feature_list) - num_remove))
        # f.write(",")
        print("当前考虑的是去除{}个特征的情况。".format(num_remove))
        feature_result.clear()
        feature_result = feature_selection(features, len(features) - num_remove)
        G_mean_fea_dict = {}
        for feature_combin in feature_result:
            print("当前处理的是去除{}个特征后，所有{}个特征组合中的第{}种组合".format(num_remove, len(feature_result), feature_result.index(feature_combin) + 1))

            samples, samples_attributes, indices = load_data(feature_combin)

            # 数据标签
            y_samples = np.array([1 if i == '漏钢' else 0 for i in samples_attributes])

            # 分层采样划分训练集和测试集
            # n_splits为划分次数, test_size为测试集比例, random_state设置随机状态(每次划分结果相同)
            split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.20, random_state = 42)

            for train_index, test_index in split.split(samples, y_samples):
                x_train, x_test = samples[train_index], samples[test_index]
                y_train, y_test = np.array([y_samples[i] for i in train_index]), np.array([y_samples[j] for j in test_index])
                index_train, index_test = np.array([indices[i] for i in train_index]), np.array([indices[j] for j in test_index])


                x_train_min = x_train.min(axis = 0)
                x_train_max = x_train.max(axis = 0)
                # 数据归一化
                x_train = (x_train - x_train_min) / (x_train_max - x_train_min)


                k = 4
                val_num = len(x_train) // k  # 向下取整

                C_list = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
                tol_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
                validation_list = []
                for ratio in np.arange(1, 21):
                    for C in C_list:
                        for tol in tol_list:
                            evaluate_score = 0
                            for fold in range(k):
                                # 训练集和验证集划分
                                evaluate_samples = x_train[val_num * fold : val_num * (fold + 1)]
                                evaluate_samples_labels = y_train[val_num * fold : val_num * (fold + 1)]

                                train_samples = np.concatenate((x_train[:val_num * fold], x_train[val_num * (fold + 1):]), axis = 0)
                                train_samples_labels = np.concatenate((y_train[:val_num * fold], y_train[val_num * (fold + 1):]), axis = 0)

                                model1 = LogisticRegression(C = C, tol = tol, max_iter = 1000)
                                sample_weight = np.array([ratio if train_samples_labels[i] == 1 else 1 for i in range(len(train_samples_labels))])

                                model1.fit(train_samples, train_samples_labels, sample_weight = sample_weight)
                                evaluate_result = model1.predict(evaluate_samples)
                                # 将召回率作为评价指标调整超参数('macro'代表在计算均值时使每个类别具有相同的权重，最后结果是每个类别的指标的算术平均值
                                # 'micro'代表计算多分类指标时赋予所有类别的每个样本相同的权重，将所有样本合在一起计算各个指标。
                                # 如果每个类别的样本数量差不多，那么宏平均和微平均没有太大差异.
                                # 如果每个类别的样本数量差异很大，那么注重样本量多的类时使用微平均，注重样本量少的类时使用宏平均)
                                # evaluate_score += metrics.recall_score(evaluate_samples_labels, evaluate_result, average='macro')
                                cnf_matrix = confusion_matrix(evaluate_samples_labels, evaluate_result)
                                recall_neg = cnf_matrix[0][0] / (cnf_matrix[0][0] + cnf_matrix[0][1])
                                recall_pos = cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1])
                                G_mean = pow(recall_neg * recall_pos, 1/2)
                                evaluate_score += G_mean

                                dict = {}
                                dict['ratio'] = ratio
                                dict['C'] = C
                                dict['tol'] = tol
                                dict['evaluate_score'] = evaluate_score / k
                                validation_list.append(dict)

                # 选取验证分数最高的超参数
                best_parm = []
                max_evaluation_score = 0
                for dict in validation_list:
                    if dict['evaluate_score'] > max_evaluation_score:
                        max_evaluation_score = dict['evaluate_score']

                print("max_evaluation_score = ", max_evaluation_score)

                for dict in validation_list:
                    if dict['evaluate_score'] == max_evaluation_score:
                        best_parm.append(dict)


                for paras in best_parm:

                    model1 = LogisticRegression(C = paras['C'], tol = paras['tol'], max_iter = 1000)
                    sample_weight = np.array([paras['ratio'] if y_train[i] == 1 else 1 for i in range(len(y_train))])
                    model1.fit(x_train,y_train, sample_weight = sample_weight)
                    # 对测试集数据归一化
                    x_test = (x_test - x_train_min) / (x_train_max - x_train_min)
                    # X_test_reduced = pca.transform(x_test)
                    y_test_proba = model1.predict_proba(x_test)[:, 1]
                    plot_AUC(y_test, y_test_proba)

                    # 阈值调整
                    # y_test_pred = y_test_proba >= 0.48
                    # cnf_matrix = confusion_matrix(y_test, y_test_pred)
                    # print("混淆矩阵为:",cnf_matrix)


                    y_test_pred = model1.predict(x_test)
                    # 预测错误的样本索引
                    false_pred_index = [index_test[k] for k in np.where((y_test == y_test_pred) == 0)]
                    print("预测错误的样本数量为:", len(false_pred_index[0]))
                    print("预测错误的样本编号为:", false_pred_index[0])
                    print("The best ratio is ", paras['ratio'])
                    print("The best C is", paras['C'])
                    print("The best tol is ", paras['tol'])
                    # print("The best number of estimator is ", paras['n_estimators'])

                    disp = plot_confusion_matrix(model1, x_test, y_test,
                                                     display_labels=[0,1],
                                                     cmap=plt.cm.Blues,
                                                     values_format=''
                                                     )

                    disp.ax_.set_title('Confusion matrix')
                    plt.show()

                    test_cnf_matrix = confusion_matrix(y_test, y_test_pred)
                    P_breakout = test_cnf_matrix[1][1] / (test_cnf_matrix[0][1] + test_cnf_matrix[1][1])
                    P_nonbreakout = test_cnf_matrix[0][0] / (test_cnf_matrix[0][0] + test_cnf_matrix[1][0])
                    R_breakout = test_cnf_matrix[1][1] / (test_cnf_matrix[1][0] + test_cnf_matrix[1][1])
                    R_nonbreakout = test_cnf_matrix[0][0] / (test_cnf_matrix[0][0] + test_cnf_matrix[0][1])
                    belta = 1.0
                    F1_breakout = (1 + belta * belta) * P_breakout * R_breakout / (belta * belta * P_breakout + R_breakout)
                    F1_nonbreakout = (1 + belta * belta) * P_nonbreakout * R_nonbreakout / (belta * belta * P_nonbreakout + R_nonbreakout)
                    G_mean = pow(R_breakout * R_nonbreakout, 0.5)
                    print("漏钢样本的查准率为{:.3f},非漏钢样本的查准率为{:.3f}".format(P_breakout, P_nonbreakout))
                    print("漏钢样本的查全率为{:.3f},非漏钢样本的查全率为{:.3f}".format(R_breakout, R_nonbreakout))
                    print("漏钢样本的F1分数为{:.3f},非漏钢样本的F1分数为{:.3f}".format(F1_breakout, F1_nonbreakout))
                    print("模型的G-mean为:",G_mean)
                    break
    #######################################################################################################################

            G_mean_fea_dict[G_mean] = feature_combin

        G_mean_max_temp = max(G_mean_fea_dict.keys())
        # f.write(str(format(G_mean_max_temp, '.6f')))
        # f.write("\n")
        if G_mean_max_temp >= G_mean_max:
            G_mean_max = G_mean_max_temp
            best_feature_combin = G_mean_fea_dict[G_mean_max]
        else:
            break


# 最佳G-mean分数为: 0.9072647087265547
# 最佳特征组合为: ['Vc', 'Height', 'Width', 'Area', 'Gave', 'Fourier', 'Sticking_Expansion']

print("最佳G-mean分数为:", G_mean_max)
print("最佳特征组合为:", best_feature_combin)





# 对训练集进行PCA降维(注意先归一化再降维，PCA降维使各维特征的方差和最大化，因此归一化后保证各维特征处在相同量级)
# pca = PCA(n_components=2)
# x_train_reduced = pca.fit_transform(x_train)

######################################################################################################
# # EasyEnsemble
# # K折交叉验证
# k = 4
# val_num = len(x_train) // k        #向下取整
#
# # num_estimators = [5, 10, 20, 30, 50, 100]
# num_estimators = np.arange(2, 52, 2)
#
# validation_list = []
# for n in num_estimators:
#     evaluate_score = 0
#     for fold in range(k):
#         # 训练集和验证集划分
#         evaluate_samples = x_train[val_num * fold : val_num * (fold + 1)]
#         evaluate_samples_labels = y_train[val_num * fold : val_num * (fold + 1)]
#
#         train_samples = np.concatenate((x_train[:val_num * fold], x_train[val_num * (fold + 1):]), axis = 0)
#         train_samples_labels = np.concatenate((y_train[:val_num * fold], y_train[val_num * (fold + 1):]), axis = 0)
#
#         model3 = EasyEnsembleClassifier(n_estimators=n, base_estimator = AdaBoostClassifier())
#         model3.fit(train_samples, train_samples_labels)
#
#         evaluate_result = model3.predict(evaluate_samples)
#
#         evaluate_score += metrics.recall_score(evaluate_samples_labels, evaluate_result, average='macro')
#
#
#     dict = {}
#     dict['num_estimators'] = n
#     dict['evaluate_score'] = evaluate_score / k
#     validation_list.append(dict)
#
#
# # 选取验证分数最高的超参数
# best_parm = []
# max_evaluation_score = 0
# for dict in validation_list:
#     if dict['evaluate_score'] > max_evaluation_score:
#         max_evaluation_score = dict['evaluate_score']
#
# print("max_evaluation_score = ", max_evaluation_score)
#
# for dict in validation_list:
#     if dict['evaluate_score'] == max_evaluation_score:
#         best_parm.append(dict)
#
#
# for paras in best_parm:
#
#     # 模型3(EasyEnsemble)
#     model3 = EasyEnsembleClassifier(n_estimators=paras['num_estimators'], base_estimator=AdaBoostClassifier())
#     model3.fit(x_train,y_train)
#     x_test = (x_test - x_train_min) / (x_train_max - x_train_min)
#     probs = model3.predict_proba(x_test)[:, 1]  # probs为将每个测试样本预测为正例的概率列表
#     plot_AUC(y_test, probs)
#     y_pred = model3.predict(x_test)
#     print("best estimator numbers = ", paras['num_estimators'])
#
#     disp = plot_confusion_matrix(model3, x_test, y_test,
#                                      display_labels=[0,1],
#                                      values_format='',
#                                      cmap=plt.cm.Blues
#                                      )
#     disp.ax_.set_title('confusion matrix')
#     # print(metrics.confusion_matrix(y_test, y_pred))
#     plt.show()
##############################################################################################################


##############################################################################################################
# #   BalanceCascade
# def BalanceCascade(X_train, y_train, X_test, num):
#     negnum = y_train[y_train == 0].shape[0]  # 将标签为0的样本取出，计算个数
#     posnum = y_train[y_train == 1].shape[0]
#     neg_index = np.argwhere(y_train == 0).reshape(negnum, )  # np.argwhere(a) 查找满足条件a的索引
#     pos_index = np.argwhere(y_train == 1).reshape(posnum, )
#     pos_train = X_train[pos_index, :]
#
#     FP = pow(posnum / negnum, 1 / (num - 1))    # 计算False Positive Rate
#     classifiers = {}
#     thresholds = {}
#     test_prob = np.empty((X_test.shape[0], num))
#     thre_ave = 0
#     # cur_adboost_alpha = np.zeros(num)
#     for i in range(num):
#         classifiers[i] = AdaBoostClassifier()
#         # 对负样本索引序列随机打乱顺序(np.random.permutation)并取出与正样本数量相同的样本进行训练
#         neg_train_index = np.random.permutation(neg_index)[:posnum]
#         neg_train = X_train[neg_train_index, :]
#         # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等;np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
#         cur_X_train = np.r_[pos_train, neg_train]
#         cur_y_train = np.r_[y_train[pos_index], y_train[neg_train_index]]
#         # sample_weight = np.array([ratio if cur_y_train[i] == 1 else 1 for i in range(len(cur_y_train))])
#         classifiers[i].fit(cur_X_train, cur_y_train)
#
#         # errors = classifiers[i].estimator_errors_
#         # for j in range(len(errors)):
#         #     cur_adboost_alpha[i] += float(0.5 * np.log((1.0 - errors[j]) / max(errors[j], 1e-16)))
#         #
#         # cur_adboost_alpha[i] /= 2
#
#         predict_result = classifiers[i].predict_proba(X_train[neg_index, :])[:, -1]    # 用训练好的分类器对所有负例样本进行预测(预测其为正例的概率)
#         # 计算在类别不平衡条件下的阈值与标准分类阈值(0.5)的偏移量,在分类阈值左侧的样本(预测为正例的概率更小，预测为负例的概率更大)为分类正确的样本需要剔除
#         thresholds[i] = np.sort(predict_result)[int(neg_index.shape[0] * (1 - FP))] - 0.5
#         neg_index = np.argwhere(predict_result >= (thresholds[i] + 0.5)).reshape(-1, )  # 保留预测概率大于阈值的负例样本(即预测错误的样本)
#         test_prob[:, i] = classifiers[i].predict_proba(X_test)[:, -1] + thresholds[i]   # 对测试集样本的预测结果需要加上阈值偏移量(因为类别不平衡)
#         thre_ave += thresholds[i] + 0.5
#         print("No.{} Classifier Training Finished".format(i))
#     test_prob_result = np.average(test_prob, axis=1)    # 对所有子分类器的预测结果取均值
#     # 考虑阈值设置的问题！！！
#     thre_ave /= num
#     test_pred_result = test_prob_result >= thre_ave     # 返回的矩阵中True代表1(正例), False代表0(反例)
#
#     return test_prob_result, test_pred_result
#
#
# # K折交叉验证
# k = 4
# val_num = len(x_train) // k        #向下取整
#
# # num_estimators = [5, 10, 20, 30, 50, 100]
# num_estimators = np.arange(2, 52, 2)
#
# validation_list = []
# for n in num_estimators:
#         evaluate_score = 0
#         for fold in range(k):
#             # 训练集和验证集划分
#             evaluate_samples = x_train[val_num * fold : val_num * (fold + 1)]
#             evaluate_samples_labels = y_train[val_num * fold : val_num * (fold + 1)]
#
#             train_samples = np.concatenate((x_train[:val_num * fold], x_train[val_num * (fold + 1):]), axis = 0)
#             train_samples_labels = np.concatenate((y_train[:val_num * fold], y_train[val_num * (fold + 1):]), axis = 0)
#
#             _, evaluate_result = BalanceCascade(train_samples, train_samples_labels, evaluate_samples, n)
#
#
#             evaluate_score += metrics.recall_score(evaluate_samples_labels, evaluate_result, average='macro')
#
#         dict = {}
#         dict['num_estimators'] = n
#         # dict['ratio'] = ratio
#         dict['evaluate_score'] = evaluate_score / k
#         validation_list.append(dict)
#
#
# # 选取验证分数最高的超参数
# best_parm = []
# max_evaluation_score = 0
# for dict in validation_list:
#     if dict['evaluate_score'] > max_evaluation_score:
#         max_evaluation_score = dict['evaluate_score']
#
# print("max_evaluation_score = ", max_evaluation_score)
#
# for dict in validation_list:
#     if dict['evaluate_score'] == max_evaluation_score:
#         best_parm.append(dict)
#
#
# for paras in best_parm:
#
#     # 模型3(BalanceCascade)
#     x_test = (x_test - x_train_min) / (x_train_max - x_train_min)
#     y_test_proba, y_test_pred  = BalanceCascade(x_train, y_train, x_test, paras['num_estimators'])
#
#     plot_AUC(y_test, y_test_proba)
#     print("best estimator numbers = ", paras['num_estimators'])
#     # print("best ratio is ", paras['ratio'])
#     print(metrics.confusion_matrix(y_test, y_test_pred))
#     plt.show()
###########################################################################################################


###########################################################################################################
# # 模型2(SMOTE + Adaboost分类器)
# from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
#
# for paras in best_parm:
#
#     sm = ADASYN(random_state=10)
#     X_train_balanced, y_train_balanced = sm.fit_resample(x_train, y_train)
#     print('breakout-samples rate are: ', y_train_balanced.mean())
#
#     model2 =AdaBoostClassifier()
#     # w_pos = 0.5 / np.sum(y_train_balanced == 1)
#     # w_neg = 0.5 / np.sum(y_train_balanced == 0)
#     sample_weight = np.array([paras['ratio'] if y_train_balanced[i] == 1 else 1 for i in range(len(y_train_balanced))])
#     model2.fit(X_train_balanced,y_train_balanced)
#     # 对测试集数据归一化
#     x_test = (x_test - x_train_min) / (x_train_max - x_train_min)
#     y_test_proba = model2.predict_proba(x_test)[:, 1]
#     plot_AUC(y_test, y_test_proba)
#
#     y_test_pred = model2.predict(x_test)
#     # 预测错误的样本索引
#     false_pred_index = [index_test[k] for k in np.where((y_test == y_test_pred) == 0)]
#     print("预测错误的样本数量为:", len(false_pred_index[0]))
#     print("预测错误的样本编号为:", false_pred_index)
#
#     disp = plot_confusion_matrix(model2, x_test, y_test,
#                                      display_labels=[0,1],
#                                      cmap=plt.cm.Blues,
#                                      values_format=''
#                                      )
#
#     disp.ax_.set_title('confusion matrix')
#     plt.show()
#     break
###############################################################################################################


















# # K折交叉验证
# k = 5
# val_num = len(x_train) // k        #向下取整
#
#
# # 超参数集合
# Cs = [0.001, 0.01, 0.1, 1]
# tols = [1e-4, 1e-3, 1e-2, 1e-1]
# validation_list = []
# for C in Cs:
#     for tol in tols:
#         evaluate_score = 0
#         for fold in range(k):
#             # 训练集和验证集划分
#             evaluate_samples = x_train[val_num * fold : val_num * (fold + 1)]
#             evaluate_samples_labels = y_train[val_num * fold : val_num * (fold + 1)]
#
#             train_samples = np.concatenate((x_train[:val_num * fold], x_train[val_num * (fold + 1):]), axis = 0)
#             train_samples_labels = y_train[:val_num * fold] + y_train[val_num * (fold + 1):]
#
#             clf = LogisticRegression(penalty='l2', dual=False, tol = tol, C = C,
#                          fit_intercept=True, intercept_scaling=1, class_weight=None,
#                          random_state=None, solver='liblinear', max_iter=100,
#                          multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
#
#             clf.fit(train_samples, train_samples_labels)
#             evaluate_score += clf.score(evaluate_samples, evaluate_samples_labels)
#
#         dict = {}
#         dict['tol'] = tol
#         dict['C'] = C
#         dict['evaluate_score'] = evaluate_score / k
#         validation_list.append(dict)
#
# # 选取验证分数最高的超参数组合
# best_parm = []
# max_evaluation_score = 0
# for dict in validation_list:
#     if dict['evaluate_score'] > max_evaluation_score:
#         max_evaluation_score = dict['evaluate_score']
#
# print("max_evaluation_score = ", max_evaluation_score)
#
# for dict in validation_list:
#     if dict['evaluate_score'] == max_evaluation_score:
#         best_parm.append(dict)
#
# # 开始在总的训练样本上进行训练(训练集 + 验证集),在测试集上评估模型的泛化性能
# for params in best_parm:
#
#     clf = LogisticRegression(penalty='l2', dual=False, tol = params['tol'], C = params['C'],
#                              fit_intercept=True, intercept_scaling=1, class_weight=None,
#                              random_state=None, solver='liblinear', max_iter=100,
#                              multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
#
#     clf.fit(x_train, y_train)
#     print("训练精度为：",clf.score(x_train, y_train))
#
#     # 对测试集进行预测
#     clf_y_predict = clf.predict(x_test)
#
#     missing_reports = len([i for i in range(len(clf_y_predict)) if clf_y_predict[i] != y_test[i] and y_test[i] == 1])
#     false_alarms = len([i for i in range(len(clf_y_predict)) if clf_y_predict[i] != y_test[i] and y_test[i] == 0])
#
#
#     print("超参数组合：tol = %.5f, C = %.5f" %(params['tol'], params['C']))
#     print("漏报次数 = %d, 误报次数 = %d" %(missing_reports, false_alarms))
