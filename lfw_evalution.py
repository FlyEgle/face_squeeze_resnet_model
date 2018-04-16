# !/usr/local/bin/python3.6
# -*— coding: utf-8 -*-
"""
lfw verfication
lfw pairs.txt
10 300代表重复10次，300表示的是300个匹配图片 1， 4表示这个文件夹下的图片ID
10	300
bel_Pacheco	1	4
...
300行以后开始不匹配
Abdel_Madi_Shabneh	1	Dean_Barker	1
一共重复10次构成完整的pairs.txt， 一共3000 matched, 3000个 mismatched

feature: 把image输入到model中，提取feature，对其进行pca降维生成vector
cosine distance:
    cosine_distance = (A.B) / (norm(A)*norm(B))

TP: 表示实际类别和预测类别都是正实例， 真阳性
TN: 将负类预测为负类数，真阳性
FP: 表示实际类别为负实例，预测类别为正实例， 假阳性
FN: 表示实际类别和预测类别都是负实例

准确率
accuracy = (TP+TN)/(TP+FP+TN+FN)
精确率
precision = TP/(TP+FP)
召回率
recall = TP/(TP+FN)
"""
import os
import numpy as np
import pandas as pd
import numpy.linalg as la
import time
import matplotlib.pyplot as plt

from PIL import Image
from RESNET50 import resnet_model
from keras.models import Model, Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.losses import binary_crossentropy


# 图片名称
def lfw_pair_image(name, image_id):
    if int(image_id) < 10:
        return str(name)+'_000'+str(image_id)
    elif int(image_id) < 100:
        return str(name)+'_00'+str(image_id)
    elif int(image_id) < 1000:
        return str(name)+'_0'+str(image_id)
    else:
        return str(name)+str(image_id)


# 图片id
def lfw_image_id(name):
    return name.split('\n')[0].split('\t')


# 生成同一个人图片索引和不同人图片索引
def person_image_index(path):
    """
    :param path: pairs.txt path
    :return: same_person list, not same_person list
    """
    lfw_image = '/Users/jmc/Desktop/facepaper/lfw_2_align/'
    same_person_1 = []
    same_person_2 = []
    not_same_person_1 = []
    not_same_person_2 = []
    with open(path, 'r') as f:
        change_pairs = [x for x in range(20)]
        paris_txt = []
        # 构建pair信息list
        for line in f:
            paris_txt.append(line)
        # 从paris_txt中读取同一个人和不同人的图片id
        for x in change_pairs:
            # 同一个人
            if x % 2 == 0:
                for index in range(300*x, 300*(x+1)):
                    # 处理同一个人id
                    name_id = lfw_image_id(paris_txt[index])
                    same_person_1.append(lfw_image+name_id[0]+'/'+lfw_pair_image(name_id[0], name_id[1])+'.jpg')
                    same_person_2.append(lfw_image+name_id[0]+'/'+lfw_pair_image(name_id[0], name_id[2])+'.jpg')
            else:
                for index in range(300*x, 300*(x+1)):
                    # 处理非同一人id
                    name_id = lfw_image_id(paris_txt[index])
                    not_same_person_1.append(lfw_image+name_id[0]+'/'+lfw_pair_image(name_id[0], name_id[1])+'.jpg')
                    not_same_person_2.append(lfw_image+name_id[2]+'/'+lfw_pair_image(name_id[2], name_id[3])+'.jpg')

    same_person = np.vstack((same_person_1, same_person_2)).T
    not_same_person = np.vstack((not_same_person_1, not_same_person_2)).T
    return same_person, not_same_person


# 生成data和label, 其中同一个人的label=1, 不同人的lable=0
def generator_data(same_person, not_same_person):
    """
    :param same_person:  list
    :param not_same_person: list
    :return: array(data, lable) shape=(6000, 2)
    """
    same_person_label = np.array([1 for _ in range(3000)]).reshape((-1, 1))  # same label
    not_same_person_label = np.array([0 for _ in range(3000)]).reshape((-1, 1))  # not same label
    same_arr = np.hstack((same_person, same_person_label))   # (3000, 2)
    not_same_arr = np.hstack((not_same_person, not_same_person_label))  # (3000, 3)
    person_data = np.vstack((same_arr, not_same_arr))  # (6000, 3)
    return person_data


# 是否文件存在
def isfile_(file_list):
    for path1, path2 in file_list:
        if not os.path.isfile(path1):
            print(path1, 'this jpeg is not in dir')
        if not os.path.isfile(path2):
            print(path2, 'this jpeg is not in dir')


# 输出csv
def arr2csv(file_list, sameperson=True):
    """
    输出图片路径到csv文件中，方便以后读取
    :param file_list:
    :param sameperson:
    :return:
    """
    if sameperson:
        file1 = file_list[:, 0]
        file2 = file_list[:, 1]
        df = pd.DataFrame({'person_a': file1, 'person_b': file2})
        df.to_csv('person_same_align.csv', sep=',', index=False)
    else:
        file1 = file_list[:, 0]
        file2 = file_list[:, 1]
        df = pd.DataFrame({'person_a': file1, 'person_b': file2})
        df.to_csv('person_notsame_align.csv', sep=',', index=False)


# 读取图片, 并对其进行resize
def read_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    # image[:,:,0] -= 129.1863
    # image[:,:,1] -= 104.7624
    # image[:,:,2] -= 93.5940
    # image = image.transpose((2, 0, 1))
    image = (image - 121.58) / 67.6
    image = np.expand_dims(image, axis=0)
    return image


# 获取image在model中的features, 用pca进行降维
def get_feats(image, model):

    feats = model.predict(image)
    """
    需要写一个提取feature的方法, 晚点更新
    """
    # pca 进行降维度
    # feats_2 = pca.PCA(feats, 2)
    return feats


# 计算两个features之间的cosine距离
def evalute_cosine(left_features, right_feautres):
    """
    features是一个vector, 需要提前从model中读取特征并降低维度
    :param left_features: vector
    :param right_feautres:  vector
    :return:
    """
    left_features = np.squeeze(left_features)
    right_feautres = np.squeeze(right_feautres)
    cosine_distance = np.dot(left_features, right_feautres)/(la.norm(left_features)*la.norm(right_feautres))
    return cosine_distance


# 计算两个featursvecotr之间的欧式距离
def caleuclideandistance(left_features, right_features):
    left_features = np.squeeze(left_features)
    right_features = np.squeeze(right_features)
    euclidean_distance = np.sqrt(np.sum(np.square(left_features - right_features)))
    return euclidean_distance


# 计算相似距离，保存list
def calcute_distance(image_list, weight=None, mode='cosine'):
    """
    :param image_list: list (same_person, not_same_person)
    :param weight: model weight's path
    :param mode: 'cosine': cosine distance, 'euclidean': euclidean distance
    :return: distance list
    """
    model = basemodel_extraction(weight, 'feats_dense')  # get resnet features 1024 vector
    distance_result = []
    image_count = 1
    for image1_list, image2_list in image_list:
        image1 = read_image(image1_list)
        image2 = read_image(image2_list)

        feats1 = get_feats(image1, model)
        feats2 = get_feats(image2, model)

        print('calcute image {}'.format(image_count))
        if mode == 'cosine':
            distance_result.append(evalute_cosine(feats1, feats2))
        else:
            distance_result.append(caleuclideandistance(feats1, feats2))

        image_count += 1
    return distance_result


# 计算paris中的相似度
def face_verfication(file, mode='cosine', threshold_dis=0.9, same_person=True):
    """
    从file中读取图片，输入到网络中，计算cosine并与threshold_dis比较
    :param file: image path list
    :param mode: 'cosine' or 'euclidean'
    :param threshold_dis: threshold for cosine distance
    :param same_person: same person or not same
    :return: acc of face verfication
    """
    weight_path = '/Users/jmc/Desktop/facepaper/weights.resnet50_2_softmax_best.hdf5'
    # weight_path = '/Users/jmc/Desktop/facepaper/vggface/vgg-face-keras-fc.h5'
    person_df = pd.read_csv(file)
    # 提取特征，softmax分类的前一层
    model = basemodel_extraction(weight_path, 'flatten_1')
    # 两个特征距离计算
    distance_result = []
    tp = 0  # 真真
    tn = 0  # 真假
    image_count = 1
    for image1_list, image2_list in zip(person_df['person_a'], person_df['person_b']):
        image1 = read_image(image1_list)
        image2 = read_image(image2_list)
        # 这里晚点更新，提取特征的方法和降低维度, 获取特征vector
        feats1 = get_feats(image1, model)
        feats2 = get_feats(image2, model)

        # 计算特征之间的cosine distance
        if mode == 'cosine':
            distance = evalute_cosine(feats1, feats2)
        else:
            distance = caleuclideandistance(feats1, feats2)

        print('number is {} this is a image1 {}, image2 {} and cosine distance is {}'.format(image_count,
                                                                                             image1_list.split('/')[-1],
                                                                                             image2_list.split('/')[-1],
                                                                                             distance))
        # 距离计算度量
        distance_result.append(distance)
        '''
        TP, TN, FP, FN
        '''
        # 如果是同一个人的两张图片
        if same_person:
            # 如果cosine大于阈值，说明两张图片是同一个人
            if distance >= threshold_dis:
                tp += 1
            else:
                tn += 1
        # 如果是两个人的照片
        else:
            # 如果cosine小于阈值, 说明两张图片是两个人 ，FN 如果不是则是 FP
            if distance < threshold_dis:
                tp += 1  # FN
            else:
                tn += 1  # FP

        image_count += 1
    return tp, tn


def basemodel_extraction(weight_path, name):
    """
    this fuc is used for get features from model
    :param weight_path: *.h5 weight_path
    :return: model
    """
    base_model = resnet_model()
    base_model.load_weights(weight_path, by_name=True)
    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer(name).output)
    return model


# 计算model在lfw上的accuracy
def total_acc(same_file, not_same_file, threshold_nums):
    """
    :param same_file: sameperson_file
    :param not_same_file: notsameperson_file
    :param threshold_nums: threshold
        choice num or list, if it is a list, return is change
    :return: accuracy of threshold
    """
    start = time.time()
    tp, tn = face_verfication(same_file, threshold_dis=threshold_nums, same_person=True)
    fp, fn = face_verfication(not_same_file, threshold_dis=threshold_nums, same_person=False)
    end = time.time()
    # accuracy
    print('time waste is {}'.format((end-start) / 6000))
    return (tp + tn) / (tp + tn + fp + fn)


# 特征向量分类器构建
def classifar(classfication):
    """
    :param classfication:  分类器
    :return: 分类器model
    """
    if classfication == 'dnn':

        # dnn 有疑问，查一下再更新
        model = Sequential()
        model.add(Dense(1, input_dim=1, activation='sigmoid'))
        return model

    elif classfication == 'svm':
        model = SVC()
        return model

    elif classfication == 'linearsvm':
        model = LinearSVC()
        return model

    elif classfication == 'xgboost':
        model = XGBClassifier()
        return model

    elif classfication == 'randomforest':
        model = RandomForestClassifier()
        return model


# 十折交叉验证训练分类器
def train_classification(data, label, model, mode='dnn'):

    history_record = []
    # 十折交叉验证
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        if mode == 'dnn':
            model.compile(optimizer='sgd', loss=binary_crossentropy, metrics=['acc'])
            history = model.fit((x_train, y_train), batch_size=256, epochs=len(x_train)//256,
                                validation_data=(x_test, y_test))
            history_record.append(history)

        else:
            model.fit(x_train, y_train)
            result = model.predict(x_test)
            count = 0
            for i in range(len(result)):
                if result[i] == y_test[i]:
                    count += 1
            acc = count / len(y_test)
            history_record.append(acc)
            mean_acc = np.mean(history_record)
    return mean_acc


# False Positive Rate, True Positive Rate
def fpr_tpr(tp, tn, fp, fn):
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return fpr, tpr


# 绘制ROC图像
def show_roc(fpr, tpr):
    """
    :param fpr: a list of fpr
    :param tpr: a list of tpr
    :return:
    """
    plt.figure()
    plt.plot(fpr, tpr, 'b-')
    plt.savefig('roc.jpg')


if __name__ == '__main__':
    path = '/Users/jmc/Desktop/facepaper/lwf_pair.txt'
    # print(total_acc('person_same_align.csv', 'person_notsame_align.csv', threshold_nums=0.5))
    # model = resnet(weights='/Users/jmc/Desktop/facepaper/squeeze_softmax.h5')

    # 生成csv文件
    # same, not_same = person_image_index(path)
    # isfile_(same)
    # isfile_(not_same)
    # arr2csv(same, sameperson=True)
    # arr2csv(not_same, sameperson=False)
    same_person, not_same_person = person_image_index(path)
    # same_distance = calcute_distance(same_person,
    #                                  weight='/Users/jmc/Desktop/facepaper/finetuneresnet50_softmax_aug.h5')
    # not_same_distance = calcute_distance(not_same_person,
    #                                      weight='/Users/jmc/Desktop/facepaper/finetuneresnet50_softmax_aug.h5')
    #
    # data = generator_data(same_distance, not_same_distance)  # (6000, 2) (data, label)
    # np.save('data.npy', data)

    # np.save('/Users/jmc/Desktop/facepaper/same_distance.npy', same_distance)
    # np.save('/Users/jmc/Desktop/facepaper/not_same_distance.npy', not_same_distance

