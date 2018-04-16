import numpy as np
from lfw_evalution import generator_data, classifar, train_classification
import matplotlib.pyplot as plt

if __name__ == '__main__':
    same = list(np.load('/Users/jmc/Desktop/facepaper/face_data/weights/same_distance_dis_new_0_0_1_s_centerloss.npy').reshape((-1, 1)))  # (3000, 1)
    notsame = list(np.load('/Users/jmc/Desktop/facepaper/face_data/weights/not_same_distance_dis_new_0_0_1_s_centerloss.npy').reshape((-1, 1)))  # (3000, 1)

    data = generator_data(same, notsame)  # (6000, 2)

    tra = np.array(data[:, 0]).reshape((-1, 1))
    label = np.array([int(x) for x in data[:, 1]]).reshape((-1, 1))

    # bilud model
    class_model = classifar('randomforest')
    acc = train_classification(tra, label, class_model, mode=None)
    print(acc)
    """
        same person
    """
    def ROC(threshold):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for x in same:
            if x >= threshold:
                tp += 1  # 真正类
            else:
                fn += 1

        for x in notsame:
            if x < threshold:
                tn += 1  # 真负类
            else:
                fp += 1
        return tp, tn, fp, fn

    threshold = list(np.arange(0.1, 1, 0.01))
    TP, TN, FP, FN, FPR, TPR = [], [], [], [], [], []
    for i in range(len(threshold)):
        tp, tn, fp, fn = ROC(threshold[i])
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)

        fpr = fp / (fp + tn)
        tpr = tp / (fn + tp)
        TPR.append(tpr)
        FPR.append(fpr)
    acc = []
    for i in range(len(threshold)):
        acc.append((TP[i] + TN[i]) / (TP[i] + FN[i] + FP[i] + TN[i]))
    print('max is acc :', max(acc))

    plt.style.use('ggplot')
    plt.figure(figsize=(16, 9))
    plt.plot(FPR, TPR, 'b-')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciver Operating Characteristics')
    plt.legend(['resnet50_image_softmax(0.9123)'])
    plt.savefig('resnet50_image_softmax_edist.png')
    plt.show()




