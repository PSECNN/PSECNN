from NLPProcess import NLPProcess
from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)
import keras
import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from PSECNN.deep.optimizer import DecagonOptimizer
from PSECNN.deep.model import DecagonModel
from PSECNN.deep.minibatch import EdgeMinibatchIterator
from PSECNN.utility import rank_metrics, preprocessing

event_num = 50
droprate = 0.4
vector_size = 620
opt = keras.optimizers.adam(lr=0.01)



################
#from skmultilearn.adapt import MLkNN

################

def DNN():
    train_input = Input(shape=(vector_size * 2,), name='Inputlayer')
    train_in = Dense(1024, activation='relu')(train_input)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(512, activation='relu')(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(256, activation='relu')(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(event_num)(train_in)
    out = Activation('sigmoid')(train_in)
    model = Model(input=train_input, output=out)
    model.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy'])

    return model

def prepare(df_drug, feature_list, vector_size,mechanism,drugA,drugB):
    d_label = {}
    d_feature = {}
    # Transfrom the interaction event to number
    # Splice the features
    d_event=[]
    for i in range(len(mechanism)):
        d_event.append(mechanism[i])
    label_value = 0
    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)
    for i in feature_list:
        vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]
    # Use the dictionary to obtain feature vector and label
    label_array=[]
    for line in open("labelC50.txt",'r'):
        label_array.append(line.split()[0])
    print(len(label_array))
    a=[]
    h=0
    new_feature = []
    new_label = []
    name_to_id = {}
    for i in range(len(d_event)):
        label=[0]*event_num
        i=h
        if h<len(drugA):
            fAB = drugA[h]+drugB[h]
            if fAB in a:
                for j in range(h,len(d_event)):
                    if drugA[j]+drugB[j]==fAB:
                        h=j+1
                        for k in range(len(label_array)):
                            if label_array[k]== d_event[j]:
                                label[k]=1
                new_label.append(label)
                new_label.append(label)
            else:
                a.append(fAB)
                new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]])))
                new_feature.append(np.hstack((d_feature[drugB[i]], d_feature[drugA[i]])))
    new_feature = np.array(new_feature)
    new_label = np.array(new_label)
    return (new_feature, new_label, event_num)


def feature_vector(feature_name, df, vector_size):
    # df are the all kinds of drugs
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|â€¦â€¦"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1
    sim_matrix = Jaccard(np.array(df_feature))
    sim_matrix1 = np.array(sim_matrix)
    pca = PCA(n_components=vector_size)  # PCA dimension
    pca.fit(sim_matrix)
    sim_matrix = pca.transform(sim_matrix)
    return sim_matrix


def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    index=[]
    for i in range(len(label_matrix)):
        index.append(np.where(label_matrix == i))
        
    kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
    k_num = 0
    for train_index, test_index in kf.split(range(len(index))):
        index_all_class[test_index] = k_num
        k_num += 1

    return index_all_class


def cross_validation(feature_matrix, label_matrix, clf_type, event_num, seed, CV, set_name):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_true = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label_matrix, event_num, seed, CV)
    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        # =============================================================================
        #     elif len(np.shape(feature_matrix))==3:
        #         for i in range((np.shape(feature_matrix)[-1])):
        #             matrix.append(feature_matrix[:,:,i])
        # =============================================================================
        feature_matrix = matrix
    b=0
    for k in range(CV):
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)
        y_p = np.zeros((len(test_index[0]), event_num), dtype=float)
        # dnn=DNN()
        for i in range(len(feature_matrix)):
            x_train = feature_matrix[i][train_index]
            x_test = feature_matrix[i][test_index]
            y_train = label_matrix[train_index]
            # one-hot encoding
            y_train_one_hot = np.array(y_train)
            #y_train_one_hot = (np.arange(y_train_one_hot.max()) == y_train[:, None]).astype(dtype='float32')
            y_test = label_matrix[test_index]
            # one-hot encoding
            y_test_one_hot = np.array(y_test)
            #y_test_one_hot = (np.arange(y_test_one_hot.max()) == y_test[:, None]).astype(dtype='float32')
            if clf_type == 'DDIMDL':
                dnn = DNN()
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                dnn.fit(x_train, y_train_one_hot, batch_size=64, epochs=15, validation_data=(x_test, y_test_one_hot))
                pred += dnn.predict(x_test)
                print("average_precision_score-micro")
                print(average_precision_score(y_test_one_hot, pred, average='micro'))
                continue
            elif clf_type == 'RF':
                clf = RandomForestClassifier(n_estimators=100)
            elif clf_type == 'GBDT':
                clf = GradientBoostingClassifier()
            elif clf_type == 'SVM':
                clf = SVC(probability=True)
            elif clf_type == 'FM':
                clf = GradientBoostingClassifier()
            elif clf_type == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=4)
            else:
                clf = LogisticRegression()
            clf.fit(x_train, y_train)
            pred += clf.predict_proba(x_test)
        a=0
        for i in range(len(y_test)):
            a+= apk2(y_test[i], pred[i], k=50)
        b+= a/len(y_test)
        pred_score = pred / len(feature_matrix)
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.vstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))

    ################################################################################
    """for i in range(len(y_score)):
        for j in range(len(y_score[i])):
            if y_score[i][j]< 0.26:
                y_score[i][j]=0
            else:
                y_score[i][j]=1"""
    print("*********")
    print("APK@50")
    print(b/CV)
    print("*********")
    ################################################################################
    result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num, set_name)
    # =============================================================================
    #         a,b=evaluate(pred_type,pred_score,y_test,event_num)
    #         for i in range(all_eval_type):
    #             result_all[i]+=a[i]
    #         for i in range(each_eval_type):
    #             result_eve[:,i]+=b[:,i]
    #     result_all=result_all/5
    #     result_eve=result_eve/5
    # =============================================================================
    return result_all, result_eve


def evaluate(pred_type, pred_score, y_test, event_num, set_name):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    #y_one_hot = label_binarize(y_test, np.arange(event_num))
    y_one_hot = y_test
    #pred_one_hot = label_binarize(pred_type, np.arange(event_num))
    pred_one_hot = pred_type

    print("y_test")
    print(y_test.shape)
    print("pred")
    print(pred_score.shape)

    precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)
    print(precision)
    print(recall)
    print(th)
    print("*****")
    # Plot the output. 
    plt.plot(th, precision[:-1], c ='r', label ='PRECISION') 
    plt.plot(th, recall[:-1], c ='b', label ='RECALL') 
    plt.grid() 
    plt.legend() 
    plt.title('Precision-Recall Curve')
    plt.savefig('foo.png')
    plt.savefig('foo.pdf')
    try:
        print("average_precision_score-micro")
        print(average_precision_score(y_one_hot, pred_score, average='micro'))
        result_all[1] = average_precision_score(y_one_hot, pred_score, average='micro')
    except ValueError:
        pass
    try:
        print("average_precision_score-samples")
        print(average_precision_score(y_one_hot, pred_score, average='samples'))
        result_all[2] = average_precision_score(y_one_hot, pred_score, average='samples')
    except ValueError:
        pass
    try:
        print("average_precision_score")
        print(average_precision_score(y_one_hot, pred_score))
        result_all[6] = average_precision_score(y_one_hot, pred_score)
    except ValueError:
        pass
    try:
        print("roc_auc_score-micro")
        print(roc_auc_score(y_one_hot, pred_score, average='micro'))
        result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    except ValueError:
        pass
    try:
        print("roc_auc_score-samples")
        print(roc_auc_score(y_one_hot, pred_score, average='samples'))
        result_all[4] = roc_auc_score(y_one_hot, pred_score, average='samples')
    except ValueError:
        pass
    try:
        print("precision_recall_curve")
        print(precision_recall_curve(y_one_hot, pred_score, pos_label=1))
        result_all[5] = precision_recall_curve(y_one_hot, pred_score, average='samples')
    except ValueError:
        pass

    result_all[0] = 7.7777
    #result_all[0] = multilabel_confusion_matrix(y_test, pred_type)
    #print(roc_aupr_score(y_one_hot, pred_score, average='samples'))
    """result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='samples')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='samples')
    result_all[5] = f1_score(y_test, pred_type, average='samples')
    result_all[5] = AP@K(y_test, pred_type, 50)
    #result_all[6] = f1_score(y_test, pred_type, average='samples')
    result_all[7] = precision_score(y_test, pred_type, average='samples')
    result_all[8] = precision_score(y_test, pred_type, average='samples')
    result_all[9] = recall_score(y_test, pred_type, average='samples')
    result_all[10] = recall_score(y_test, pred_type, average='samples')"""
    for i in range(event_num):
        result_eve[i, 0] = 77.7777
        #result_eve[i, 0] = multilabel_confusion_matrix(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        #result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=0).ravel(),average=None)
        #result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=0).ravel(),average=None)
        #result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average='binary')
        #result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average='binary')
        #result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average='binary')
    return [result_all, result_eve]


def self_metric_calculate(y_true, pred_type):
    y_true = y_true.ravel()
    y_pred = pred_type.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_pred_c = y_pred.take([0], axis=1).ravel()
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_true_c)):
        if (y_true_c[i] == 1) and (y_pred_c[i] == 1):
            TP += 1
        if (y_true_c[i] == 1) and (y_pred_c[i] == 0):
            FN += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 1):
            FP += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 0):
            TN += 1
    print("TP=", TP, "FN=", FN, "FP=", FP, "TN=", TN)
    return (TP / (TP + FP), TP / (TP + FN))


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision, reorder=True)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def drawing(d_result, contrast_list, info_list):
    column = []
    for i in contrast_list:
        column.append(i)
    df = pd.DataFrame(columns=column)
    if info_list[-1] == 'aupr':
        for i in contrast_list:
            df[i] = d_result[i][:, 1]
    else:
        for i in contrast_list:
            df[i] = d_result[i][:, 2]
    df = df.astype('float')
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    df.plot.box(ylim=[0, 1.0], grid=True, color=color)
    return 0


def save_result(feature_name, result_type, clf_type, result):
    with open(feature_name + '_' + result_type + '_' + clf_type+ '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0
######################

def apk1(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for j in range(len(predicted)):
        for i in range(len(predicted[j])):
            p=predicted[j][i]
            if actual[j][i]!=p :
                num_hits += 1.0
                score += num_hits / ((j)*100 + i + 1.0)

    return score / (min(len(actual), k))
def apk2(actual, predicted, k=10):
    
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i in range(len(predicted)):
        """print("actual[i]")
        print(actual[i])
        print("predicted[i]")
        print(predicted[i])"""
        if actual[i]!=predicted[i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    return score / min(len(actual), k)
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)   

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a, p in zip(actual, predicted)])

def sigmoid(x):
    return 1. / (1 + np.exp(-x))
######################

def main(args):
    seed = 0
    CV = 25
    interaction_num = 642
    conn = sqlite3.connect("start5.db")
    df_drug = pd.read_sql('select * from drug;', conn)

    feature_list = args['featureList']
    featureName="+".join(feature_list)
    clf_list = args['classifier']
    for feature in feature_list:
        set_name = feature + '+'
    set_name = set_name[:-1]
    result_all = {}
    result_eve = {}
    all_matrix = []
    all_label = []
    drugList=[]
    for line in open("druglistC.txt",'r'):
        drugList.append(line.split()[0])
    if args['NLPProcess']=="read":
        extraction = pd.read_sql('select * from extraction;', conn)
        mechanism = extraction['mechanism']
        drugA = extraction['drugA']
        drugB = extraction['drugB']
    else:
        mechanism,action,drugA,drugB=NLPProcess(drugList,df_interaction)
    
    new_label=[]
    new_feature=[]
    for feature in feature_list:
        print(feature)
        new_feature, new_label, event_num = prepare(df_drug, [feature], vector_size, mechanism,drugA,drugB)
        all_label.append(new_label)
        all_matrix.append(new_feature)
    start = time.clock()

    for clf in clf_list:
        print(clf)
        all_result, each_result = cross_validation(all_matrix, new_label, clf, event_num, seed, CV,
                                                   set_name)
        # =============================================================================
        #     save_result('all_nosim','all',clf,all_result)
        #     save_result('all_nosim','eve',clf,each_result)
        # =============================================================================
        save_result(featureName, 'all', clf, all_result)
        save_result(featureName, 'each', clf, each_result)
        result_all[clf] = all_result
        result_eve[clf] = each_result
    print("time used:", time.clock() - start)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f","--featureList",default=["smile","target","enzyme"],help="features to use",nargs="+")
    parser.add_argument("-c","--classifier",choices=["DDIMDL","RF","KNN","LR"],default=["DDIMDL"],help="classifiers to use",nargs="+")
    parser.add_argument("-p","--NLPProcess",choices=["read","process"],default="read",help="Read the NLP extraction result directly or process the events again")
    args=vars(parser.parse_args())
    print(args)
    main(args)
