# ---------------------------------------------------------------------------- #
# By Park Dongmin in DM-lab, KAIST                                             #
# ---------------------------------------------------------------------------- #
from __future__ import division

from math import log
import numpy as np
import pandas as pd
import copy
from operator import itemgetter
import time

def timedelta2float(td):
    res = td.microsecond/float(1000000) + (td.second + td.day * 24 * 3600)
    return res

def topk_accuracy(Y_pred, Y_true, groups, k_list, baseline = False):
    group_uni = set(groups)
    #print(len(group_uni))
    accuracy = np.zeros(len(k_list))
    avg_rank = 0
    mrr = 0
    ndcg = 0

    avg_precisions, avg_recalls, avg_ndcgs = None, None, None
    for i, g in enumerate(group_uni):
        group_idx = np.where(groups == g)[0]
        #print(len(group_idx))

        Y = Y_true[group_idx]
        true_idx = np.where(Y==1)[0]

        sorted_pred_idx = np.argsort(-Y_pred[group_idx])

        pred_rank = []
        for t in true_idx:
            p = np.where(sorted_pred_idx == t)[0][0]
            pred_rank.append(p)
        #print(pred_rank)

        if len(pred_rank) == 0:
            print("No pred index same as true")
            print(g)

        elif len(pred_rank)>0:
            # baseline normalization
            if baseline == True:
                normalized_Y_pred = Y_pred[group_idx]
                #print(np.max(normalized_Y_pred), np.min(normalized_Y_pred))
                normalized_Y_pred = (normalized_Y_pred - np.min(normalized_Y_pred)) / (
                            np.max(normalized_Y_pred) - np.min(normalized_Y_pred) + 0.000000000001)
                sorted_pred_idx = np.argsort(-normalized_Y_pred)

            avg_rank += np.min(pred_rank)+1
            mrr += 1/(np.mean(pred_rank)+1)
            #dcg = Y_pred[sorted_pred_idx[0]] / log(1 + 1, 2)

            #print(true_idx)
            #print(sorted_pred_idx)

            precisions, recalls, ndcgs = user_precision_recall_ndcg_k_list(sorted_pred_idx, true_idx, k_list)

            if i==0:
                avg_precisions, avg_recalls, avg_ndcgs = precisions, recalls, ndcgs
            else:
                avg_precisions += precisions
                avg_recalls += recalls
                avg_ndcgs += ndcgs

    avg_f1_score = avg_precisions
    for i in range(len(k_list)):
        print("here")
        if not avg_precisions[i] + avg_recalls[i] == 0:
            avg_f1_score[i] = 2 * (avg_precisions[i] * avg_recalls[i]) / (avg_precisions[i] + avg_recalls[i])
        else:
            avg_f1_score[i] = 0

    print(avg_f1_score/len(group_uni))
    #print(len(group_uni))
    return avg_rank/len(group_uni) ,avg_precisions/len(group_uni), avg_recalls/len(group_uni), \
           avg_f1_score/len(group_uni), avg_ndcgs/len(group_uni)

def topk_type_ablation(Y_pred, Y_true, groups, k_list, Y_type, target_type):
    group_uni = set(groups)
    #print(len(group_uni))
    accuracy = np.zeros(len(k_list))
    target_type_cnt = 0
    avg_poi_cnt = 0
    for g in group_uni:
        group_idx = np.where(groups == g)[0]
        #print(len(group_idx))

        Y = Y_true[group_idx]
        type = Y_type[group_idx]
        true_idx = np.where(Y==1)[0]
        true_type = np.where(Y==1)[0]

        if len(true_idx)>0:
            if true_type[0] == target_type:
                sorted_pred_idx = np.argsort(-Y_pred[group_idx])
                pred_rank = np.where(sorted_pred_idx == true_idx)[0]
                #print(pred_rank)
                for i, k in enumerate(k_list):
                    if pred_rank < k:
                        accuracy[i] += 1
                target_type_cnt+=1
                avg_poi_cnt+=len(group_idx)
    #print("target_type_cnt: ", target_type_cnt)
    #print("avg_poi_cnt: ", avg_poi_cnt/target_type_cnt)
    return accuracy/target_type_cnt

##Entropy
def entropy(Y):
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

##Entropy
def Entropy(Y):
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)

# calculate NDCG@k
def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg

def user_precision_recall_ndcg_k_list(new_user_prediction_arr, test, k_list):
    dcg_list = []

    new_user_prediction = new_user_prediction_arr.tolist()
    if len(new_user_prediction) < k_list[-1]:
        new_user_prediction += [0 for i in range(k_list[-1] - len(new_user_prediction))]
    # compute the number of true positive items at top k
    count_1, count_2, count_3 = 0, 0, 0
    for i in range(len(k_list)):
        if i < k_list[0] and new_user_prediction[i] in test:
            count_1 = 1.0
        if i < k_list[1] and new_user_prediction[i] in test:
            count_2 += 1.0
        if new_user_prediction[i] in test:
            count_3 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, k_list[0] )
    ndcg_tmp_2 = NDCG_at_k(dcg_list, idcg_list, k_list[1] )
    ndcg_tmp_3 = NDCG_at_k(dcg_list, idcg_list, k_list[2] )

    # precision@k
    precision_1 = count_1
    precision_2 = count_2 / k_list[1]
    precision_3 = count_3 / k_list[2]

    l = len(test)
    if l == 0:
        l = 1
    # recall@k
    recall_1 = count_1 / l
    recall_2 = count_2 / l
    recall_3 = count_3 / l

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_2, precision_3]),\
           np.array([recall_1, recall_2, recall_3]),\
           np.array([ndcg_tmp_1, ndcg_tmp_2, ndcg_tmp_3])

# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_precision_recall_ndcg_135(new_user_prediction_arr, test):
    dcg_list = []

    new_user_prediction = new_user_prediction_arr.tolist()
    if len(new_user_prediction) < 5:
        new_user_prediction += [0 for i in range(5 - len(new_user_prediction))]
    # compute the number of true positive items at top k
    count_1, count_3, count_5 = 0, 0, 0
    for i in range(5):
        if i == 0 and new_user_prediction[i] in test:
            count_1 = 1.0
        if i < 3 and new_user_prediction[i] in test:
            count_3 += 1.0
        if new_user_prediction[i] in test:
            count_5 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, 1)
    ndcg_tmp_3 = NDCG_at_k(dcg_list, idcg_list, 3)
    ndcg_tmp_5 = NDCG_at_k(dcg_list, idcg_list, 5)

    # precision@k
    precision_1 = count_1
    precision_3 = count_3 / 3.0
    precision_5 = count_5 / 5.0

    l = len(test)
    if l == 0:
        l = 1
    # recall@k
    recall_1 = count_1 / l
    recall_3 = count_3 / l
    recall_5 = count_5 / l

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_3, precision_5]),\
           np.array([recall_1, recall_3, recall_5]),\
           np.array([ndcg_tmp_1, ndcg_tmp_3, ndcg_tmp_5])

# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_precision_recall_ndcg(new_user_prediction, test):
    dcg_list = []

    # compute the number of true positive items at top k
    count_1, count_5, count_10, count_15 = 0, 0, 0, 0
    for i in range(5):
        if i == 0 and new_user_prediction[i][0] in test:
            count_1 = 1.0
        if i < 5 and new_user_prediction[i][0] in test:
            count_5 += 1.0
        if i < 10 and new_user_prediction[i][0] in test:
            count_10 += 1.0
        if new_user_prediction[i][0] in test:
            count_15 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, 1)
    ndcg_tmp_5 = NDCG_at_k(dcg_list, idcg_list, 5)
    ndcg_tmp_10 = NDCG_at_k(dcg_list, idcg_list, 10)
    ndcg_tmp_15 = NDCG_at_k(dcg_list, idcg_list, 15)

    # precision@k
    precision_1 = count_1
    precision_5 = count_5 / 5.0
    precision_10 = count_10 / 10.0
    precision_15 = count_15 / 15.0

    l = len(test)
    if l == 0:
        l = 1
    # recall@k
    recall_1 = count_1 / l
    recall_5 = count_5 / l
    recall_10 = count_10 / l
    recall_15 = count_15 / l

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_5, precision_10, precision_15]),\
           np.array([recall_1, recall_5, recall_10, recall_15]),\
           np.array([ndcg_tmp_1, ndcg_tmp_5, ndcg_tmp_10, ndcg_tmp_15])

# calculate the metrics of the result
def test_model_all(prediction, test_mask, train_mask):
    precision_1, precision_5, precision_10, precision_15 = 0.0000, 0.0000, 0.0000, 0.0000
    recall_1, recall_5, recall_10, recall_15 = 0.0000, 0.0000, 0.0000, 0.0000
    ndcg_1, ndcg_5, ndcg_10, ndcg_15 = 0.0000, 0.0000, 0.0000, 0.0000
    precision = np.array([precision_1, precision_5, precision_10, precision_15])
    recall = np.array([recall_1, recall_5, recall_10, recall_15])
    ndcg = np.array([ndcg_1, ndcg_5, ndcg_10, ndcg_15])

    prediction = prediction + train_mask * -100000.0

    user_num = prediction.shape[0]

    # by DM
    r_f_table = np.zeros((user_num, 17))

    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        u_test = np.where(u_test == 1)[0]  # the indices of the true positive items in the test set
        u_pred = prediction[u, :]

        #by DM
        u_train = train_mask[u, :]
        r_f_table[u][0] = len(np.where(u_train > 0)[0])

        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
        top15 = (np.array([top15_item_idx_no_train, u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if not len(u_test) == 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top15, u_test)

            r_f_table[u][1] = precision_u[0]
            r_f_table[u][2] = recall_u[0]
            r_f_table[u][3] = 2 * (precision_u[0] * recall_u[0]) / (precision_u[0] + recall_u[0]) if not precision_u[0] + recall_u[0] == 0 else 0
            r_f_table[u][4] = ndcg_u[0]

            r_f_table[u][5] = precision_u[1]
            r_f_table[u][6] = recall_u[1]
            r_f_table[u][7] = 2 * (precision_u[1] * recall_u[1]) / (precision_u[1] + recall_u[1]) if not precision_u[1] + recall_u[1] == 0 else 0
            r_f_table[u][8] = ndcg_u[1]

            r_f_table[u][9] = precision_u[2]
            r_f_table[u][10] = recall_u[2]
            r_f_table[u][11] = 2 * (precision_u[2] * recall_u[2]) / (precision_u[2] + recall_u[2]) if not precision_u[2] + recall_u[2] == 0 else 0
            r_f_table[u][12] = ndcg_u[2]

            r_f_table[u][13] = precision_u[3]
            r_f_table[u][14] = recall_u[3]
            r_f_table[u][15] = 2 * (precision_u[3] * recall_u[3]) / (precision_u[3] + recall_u[3]) if not precision_u[3] + recall_u[3] == 0 else 0
            r_f_table[u][16] = ndcg_u[3]

            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            user_num -= 1

    # compute the average over all users
    precision /= user_num
    recall /= user_num
    ndcg /= user_num
    '''
    print ('precision_1\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]' \
          % (precision[0],
             precision[1],
             precision[2],
             precision[3]))
    print ('recall_1   \t[%.7f],\t||\t recall_5   \t[%.7f],\t||\t recall_10   \t[%.7f],\t||\t recall_15   \t[%.7f]' \
          % (recall[0], recall[1],
             recall[2], recall[3]))
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[2] == 0 else 0
    f_measure_15 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[3] == 0 else 0
    print ('f_measure_1\t[%.7f],\t||\t f_measure_5\t[%.7f],\t||\t f_measure_10\t[%.7f],\t||\t f_measure_15\t[%.7f]' \
          % (f_measure_1,
             f_measure_5,
             f_measure_10,
             f_measure_15))
    f_score = [f_measure_1, f_measure_5, f_measure_10, f_measure_15]
    print ('ndcg_1     \t[%.7f],\t||\t ndcg_5     \t[%.7f],\t||\t ndcg_10     \t[%.7f],\t||\t ndcg_15     \t[%.7f]' \
          % (ndcg[0],
             ndcg[1],
             ndcg[2],
             ndcg[3]))
    '''
    print('%.7f, %.7f, %.7f, %.7f' \
          % (precision[0],
             precision[1],
             precision[2],
             precision[3]))
    print('%.7f, %.7f, %.7f, %.7f' \
          % (recall[0], recall[1],
             recall[2], recall[3]))
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[
        0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[
        1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[
        2] == 0 else 0
    f_measure_15 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[
        3] == 0 else 0
    print('%.7f, %.7f, %.7f, %.7f' \
          % (f_measure_1,
             f_measure_5,
             f_measure_10,
             f_measure_15))
    f_score = [f_measure_1, f_measure_5, f_measure_10, f_measure_15]
    print('%.7f, %.7f, %.7f, %.7f' \
          % (ndcg[0],
             ndcg[1],
             ndcg[2],
             ndcg[3]))
    return precision, recall, f_score, ndcg, r_f_table


def metric_record(precision, recall, f_score, NDCG, args, metric_path):  # record all the results' details into files
    path = metric_path + '.txt'

    with open(path, 'w') as f:
        f.write(str(args) + '\n')
        f.write('precision:' + str(precision) + '\n')
        f.write('recall:' + str(recall) + '\n')
        f.write('f score:' + str(f_score) + '\n')
        f.write('NDCG:' + str(NDCG) + '\n')
        f.write('\n')
        f.close()


def get_train_instances(train_R, neg_sample_rate):
    """
    genderate training dataset for NCF models in each iteration
    :param train_R:
    :param neg_sample_rate:
    :return:
    """
    # randomly sample negative samples
    mask = neg_sampling(train_R, range(train_R.shape[0]), neg_sample_rate)

    user_input, item_input, labels = [], [], []
    idx = np.array(np.where(mask == 1))
    for i in range(idx.shape[1]):
        # positive instance
        u_i = idx[0, i]
        i_i = idx[1, i]
        user_input.append(u_i)
        item_input.append(i_i)
        labels.append(train_R[u_i, i_i])
    return user_input, item_input, labels


def neg_sampling(train_R, idx, neg_sample_rate):
    """
    randomly negative smaples
    :param train_R:
    :param idx:
    :param neg_sample_rate:
    :return:
    """
    num_cols = train_R.shape[1]
    num_rows = train_R.shape[0]
    # randomly sample negative samples
    mask = copy.copy(train_R)
    if neg_sample_rate == 0:
        return mask
    for b_idx in idx:
        mask_list = mask[b_idx, :]
        unobsv_list = np.where(mask_list == 0)
        unobsv_list = unobsv_list[0]  # unobserved indices
        obsv_num = num_cols - len(unobsv_list)
        neg_num = int(obsv_num * neg_sample_rate)
        if neg_num > len(unobsv_list):  # if the observed positive ratings are more than the half
            neg_num = len(unobsv_list)
        if neg_num == 0:
            neg_num = 1
        neg_samp_list = np.random.choice(unobsv_list, size=neg_num, replace=False)
        mask_list[neg_samp_list] = 1
        mask[b_idx, :] = mask_list
    return mask


def pairwise_neg_sampling(train_R, r_idx, c_idx, neg_sample_rate):
    R = train_R[r_idx, :]
    R = R[:, c_idx]
    p_input, n_input = [], []
    obsv_list = np.where(R != 0)

    unobsv_mat = []
    for r in range(R.shape[0]):
        unobsv_list = np.where(R[r, :] == 0)
        unobsv_list = unobsv_list[0]
        unobsv_mat.append(unobsv_list)

    for i in range(len(obsv_list[1])):
        # positive instance
        u = obsv_list[0][i]
        # negative instances
        unobsv_list = unobsv_mat[u]
        neg_samp_list = np.random.choice(unobsv_list, size=neg_sample_rate, replace=False)
        for ns in neg_samp_list:
            p_input.append([u, obsv_list[1][i]])
            n_input.append([u, ns])
    # print('dataset size = ' + str(len(p_input)))
    return np.array(p_input), np.array(n_input)


# calculate the metrics of the result
def test_model_batch(prediction, test_mask, train_mask):
    precision_1, precision_5, precision_10, precision_15 = 0.0000, 0.0000, 0.0000, 0.0000
    recall_1, recall_5, recall_10, recall_15 = 0.0000, 0.0000, 0.0000, 0.0000
    ndcg_1, ndcg_5, ndcg_10, ndcg_15 = 0.0000, 0.0000, 0.0000, 0.0000
    precision = np.array([precision_1, precision_5, precision_10, precision_15])
    recall = np.array([recall_1, recall_5, recall_10, recall_15])
    ndcg = np.array([ndcg_1, ndcg_5, ndcg_10, ndcg_15])

    prediction = prediction + train_mask * -100000.0

    user_num = prediction.shape[0]
    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        u_test = np.where(u_test == 1)[0]  # the indices of the true positive items in the test set
        u_pred = prediction[u, :]

        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
        top15 = (np.array([top15_item_idx_no_train, u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if not len(u_test) == 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top15, u_test)
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            user_num -= 1

    return precision, recall, ndcg


# calculate the metrics of the result
def test_model_cold_start(prediction, test_mask, train_mask):
    precision_1, precision_5, precision_10, precision_15 = 0.0000, 0.0000, 0.0000, 0.0000
    recall_1, recall_5, recall_10, recall_15 = 0.0000, 0.0000, 0.0000, 0.0000
    ndcg_1, ndcg_5, ndcg_10, ndcg_15 = 0.0000, 0.0000, 0.0000, 0.0000
    precision = np.array([precision_1, precision_5, precision_10, precision_15])
    recall = np.array([recall_1, recall_5, recall_10, recall_15])
    ndcg = np.array([ndcg_1, ndcg_5, ndcg_10, ndcg_15])

    prediction = prediction + train_mask * -100000.0

    user_num = prediction.shape[0]
    n = 0
    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        u_test = np.where(u_test == 1)[0]  # the indices of the true positive items in the test set
        if len(u_test) > 10:
            continue
        u_pred = prediction[u, :]

        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
        top15 = (np.array([top15_item_idx_no_train, u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if not len(u_test) == 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top15, u_test)
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
            n += 1

    # compute the average over all users
    precision /= n
    recall /= n
    ndcg /= n
    print ('precision_1\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]' \
          % (precision[0],
             precision[1],
             precision[2],
             precision[3]))
    print ('recall_1   \t[%.7f],\t||\t recall_5   \t[%.7f],\t||\t recall_10   \t[%.7f],\t||\t recall_15   \t[%.7f]' \
          % (recall[0], recall[1],
             recall[2], recall[3]))
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[2] == 0 else 0
    f_measure_15 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[3] == 0 else 0
    print ('f_measure_1\t[%.7f],\t||\t f_measure_5\t[%.7f],\t||\t f_measure_10\t[%.7f],\t||\t f_measure_15\t[%.7f]' \
          % (f_measure_1,
             f_measure_5,
             f_measure_10,
             f_measure_15))
    f_score = [f_measure_1, f_measure_5, f_measure_10, f_measure_15]
    print ('ndcg_1     \t[%.7f],\t||\t ndcg_5     \t[%.7f],\t||\t ndcg_10     \t[%.7f],\t||\t ndcg_15     \t[%.7f]' \
          % (ndcg[0],
             ndcg[1],
             ndcg[2],
             ndcg[3]))
    return precision, recall, f_score, ndcg


def test_model_factor(prediction, test_mask, train_mask):
    item_list = np.zeros(train_mask.shape[1])
    item_list_rank = np.zeros(train_mask.shape[1])

    prediction = prediction + train_mask * -100000.0

    user_num = prediction.shape[0]
    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        u_test = np.where(u_test == 1)[0]  # the indices of the true positive items in the test set
        len_u_test = len(u_test)
        u_pred = prediction[u, :]

        top10_item_idx_no_train = np.argpartition(u_pred, -10)[-10:]
        item_list[top10_item_idx_no_train] += 1
        for i in range(len(top10_item_idx_no_train)):
            item_list_rank[top10_item_idx_no_train[i]] += (10 - i)

    item_count = np.sum(train_mask, axis=0)
    df = pd.DataFrame({'item_pred_freq': item_list, 'item_count': item_count})
    df.to_csv('data/no-factor' + time.strftime('%y-%m-%d-%H-%M-%S', time.localtime()) + '.csv')
    df = pd.DataFrame({'item_pred_rank': item_list_rank, 'item_count': item_count})
    df.to_csv('data/rank-no-factor' + time.strftime('%y-%m-%d-%H-%M-%S', time.localtime()) + '.csv')
