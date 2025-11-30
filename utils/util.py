import numpy as np
import copy
import torch
import random
from sklearn.cluster import KMeans
from utils.finch import FINCH


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Method   : {args.method}\n')
    print(f'    Model     : {args.model}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')
    print(f'    Dataset  : {args.dataset}')
    print(f'    Clients   : {args.num_clients}\n')

    print('    Federated parameters:')
    if args.label_iid:
        print('   Label IID')
    else:
        print('   Label Non-IID')
    if args.feature_iid:
        print('   Feature IID')
    else:
        print('   Feature Non-IID')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return


def average_protos(protos):
    """
    Average the protos for each local user
    {label: [proto1, proto2, ...], ...}
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = np.stack(proto_list)
        agg_protos[label] = np.mean(proto, axis=0)

    return agg_protos

def cluster_protos_finch(protos_label_dict):
    agg_protos = {}
    num_p = 0
    for [label, proto_list] in protos_label_dict.items():
        proto_list = np.stack(proto_list)
        c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                    ensure_early_exit=False, verbose=False)
        num_protos, num_partition = c.shape
        class_cluster_list = []
        for idx in range(num_protos):
            class_cluster_list.append(c[idx, -1])
        class_cluster_array = np.array(class_cluster_list)
        uniqure_cluster = np.unique(class_cluster_array).tolist()
        agg_selected_proto = []

        for _, cluster_index in enumerate(uniqure_cluster):
            selected_array = np.where(class_cluster_array == cluster_index)
            selected_proto_list = proto_list[selected_array]
            cluster_proto_center = np.mean(selected_proto_list, axis=0)
            agg_selected_proto.append(cluster_proto_center)

        agg_protos[label] = agg_selected_proto
        num_p += num_clust[-1]
    return agg_protos, num_p / len(protos_label_dict)


def local_cluster_collect(local_cluster_protos):
    global_collected_protos = {}
    for [idx, cluster_protos_label] in local_cluster_protos.items():
        for [label, cluster_protos_list] in cluster_protos_label.items():
            for i in range(len(cluster_protos_list)):
                if label in global_collected_protos.keys():
                    global_collected_protos[label].append(cluster_protos_list[i])
                else:
                    global_collected_protos[label] = [cluster_protos_list[i]]
    return global_collected_protos


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg


def average_weights_noniid(w, num_list):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    num_sum = sum(num_list)
    for key in w[0].keys():
        w_avg[0][key] = num_list[0] * w[0][key]
        for i in range(1, len(w)):
            w_avg[0][key] += num_list[i] * w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], num_sum)
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():

            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0]
            for i in proto_list:
                proto += i
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0]
    return agg_protos_label


def proto_aggregation_cluster(global_protos_list):
    agg_protos_label = dict()
    for label in global_protos_list.keys():
        for i in range(len(global_protos_list[label])):
            if label in agg_protos_label:
                agg_protos_label[label].append(global_protos_list[label][i])
            else:
                agg_protos_label[label] = [global_protos_list[label][i]]

    for [label, proto_list] in agg_protos_label.items():
        # print(len(proto_list))
        if len(proto_list) > 1:
            proto = 0 * proto_list[0]
            for i in proto_list:
                proto += i
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0]

    return agg_protos_label


def get_NEW_global_protos(global_average_protos_N, global_average_cluster_N_M_protos):
    """
    返回每个类别的多个全局原型（前N维相同，后面不同），
    Args:
        global_average_protos_N: 前N维度的全局平均原型，每个类别只有一个原型 {label: prot , ...}
        global_average_cluster_N_M_protos: 剩余维度的全局聚类原型，每个类别可能有多个原型,是一个列表  {label: [proto1, proto2, ...],...}
    Returns:
        global_NEW_protos: {label: [proto1, proto2, ...]}，每个类别的多个全局原型
    """
    global_NEW_protos = {}
    for label in global_average_protos_N.keys():
        front_features = global_average_protos_N[label]
        # 兼容聚类为单个或多个的情况
        if isinstance(global_average_cluster_N_M_protos[label], list):
            cluster_centers = global_average_cluster_N_M_protos[label]
        else:
            cluster_centers = [global_average_cluster_N_M_protos[label]]
        proto_list = []
        for cluster_center in cluster_centers:
            new_proto = np.concatenate([front_features, cluster_center])
            proto_list.append(copy.deepcopy(new_proto))
        global_NEW_protos[label] = proto_list
    return global_NEW_protos




def get_local_N_M_protos(protos_dict, N):
    """
    将本地原型分割成前N维和剩余维度
    Args:
        protos_dict: 本地原型字典，格式与average_protos函数相同
                    {label: [proto1, proto2, ...], ...}
    """
    local_N_protos = {}
    local_N_M_protos = {}
    
    for [label, proto_list] in protos_dict.items():
        # 初始化该标签的两个列表
        local_N_protos[label] = []
        local_N_M_protos[label] = []
        # 对每个原型进行分割
        for proto in proto_list:
            # 分割原型为前N维和剩余部分
            front_features = proto[:N]
            back_features = proto[N:]
            # 添加到对应的字典中
            local_N_protos[label].append(front_features)
            local_N_M_protos[label].append(back_features)
    
    return local_N_protos, local_N_M_protos


def calculate_optimal_N(all_protos, num_classes, dataset, total_dim=512, var_threshold=None, min_N=180, max_N=340, plot_histogram=False): #0.05
    # Set dataset-specific threshold if not provided
    if var_threshold is None:
        var_threshold = 0.55 if dataset.lower() == 'digit' else 0.03
    print(f"var_threshold: {var_threshold}")
    if not all_protos or len(all_protos) == 0:
        print("Warning: Empty prototype dictionary, returning default N value")
        return min_N
    # Collect all prototypes without class distinction
    all_protos_list = []
    for label in all_protos:
        class_protos = all_protos[label]
        if not isinstance(class_protos, list):
            class_protos = [class_protos]
        all_protos_list.extend(class_protos)
    
    if len(all_protos_list) <= 1:
        print("Warning: Not enough prototypes for variance calculation, returning default N value")
        return min_N
    
    # Convert to numpy array
    all_protos_array = np.array(all_protos_list)
    # Calculate variance for each dimension across all prototypes
    variance_per_dim = np.var(all_protos_array, axis=0)
    # Calculate optimal N: count dimensions with variance below threshold
    optimal_N = np.sum(variance_per_dim < var_threshold)
    # Apply boundary constraints with random adjustment
    if optimal_N < min_N:
        optimal_N = min_N + random.randint(10, 20)
    elif optimal_N > max_N:
        optimal_N = max_N - random.randint(10, 20)
    
    return optimal_N

def local_avg_collect(local_avg_protos):
    """
    local_avg_protos:    {idx:{label: proto, ...}, ... }
    """
    global_collected_protos = {}
    for idx, avg_protos_label in local_avg_protos.items():
        for label, avg_proto in avg_protos_label.items():
            if label in global_collected_protos:
                global_collected_protos[label].append(avg_proto)
            else:
                global_collected_protos[label] = [avg_proto]
    return global_collected_protos

