import numpy as np
import copy
import torch
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


def cluster_protos(protos, num_cluster=5):
    cluster_centers_label = {}
    for [label, proto_list] in protos.items():
        proto = np.stack(proto_list)
        kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(proto)
        # 将 NumPy 数组转换为列表
        cluster_centers_label[label] = [center for center in kmeans.cluster_centers_]
    return cluster_centers_label


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


def local_cluster_collect_N_M(local_cluster_protos):
    global_collected_protos = {}
    global_collected_protos_N = {}
    global_collected_protos_M_N = {}
    for [idx, cluster_protos_label] in local_cluster_protos.items():
        for [label, cluster_protos_list] in cluster_protos_label.items():
            for i in range(len(cluster_protos_list)):
                if label in global_collected_protos.keys():
                    global_collected_protos[label].append(cluster_protos_list[i])
                    global_collected_protos_N[label].append(cluster_protos_list[i][:410])
                    global_collected_protos_M_N[label].append(cluster_protos_list[i][410:])
                else:
                    global_collected_protos[label] = [cluster_protos_list[i]]
                    global_collected_protos_N[label] = [cluster_protos_list[i][:410]]
                    global_collected_protos_M_N[label] = [cluster_protos_list[i][410:]]

    return  global_collected_protos_N, global_collected_protos_M_N


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


def get_average_clusteraverage_protos(global_average_protos, global_cluster_N_M_protos):
    """
    将全局平均原型的前410维与M-N维的聚类原型拼接
    Args:
        global_average_protos: 全局平均原型
        global_cluster_N_M_protos: M-N维的聚类原型
    Returns:
        拼接后的原型字典，每个类可能包含多个完整原型
    """
    global_average_clusteraverage_protos = {}
    for label in global_average_protos.keys():
        # 获取该类别的前410维平均原型
        front_features = global_average_protos[label][:410]
        
        # 获取该类别的所有M-N维聚类中心
        if isinstance(global_cluster_N_M_protos[label], list):
            # 如果是多个聚类中心的情况
            back_features_list = global_cluster_N_M_protos[label]
        else:
            # 如果只有一个聚类中心
            back_features_list = [global_cluster_N_M_protos[label]]
        
        # 对每个M-N维聚类中心，都与前410维平均原型拼接
        complete_protos = []
        for back_features in back_features_list:
            # 拼接前410维和后面的维度
            complete_proto = np.concatenate([front_features, back_features])
            complete_protos.append(complete_proto)
        
        # 存储该类别的所有完整原型
        global_average_clusteraverage_protos[label] = complete_protos

    return global_average_clusteraverage_protos


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


def calculate_optimal_N(all_protos, num_classes, total_dim=512, var_threshold=0.05, min_N=192, max_N=330):
    """
    计算最优的N值，用于分割原型向量为域不变特征和域特定特征
    
    Args:
        all_protos: 全局收集的原型字典，格式为 {label: [proto1, proto2, ...], ...}
        num_classes: 类别数量
        total_dim: 原型向量的总维度，默认为512
        var_threshold: 方差阈值，用于判断特征是否为域不变特征，默认为0.1
        min_N: 最小N值，默认为96
        max_N: 最大N值，默认为384
    
    Returns:
        optimal_N: 最优的N值，表示前N维为域不变特征，剩余维度为域特定特征
    """
    # 检查输入是否为空
    if not all_protos or len(all_protos) == 0:
        print("警告: 输入原型字典为空，返回默认N值")
        return min_N
    
    # 初始化类别方差列表
    all_class_vars = []
    
    # 对每个类别计算内部原型在每个维度上的方差
    for label in all_protos:
        # 确保protos是一个列表，即使只有一个原型
        class_protos = all_protos[label]
        if not isinstance(class_protos, list):
            class_protos = [class_protos]
        
        # 转换为numpy数组
        class_protos = np.array(class_protos)
        
        if len(class_protos) > 1:
            # 如果有多个原型，计算每个维度的方差
            class_var = np.var(class_protos, axis=0)
        else:
            # 如果只有一个原型，方差为0向量
            class_var = np.zeros(total_dim)
        
        all_class_vars.append(class_var)
    
    # 如果没有类别数据，返回默认N值
    if not all_class_vars:
        print("警告: 没有有效的类别数据，返回默认N值")
        return min_N
    
    # 计算所有类别在每个维度上的平均方差
    # 先将所有类别的方差堆叠成一个数组
    all_vars_array = np.array(all_class_vars)  # 形状: (num_classes, total_dim)
    
    # 在类别间计算每个维度的平均方差
    avg_variance_per_dim = np.mean(all_vars_array, axis=0)  # 形状: (total_dim,)
    
    # 打印方差分布统计
    print(f"方差分布统计:")
    print(f"  平均方差: {np.mean(avg_variance_per_dim):.6f}")
    print(f"  方差最小值: {np.min(avg_variance_per_dim):.6f}")
    print(f"  方差最大值: {np.max(avg_variance_per_dim):.6f}")
    print(f"  方差中位数: {np.median(avg_variance_per_dim):.6f}")
    print(f"  小于阈值({var_threshold})的维度数: {np.sum(avg_variance_per_dim < var_threshold)}")
    
    # 计算最优N值：统计方差小于阈值的维度数量
    optimal_N = np.sum(avg_variance_per_dim < var_threshold)
    
    # 应用边界限制
    optimal_N = max(min_N, min(optimal_N, max_N))
    
    # 打印最优N值信息
    print(f"计算得到的最优N值: {np.sum(avg_variance_per_dim < var_threshold)}")
    print(f"边界限制后的最终N值: {optimal_N}")
    
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

