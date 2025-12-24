import torch
import numpy as np
import random
from tqdm import tqdm
import copy
import swanlab

from utils.options import args_parser
from utils.util import exp_details, average_protos, average_weights, cluster_protos_finch, local_cluster_collect, average_weights_noniid, get_NEW_global_protos, get_local_N_M_protos, local_avg_collect, calculate_optimal_N
from utils.data_util import prepare_data_digit, prepare_data_digits_noniid, prepare_data_office, prepare_data_office_noniid, prepare_data_domain
from utils.update import train_update, test_update
from models.resnet import resnet10

def proposed(args, train_dataset_list, test_dataset_list, user_groups, user_groups_test, local_model_list):

    global_cluster_N_M_protos={}
    local_cluster_protos_N_M={}
    local_avg_N_protos = {}
    global_N_protos = {}
    global_avg_cluster_protos={}
    global_cluster_N_M_protos_avg={}
    local_cluster_protos = {}
    global_collected_protos = {}
    N = 0
    if args.fixed_N:
        N = {'digit': 316, 'office': 192, 'domain': 192}[args.dataset]
       
    protos_test_list = [[[] for _ in range(args.num_classes)] for _ in range(args.num_clients)]
    num_list = []
    for i in range(len(user_groups)):
        num_list.append(len(user_groups[i]))
    print(num_list)
    for rd in tqdm(range(args.rounds)):
        # variance-aware decomposition
        if not args.fixed_N and rd >= 5 and rd % 5 == 0:  
            N = calculate_optimal_N(global_collected_protos, args.num_classes, dataset=args.dataset)             
            print(f"rd: {rd}，N modified to {N}")
            global_cluster_N_M_protos = {}
            global_N_protos = {}
            global_avg_cluster_protos = {}
            global_cluster_N_M_protos_avg = {}
            local_avg_N_protos = {}
            local_cluster_protos_N_M = {}
            global_collected_protos = {}

        print(f'\n | Global Training Round : {rd}')
        local_weights, local_loss1, local_loss2, local_loss_total, = [], [], [], []
        for idx in range(args.num_clients):
            local_model = train_update(args=args, dataset=train_dataset_list[idx % len(train_dataset_list)], idxs=user_groups[idx])
            w, loss, all_protos_dict = local_model.update_weights_proposed(idx, local_avg_N_protos,local_cluster_protos_N_M, global_N_protos, global_cluster_N_M_protos,global_cluster_N_M_protos_avg, global_avg_cluster_protos, model=copy.deepcopy(local_model_list[idx]), global_round=rd, N=N)
            
            # local dual-prototypes
            local_N_protos, local_N_M_protos = get_local_N_M_protos(all_protos_dict, N)  
            local_avg_N_protos[idx] = copy.deepcopy(average_protos(local_N_protos))  
            local_cluster_protos_N_M_idx, _ = cluster_protos_finch(local_N_M_protos)  
            local_cluster_protos_N_M[idx] = copy.deepcopy(local_cluster_protos_N_M_idx)  

            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            local_weights.append(copy.deepcopy(w))
            local_cluster_protos_dict, num_cls = cluster_protos_finch(all_protos_dict)
            local_cluster_protos[idx] = copy.deepcopy(local_cluster_protos_dict)

        if args.label_iid:
            local_weights_list = average_weights(local_weights)
        else:
            local_weights_list = average_weights_noniid(local_weights, num_list)
        for idx in range(args.num_clients):
            local_model_list[idx].load_state_dict(local_weights_list[idx])

        # local complete cluster prototypes (collected for N)
        global_collected_protos = local_cluster_collect(local_cluster_protos)

        # global dual-prototypes
        global_collected_N_protos = local_avg_collect(local_avg_N_protos)
        global_collected_N_M_protos =  local_cluster_collect(local_cluster_protos_N_M)
        global_N_protos = average_protos(global_collected_N_protos)  
        global_cluster_N_M_protos, _ = cluster_protos_finch(global_collected_N_M_protos)     
        global_cluster_N_M_protos_avg = average_protos(global_cluster_N_M_protos) 
        global_avg_cluster_protos = get_NEW_global_protos(global_N_protos, global_cluster_N_M_protos)  

        if rd % 10 == 0:
            with torch.no_grad():
                accs = []
                for idx in range(args.num_clients):
                    print('Test on user {:d}'.format(idx))
                    local_test = test_update(args=args, dataset=test_dataset_list[idx % len(test_dataset_list)], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights_list[idx], strict=True)
                    local_model_for_test.eval()
                    acc, _ = local_test.test_inference(idx, local_model_for_test)
                    accs.append(acc)
                print('Test avg acc:{:.5f}'.format(sum(accs) / len(accs)))
                if args.swanlab:
                    swanlab.log({
                        "test_acc": sum(accs) / len(accs)
                    })

    acc_mtx = torch.zeros([args.num_clients])
    with torch.no_grad():
        accs = []
        for idx in range(args.num_clients):
            print('Test on user {:d}'.format(idx))
            local_test = test_update(args=args, dataset=test_dataset_list[idx % len(test_dataset_list)], idxs=user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights_list[idx], strict=True)
            local_model_for_test.eval()
            acc, protos_test = local_test.test_inference(idx, local_model_for_test)
            accs.append(acc)
            acc_mtx[idx] = acc
            protos_test_list[idx] = protos_test
        print('Test avg acc:{:.5f}'.format(sum(accs) / len(accs)))
        if args.swanlab:
            swanlab.log({
                "test_acc": sum(accs) / len(accs),
            })
    return acc_mtx

def main(args):
    exp_details(args)

    # set random seed
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    print("Training on", args.device, '...')
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)
    else:
        torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # initialize dataset
    if (not args.feature_iid) and args.label_iid:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digit(args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office(args)
        elif args.dataset == 'domain':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domain(args)
    elif (not args.feature_iid) and (not args.label_iid):
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits_noniid(args.num_clients, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office_noniid(args.num_clients, args=args)
    # initialize model
    local_model_list = []
    for _ in range(args.num_clients):
        local_model = resnet10()
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    acc_mtx = proposed(args, train_dataset_list, test_dataset_list, user_groups, user_groups_test, local_model_list)

    return acc_mtx



if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    args = args_parser()
    if args.swanlab:
        swanlab.init(
            project="FedOrthrus",
            name=f"exp_{args.dataset}"
        )
    acc_mtx = np.zeros([args.num_exps, args.num_clients])

    for i in range(args.num_exps):
        print("Experiment:", i)
        args.seed = i
        acc_mtx[i, :] = main(args)
    np.save('acc.npy', np.array(acc_mtx), allow_pickle=True)

    print("The avg test acc of all exps are:")
    for j in range(args.num_clients):
        print('{:.2f}'.format(np.mean(acc_mtx[:, j]) * 100))

    print("The stdev of test acc of all exps are:")
    for j in range(args.num_clients):
        print('{:.2f}'.format(np.std(acc_mtx[:, j]) * 100))

    acc_avg = np.zeros([args.num_exps])
    for i in range(args.num_exps):
        acc_avg[i] = np.mean(acc_mtx[i, :]) * 100
        
    print("The avg and stdev test acc of all clients in the trials:")
    print('{:.2f}'.format(np.mean(acc_avg)))
    if args.swanlab:
        swanlab.log({
            "test_acc_avg": np.mean(acc_avg)
        })
    print('{:.2f}'.format(np.std(acc_avg)))
    if args.swanlab:
        swanlab.log({
            "test_acc_stdev": np.std(acc_avg)
        })
    for i in range(args.num_exps):
        print('{:.2f}'.format(acc_avg[i]),end=' ')