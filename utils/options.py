import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default="cuda", type=str, help="cpu, cuda, or others")
    parser.add_argument('--gpu', default=0, type=int, help="index of gpu")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')
    parser.add_argument('--method', type=str, default='proposed', help="proposed")
    parser.add_argument('--model', type=str, default='resnet', help='model type')
    parser.add_argument('--percent', type=float, default=1, help="percentage of dataset to train")
    parser.add_argument('--tau', type=float, default=0.07, help="loss temperature")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum')
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size")
    parser.add_argument('--test_bs', type=int, default=32, help="test batch size")
    parser.add_argument('--train_ep', type=int, default=2, help="number of local epochs")
    parser.add_argument('--saveprotos', type=bool, default=False, help="save the test protos for visualization")

    parser.add_argument('--feature_iid', type=bool, default=False, help='feature IID')
    parser.add_argument('--label_iid', type=bool, default=True, help='label IID')
    parser.add_argument('--beta', type=float, default=0.5, help="diri distribution parameter")

    parser.add_argument('--dataset', type=str, default='office', help="digit,office,domain") 
    parser.add_argument('--num_clients', type=int, default=4, help="number of clients: 5 for digit, 4 for office, 6 for domain")
    parser.add_argument('--rounds', type=int, default=80, help="number of rounds of training: 50 for digit, 80 for office, 200 for domain")
    parser.add_argument('--lamb', type=float, default=10, help="CE loss weight parameter: 50 for digit, 10 for office, 1 for domain")
    parser.add_argument('--fixed_N', type=bool, default=False, help="N: The number of dimensions in the generalized portion (fix it for faster calculation).")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--n_per_class', type=int, default=10, help="num of samples per class: 10 for digit and office, 30 for domain")
    parser.add_argument('--n_per_class_test', type=int, default=100, help="num of samples per class for testing")
    
    parser.add_argument('--swanlab', type=bool, default=False, help="swanlab")
    parser.add_argument('--num_exps', type=int, default=5, help="number of experiments")

    args = parser.parse_args()

    return args

