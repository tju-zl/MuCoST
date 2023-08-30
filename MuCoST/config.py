import argparse


def set_arg():
    parser = argparse.ArgumentParser(description='Hyperparameter of MuCoST')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--latent_dim', type=int, default=50, help='default dim of latent embedding')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='default weight decay of optimizer')
    parser.add_argument('--log_step', type=int, default=10, help='printing times of total loss')
    parser.add_argument('--drop_feat_p', type=float, default=0.2, help='probability of feature dropout')
    parser.add_argument('--flow', type=str, default='source_to_target', help='flow of message passing')
    parser.add_argument('--radius', type=int, default=150, help='radius KNN, 10x=150, stereo=45')
    parser.add_argument('--rknn', type=int, default=6, help='radius KNN, k=6')
    parser.add_argument('--knn', type=int, default=6, help='num of nearest neighbors')
    parser.add_argument('--n_domain', type=int, default=0, help='number of spatial domains, key param for clustering')
    parser.add_argument('--temp', type=float, default=0.05, help='InfoNCE temperature')
    parser.add_argument('--n_refine', type=int, default=25, help='Refine labels using spatial map')
    parser.add_argument('--mode_his', type=str, default='noh', help='noh=no histology, his=histology')
    parser.add_argument('--mode_rknn', type=str, default='rknn', help='rknn, knn')

    return parser
