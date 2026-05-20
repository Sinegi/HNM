import numpy as np
import argparse
import utils as ut
import os
import torch



def main_nonlinear_syn(args):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # set up random seed
    ut.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # specify hyperparameters
    n, d, s0, graph_type, sem_type = args.sample_size, args.num_size, args.num_size * args.s0, args.graph_type, args.sem

    # generate graph
    W_true = ut.simulate_dag(d, s0, graph_type)
    
    
    dirname = ""
    if args.data_type == "hetero_nonlinear":
        dirname = "hetero"
    elif args.data_type == "homo_ev_nonlinear":
        dirname = "homoev"
    else:
        dirname = "homonv"
    
    os.makedirs(f"./{dirname}_dataset/ER{args.s0}_dataset",exist_ok=True)
    np.savetxt(f"./{dirname}_dataset/ER{args.s0}_dataset/W_true_{args.num_size}_{args.random_seed}.csv", W_true, delimiter=',')
    

    # generate nonlinear data
    if args.data_type == 'homo_ev_nonlinear':
        X = ut.simulate_nonlinear_sem(W_true, n, sem_type)
        # np.savetxt(f"X_{args.data_type}.csv", X, delimiter=',')
        np.savetxt(f"./{dirname}_dataset/ER{args.s0}_dataset/X_num{args.num_size}_{args.data_type}_seed{args.random_seed}.csv", X, delimiter=',')
    elif args.data_type == 'homo_nv_nonlinear':
        noise_var = np.random.uniform(0.5, 2, size=(d))
        noise_scale = np.sqrt(noise_var)
        X = ut.simulate_nonlinear_sem(W_true, n, sem_type, noise_scale)
        np.savetxt(f"./{dirname}_dataset/ER{args.s0}_dataset/X_num{args.num_size}_{args.data_type}_seed{args.random_seed}.csv", X, delimiter=',')
        # np.savetxt(f"X_{args.data_type}.csv", X, delimiter=',')
    elif args.data_type == 'hetero_nonlinear':
        X, var_est = ut.simulate_nonlinear_sem_hetero(W_true, n, sem_type)
        # np.savetxt(f"X_est.csv", X, delimiter=',')
        # np.savetxt(f"var_est.csv", var_est, delimiter=',')
        np.savetxt(f"./{dirname}_dataset/ER{args.s0}_dataset/X_num{args.num_size}_{args.data_type}_seed{args.random_seed}.csv", X, delimiter=',')
        # np.savetxt(f"var_{args.num_size}_{args.data_type}_{args.random_seed}.csv", var_est, delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER', choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=1, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=4, help="variable size to be generated")
    parser.add_argument('--data_type', type=str, default='hetero_nonlinear',
                        choices=['homo_ev', 'hetero_nonlinear', 'homo_nv'], help='data type')
    parser.add_argument('--sem', type=str, default='mlp', choices=['mlp', 'gp'], help='Types of SEM model')
    parser.add_argument('--lamb1', type=float, default=0.03, help="The coefficient of l1 regularization on parameters.")
    parser.add_argument('--lamb2', type=float, default=0.03, help="The coefficient of l2 regularization on parameters.")
    parser.add_argument('--random_seed', type=int, default=123, help="Random seeds.")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cuda_device', type=str, default="1", choices=["0", "1", "2", "3", "4", "5"])
    args = parser.parse_args()

    # generate nonlinear data
    print(f"Generate {args.sample_size} nonlinear {args.data_type} data from {args.graph_type}{args.s0} graphs with {args.num_size} variables.")
    main_nonlinear_syn(args)
    print(f"Data generation completes!")