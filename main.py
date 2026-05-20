import numpy as np
import torch
import torch.nn as nn
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
from locally_connected import LocallyConnected
import os
from torch.nn import functional as F
import random
import argparse
from utils import count_accuracy, set_random_seed
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MLP(nn.Module):
    def __init__(self, dims, bias=False):
        super(MLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims

        # First layer weights, W1 -> W1+, W1-
        self.W1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.W1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.W1_pos.weight.bounds = self._bounds()
        self.W1_neg.weight.bounds = self._bounds()
        self.edge_probs = nn.Parameter(torch.full((d, d * dims[1]), -0.05))

        # Second layer weights for mean estimation W2
        self.W2 = LocallyConnected(d, dims[1], 1, bias=bias)
        self.W2.weight.data[:] = torch.from_numpy(np.random.randn(d, dims[1], 1))

        # Second layer weights for variance estimate W3
        self.W3 = LocallyConnected(d, dims[1], 1, bias=bias)
        self.W3.weight.data[:] = torch.from_numpy(np.random.randn(d, dims[1], 1))
        # self.W4 = nn.Parameter(torch.randn(1,))
        self.acfun = nn.Softplus()
        self.tanh = nn.Tanh()

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds
    
    def sample_adj_matrix(self, tau=1.0, hard=False):
        if hard:
            return torch.sigmoid(self.edge_probs)
        noise = torch.rand_like(self.edge_probs)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        logits = self.edge_probs + gumbel_noise*0.1
        adj_matrix = torch.sigmoid(logits / tau)
        return adj_matrix

    def forward(self, x):
        A = self.sample_adj_matrix(hard=True)
        W1_pos = self.W1_pos.weight * A.T
        W1_neg = self.W1_neg.weight * A.T
        pos = F.linear(x, W1_pos, self.W1_pos.bias) 
        neg = F.linear(x, W1_neg, self.W1_neg.bias)  
        x = pos - neg # [n, d * m1]

        x = x.view(-1, self.dims[0], self.dims[1]) # [n, d, m1]
        h = torch.sigmoid(x) # [n, d, m1]
        mu = self.W2(h) # [n, d, m2 = 1]
        mu = mu.squeeze(dim=2)  # [n, d]

        var = F.softplus(self.W3(h.detach())) #+ torch.sigmoid(self.W4)
        var = var.squeeze(dim=2) # [n, d]
        return mu, var

    def h_func(self):
        d = self.dims[0]
        W1 = (self.W1_pos.weight - self.W1_neg.weight) * torch.sigmoid(self.edge_probs).T
        W1 = W1.view(d, -1, d)
        A = torch.sum(W1 * W1, dim=1).t()
        h = trace_expm(A) - d
        return h

    def l2_reg(self):
        reg = 0.
        W1_weight = self.W1_pos.weight - self.W1_neg.weight  # [j * m1, i]
        reg += torch.sum(W1_weight ** 2)
        reg += torch.sum(self.W2.weight ** 2)
        reg += torch.sum(self.W3.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.W1_pos.weight + self.W1_neg.weight)
        reg += torch.sum(torch.sigmoid(self.edge_probs))
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        d = self.dims[0]
        W1 = (self.W1_pos.weight - self.W1_neg.weight) * torch.sigmoid(self.edge_probs).T
        W1 = W1.view(d, -1, d)
        A = torch.sum(W1 * W1, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W


def negative_log_likelihood_loss(mu, var, target, max_epoch=1000, init_T=100, tau=5):
    global COUNTER
    initial_T = init_T
    max_epochs = max_epoch
    tau = max_epochs / tau
    flag = True
    if COUNTER < max_epochs:
        T_value = initial_T * np.exp(-np.arange(max_epochs)[COUNTER] / tau)
        COUNTER = COUNTER + 1
    else:
        T_value = initial_T * np.exp(-(max_epochs) / tau)
        flag = False
    n = target.shape[0]
    if flag:
        stopnll = 0.5 / n * torch.sum(torch.log(2 * np.pi * var) + (target - mu.detach()) ** 2 / var)
        mse = T_value * 0.5 / n * torch.sum((mu - target) ** 2)
        varreg = T_value * 0.5 / n * torch.sum((torch.sqrt(var) - torch.abs(mu - target)) ** 2)
        return (stopnll + mse + varreg)*0.333
    else:
        nll = 0.5 / n * torch.sum(torch.log(2 * np.pi * var) + (target - mu) ** 2 / var)
        return nll

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, x, lamb1, lamb2, rho, alpha, h, rho_max, max_epoch, init_T, tau):
    # lamb1 and lamb2 are the coefficients for the L1 and L2 regualrization. The values for these two hypterparameters need to be tuned.
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            x_hat, var = model(x)
            loss = negative_log_likelihood_loss(x_hat, var, x, max_epoch, init_T, tau)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lamb2 * model.l2_reg()
            l1_reg = lamb1 * model.fc1_l1_reg()
            obj = loss + penalty + l2_reg + l1_reg
            obj.backward()
            return obj
        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def train(model: nn.Module,
           X: torch.tensor,
           lamb1: float,
           lamb2: float,
           max_epoch:int,
           init_T=int,
           tau=int,
           max_iter: int = 100,
           h_tol: float = 1e-8,
           rho_max: float = 1e+16):
    # This is the Phase-II step, where we optimize over the structural parameters using fixed variances.
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lamb1, lamb2, rho, alpha, h, rho_max, max_epoch, init_T, tau)
        if h <= h_tol or rho >= rho_max:
            break


def run(model,
         X,
         lamb1,
         lamb2,
         max_epoch,
         init_T,
         tau,):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    X_torch = torch.from_numpy(X)
    train(model=model, X=X_torch, lamb1=lamb1, lamb2=lamb2, max_epoch=max_epoch, init_T=init_T, tau=tau)
    w_est = model.fc1_to_adj()
    return w_est

def main(args):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    dirname = ""
    if args.data_type == "hetero_nonlinear":
        dirname = "hetero"
    elif args.data_type == "homo_ev_nonlinear":
        dirname = "homoev"
    else:
        dirname = "homonv"
    X = np.loadtxt(f"./{dirname}_dataset/ER{args.s0}_dataset/X_num{args.num_size}_{args.data_type}_seed{args.random_seed}.csv", delimiter=",")
    W_true = np.loadtxt(f"./{dirname}_dataset/ER{args.s0}_dataset/W_true_{args.num_size}_{args.random_seed}.csv", delimiter=",")

    n, d = X.shape
    print(n, d, "s=", np.sum(W_true))
    global COUNTER
    COUNTER = 0

    model = MLP(dims=[d, 10, 1], bias=False)
    A_est = run(model=model, X=X, lamb1=args.lamb1, lamb2=args.lamb2, max_epoch=args.max, init_T=args.init, tau=args.tau)
    A_est[A_est < 0.3] = 0
    result = count_accuracy(W_true, A_est != 0)
    os.makedirs("./results",exist_ok=True)
    with open(f"./results/er{args.s0}n{args.num_size}.txt", "a", encoding="utf-8") as file:
        file.write(f"SEED: {args.random_seed}\t")
        file.write(str(result))
        file.write("\n")
    print(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER', choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=1, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=10, help="variable size to be generated")
    parser.add_argument('--data_type', type=str, default='hetero_nonlinear', choices=['homo_ev_nonlinear', 'hetero_nonlinear', 'homo_nv_nonlinear'],help='data type')
    parser.add_argument('--sem', type=str, default='mlp', choices=['mlp', 'gp'], help='Types of SEM model')
    parser.add_argument('--lamb1', type=float, default=0.01, help="The coefficient of l1 regularization on parameters.")
    parser.add_argument('--lamb2', type=float, default=0.01, help="The coefficient of l2 regularization on parameters.")
    parser.add_argument('--random_seed', type=int, default=0, help="Random seeds.")
    # Our hyperparameters for scheduling
    parser.add_argument('--tau', type=int, default=5, help="Decay rate.")
    parser.add_argument('--max', type=int, default=1000, help="Transition point.")
    parser.add_argument('--init', type=int, default=100, help="The initial coefficient.")
    args = parser.parse_args()
    set_random_seed(args.random_seed)
    print("SEED:", args.random_seed)
    main(args)
