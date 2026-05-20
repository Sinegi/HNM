# Stabilizing Causal Structure Learning under Heteroscedasticity: Analysis and Mitigation of Optimization Failures

Official implementation of **Stabilizing Causal Structure Learning under Heteroscedasticity: Analysis and Mitigation of Optimization Failures** (KDD 2026).

## Requirements

- python=3.9
- numpy=2.0.1
- scipy=1.13.1
- torch = 2.4.0
- scikit-learn=1.6.1
- python-igraph

## Example: generating nonlinear heteroscedastic data

To generate nonlinear data with heteroscedastic noise, run:

```console
$ python data_generation.py --num_size 5 --s0 2 --sem mlp --graph_type ER --sample_size 1000 --data_type hetero
````

## Example: running the algorithm

After generating the nonlinear heteroscedastic data using the script above, run the proposed algorithm as follows:

```console
$ python main.py --tau 5 --max 1000 --init 100
```

where `--tau` denotes the decay-rate parameter `tau`, `--max` denotes the transition step `t_star`, and `--init` denotes the initial scheduling weight `lambda_reg(0)`.

The scheduling coefficient is computed as:

```text
lambda_reg(t) = lambda_reg(0) * exp(-t / (t_star / tau))
```

In the example above, `--tau 5` sets `tau = 5`, `--max 1000` sets `t_star = 1000`, and `--init 100` sets `lambda_reg(0) = 100`.

Here, `t_star` corresponds to the transition point $t^{*}$ in the paper.
