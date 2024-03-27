import time
import logging
import subprocess

import torch
from matplotlib import pyplot as plt

import skopt
from skopt import plots
from skopt.space import Real, Integer


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
                    handlers=[logging.FileHandler("../output/bo/printlog.txt", mode="w"),
                              logging.StreamHandler()])

space = [Real(2, 13, name='che_cutoff'),
         Integer(1, 15, name='che_max_num_nbrs'),
         Real(2, 15, name='vdw_cutoff'),
         Integer(1, 15, name='vdw_max_num_nbrs'),
         Integer(8, 128, name='edge_embedding_size'),
         Integer(32, 512, name='hidden_size'),
         Integer(1, 4, name='num_interactions'),
         ]


def evaluator(che_cutoff, che_max_num_nbrs, vdw_cutoff, vdw_max_num_nbrs, edge_embedding_size, hidden_size, num_interactions):
    with open("args.txt", 'w') as file:
        file.write('..\n')
        file.write('painn_cv\n')
        file.write('train\n')
        file.write('--device\ncuda\n')
        file.write('--pin_memory\n')
        file.write('--val_ratio\n0.1\n--batch_size\n400\n')
        file.write('--plateau_scheduler\n--initial_lr\n0.001\n--lr_factor\n0.5\n--lr_patience\n5\n')
        file.write('--stop_escape\n100\n--stop_patience\n25\n--max_steps\n10000\n--pre_weight\n0.1\n')
        file.write('--che_cutoff\n' + str(che_cutoff) + '\n--che_max_num_nbrs\n' + str(che_max_num_nbrs) + '\n--vdw_cutoff\n' + str(vdw_cutoff) + '\n--vdw_max_num_nbrs\n' + str(vdw_max_num_nbrs) + '\n')
        file.write('--node_input_size\n13\n--edge_embedding_size\n' + str(edge_embedding_size) + '\n--hidden_size\n' + str(hidden_size) + '\n--normalization\n')
        file.write('--num_interactions\n' + str(num_interactions) + '\n--atomwise_normalization\n')

    subprocess.run(["python", "main.py", "@args.txt"])
    
    state_dict = torch.load("../output/train/best_model.pth")

    logging.info("Iteration {} with RMSE = {:.5f} / {:.5f} (train / validation)"
                 .format(n, state_dict["loss"][-1][1], state_dict["loss"][-1][2]))

    return float(state_dict["best_val_loss"])


@skopt.utils.use_named_args(space)
def objective(**params):
    return evaluator(**params)


if __name__ == "__main__":

    n = 0  # The number of iterations that have occurred before the breakpoint
    num_calls = 100  # The total number of iterations to perform
    num_random = 100  # The number of initial random iterations (simulates random search when = num_calls)

    for i in range(num_calls - n):
        if n == 0:
            results = skopt.forest_minimize(objective, space, n_calls=1, n_initial_points=1)
            n += 1
            skopt.dump(results, f"../output/bo/results/{n}.pkl")
        else:
            old_results = skopt.load(f"../output/bo/results/{n}.pkl")
            if n < num_random:
                results = skopt.forest_minimize(objective, space, x0=old_results.x_iters, y0=old_results.func_vals, n_calls=1, n_initial_points=1)
            else:
                results = skopt.forest_minimize(objective, space, x0=old_results.x_iters, y0=old_results.func_vals, n_calls=1, n_initial_points=0)
            n += 1
            skopt.dump(results, f"../output/bo/results/{n}.pkl")
        time.sleep(60)

    results = skopt.load(f"../output/bo/results/{num_calls}.pkl")
    plt.plot(range(num_calls), results['func_vals'])
    plt.savefig("f.png")
    plots.plot_convergence(results)
    plt.savefig("../output/bo/convergence.png")
    plots.plot_evaluations(results)
    plt.savefig("../output/bo/evaluations.png")
    plt.figure(dpi=200, figsize=(8, 8))
    plots.plot_objective(results)
    plt.gcf().subplots_adjust(bottom=0.08)
    plt.gcf().subplots_adjust(left=0.08)
    plt.savefig("../output/bo/objective.png")

    logging.info(results.x)
