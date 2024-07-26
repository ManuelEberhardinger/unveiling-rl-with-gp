from dsl import *
from tasks import *
from gp import GPStitch
import wandb
import os
import argparse


parser = argparse.ArgumentParser(
    description='Train the logical program policies on the ForagingEnv')
parser.add_argument('--wandb_key', type=str, default=None,
                    help='Wandb API key for logging')
parser.add_argument('--max_depth', type=int, default=6,
                    help='max depth of random subtrees')
parser.add_argument('--bloat_weight', type=float, default=0.025,
                    help='the bloat weight to tackle exploding of the tree size')
parser.add_argument('--generations', type=int, default=20000,
                    help='Number of iterations')
parser.add_argument('--tournament_size', type=int, default=50,
                    help='Number of trees in tournament selection')
parser.add_argument('--pop_size', type=int, default=1000,
                    help='Population size')
parser.add_argument('--compress', action='store_true', default=True)
parser.add_argument('--calc_stats', action='store_true', default=False)
parser.add_argument('--threshold', type=float, default=0.98,
                    help='Start threshold to increase sequence length')
parser.add_argument('--crossover_prob', type=float, default=0.2,
                    help='The crossover probability')
parser.add_argument('--mutate_prob', type=float, default=0.5,
                    help='The mutation probability')
parser.add_argument('--adapt_threshold', type=float, default=0.02,
                    help='Decrease threshold after increase of sequence length')
parser.add_argument('--epsilon_start', type=float, default=0.5,
                    help='Start value of the epsilon schedule')
parser.add_argument('--random_state', type=int, default=42,
                    help='Random state')
parser.add_argument('--max_invented_length', type=int, default=6,
                    help='Maximum number of used production rules for an extracted function')
parser.add_argument('--inrease_each_n_steps', type=int, default=10,
                    help='If set, no threshold is used and sequence length is incremented each n steps.')
parser.add_argument('--max_n_invented', type=int, default=5,
                    help='Only add max n functions')

args = parser.parse_args()
print('Run GP with args:', ' '.join(f'{k}={v}' for k, v in vars(args).items()))

data_file = 'data/minigrid-perfect_maze-eval.npy'  # train data
data_file = "data/minigrid-perfect_maze-train.npy"  # eval data
data = np.load(data_file, allow_pickle=True)
parsed_data = parseData(data)
g = Grammar(*base_primitives(), max_invented_length=args.max_invented_length)
run_name = f"random_state={args.random_state}-inrease_each_n_steps={args.inrease_each_n_steps}-compress={args.compress}"

if args.wandb_key is not None:
    os.environ['WANDB_API_KEY'] = args.wandb_key
    wandb.init(project="GP-unveil-RL", name=run_name, config=vars(args))
gp = GPStitch(g, max_depth=args.max_depth, bloat_weight=args.bloat_weight, generations=args.generations, log_results=args.wandb_key is not None,
              run_name=run_name, tournament_size=args.tournament_size, pop_size=args.pop_size, random_state=args.random_state)
gp.train(parsed_data, compress=args.compress, threshold=args.threshold, mutate_prob=args.mutate_prob,
         adapt_threshold=args.adapt_threshold, calc_stats=args.calc_stats, crossover_prob=args.crossover_prob, epsilon_start=args.epsilon_start, inrease_each_n_steps=args.inrease_each_n_steps, max_n_invented=args.max_n_invented)
