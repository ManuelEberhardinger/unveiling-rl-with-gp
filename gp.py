import traceback
from collections import defaultdict
import random
from copy import deepcopy
import numpy as np
import stitch_core
import wandb
from pathlib import Path
from tree import *
from tasks import *
from functools import reduce
from collections import Counter
import time

TABLE_DATA = []


def log_solved_tasks(stats, seq_len, gen, verbose=False):
    try:
        t = None
        all = int(stats['tasks'])
        top = int(stats['hits'])
        rate_top = top/all * 100

        if verbose:
            print(f'{top}/{all} -> {rate_top}%')
        TABLE_DATA.append([seq_len, rate_top, top, all])
        t = wandb.Table(
            columns=["sequence length", "rate", "found programs", "all"], data=TABLE_DATA)
        if t is not None:
            wandb.log({'solved_stats': t}, step=gen)
    except Exception as e:
        traceback.print_exc()
        print(e)
        print(TABLE_DATA)


class GPResult():
    def __init__(self, population, fitnesses, grammar):
        self.population = population
        self.fitnesses = fitnesses
        self.grammar = grammar


class GPAlgorithm():
    def __init__(self, grammar, pop_size=60, min_depth=2, max_depth=4, generations=200, tournament_size=20, bloat_weight=0, random_state=42, run_name=None, log_results=False):
        self.pop_size = pop_size    # population size
        self.min_depth = min_depth     # minimal initial random tree depth
        self.max_depth = max_depth   # maximal initial random tree depth
        self.generations = generations   # maximal number of generations to run evolution
        # size of tournament for tournament selection
        self.tournament_size = tournament_size
        self.bloat_weight = bloat_weight  # True adds bloat control to fitness function
        self.random_state = random_state
        self.grammar = grammar
        self.log_results = log_results
        self.run_name = run_name
        if self.run_name is not None:
            self.output_path = f"./output/{self.run_name}/trees/"
            Path(self.output_path).mkdir(exist_ok=True, parents=True)

        # inist internal state of random number generator
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def init_population(self):
        pop = []

        while len(pop) < self.pop_size:
            try:
                t = TreeRoot(self.grammar)
                t.build_random_tree(['taction'])
                pop.append(t)
            except:
                continue

        return pop

    def error(self, tree: TreeRoot, tasks):
        # run func on input examples and calculate the correct predicted actions -> norm to 0 - 1
        preds_strict = []
        for task in tasks:
            try:
                correct = tree.test_program(task)
                preds_strict.append(int(correct))
            except ProgramNotCorrectlyTypedException as e:
                traceback.print_exc()
                print(e)
                assert False

        # (np.mean(preds) + np.mean(preds_strict))
        return len(tasks) - np.sum(preds_strict)

    def population_error(self, trees, tasks):
        solutions = defaultdict(list)

        errors = []
        for tree in trees:
            try:
                for task in tasks:
                    correct = tree.test_program(task)
                    if correct:
                        solutions[task].append(tree)
            except ProgramNotCorrectlyTypedException as e:
                traceback.print_exc()
                print(e)

        return solutions

    def fitness(self, individual, dataset):
        try:
            size = individual.size()
        except Exception as e:
            print(e)
            return 0

        if size > 500:
            return 0
        if self.bloat_weight > 0:
            err = 1 / (1 + self.error(individual, dataset) +
                       self.bloat_weight * size)
        else:
            err = 1 / (1 + self.error(individual, dataset))
        return err

    # select one individual using tournament selection
    def selection(self, population, fitnesses):
        # select tournament contenders
        tournament = [random.randint(0, len(population)-1)
                      for i in range(self.tournament_size)]
        tournament_fitnesses = [fitnesses[tournament[i]]
                                for i in range(self.tournament_size)]
        sel = population[tournament[tournament_fitnesses.index(
            max(tournament_fitnesses))]]
        tree = deepcopy(sel)
        return tree

    def train(self, parsed_data, start_iter=3, threshold=0.8, adapt_threshold=0.3, compress=False, calc_stats=False, epsilon_start=0.8, crossover_prob=0.2, mutate_prob=0.5, inrease_each_n_steps=0, max_n_invented=5):
        print('Generate population')
        population = self.init_population()

        tasks = makeTasks(parsed_data, randomChunkSize=False,
                          fixedChunkSize=start_iter)

        current_iter = start_iter
        pop_err = self.population_error(population, tasks)
        best_of_run = None
        best_of_run_error = 1e20
        best_population_acc = 0.0
        best_population = None
        best_population_err = None
        epsilon = epsilon_start
        fitnesses = [self.fitness(ind, tasks) for ind in population]
        p = None

        for gen in range(self.generations):
            start_time = time.time()
            nextgen_population = []
            while len(nextgen_population) < self.pop_size:
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)
                parent1.mutate(p=p, epsilon=epsilon, prob=mutate_prob)
                nextgen_population.append(deepcopy(parent1))
                if crossover_prob > 0.0:
                    parent1.crossover(parent2, prob=crossover_prob)
                    nextgen_population.append(parent1)
                    nextgen_population.append(parent2)

            population = nextgen_population
            fitnesses = [self.fitness(ind, tasks) for ind in population]
            errors = [self.error(ind, tasks) for ind in population]
            solutions = self.population_error(population, tasks)
            pop_err = len(solutions.keys()), len(tasks)
            population_acc = pop_err[0]/pop_err[1]
            epsilon -= 0.01

            # for best program
            if min(errors) < best_of_run_error:
                best_of_run_error = min(errors)
                best_of_run = deepcopy(population[errors.index(min(errors))])

                if self.output_path:
                    best_of_run.draw_tree(f"{self.output_path}/curr_{current_iter}-gen_{gen}",
                                          f"size: {current_iter} gen: {gen} error: {round(best_of_run_error,3)} population acc: {population_acc}", show=False)

            # for best population
            if population_acc > best_population_acc:
                best_population_err = pop_err
                best_population_acc = best_population_err[0] / \
                    best_population_err[1]
                best_population = deepcopy(population)

            avg_size = np.mean([ind.size() for ind in population])
            if len(solutions.values()) == 0:
                programs = []
            else:
                programs = list(
                    set(reduce(lambda a, b: a+b, solutions.values())))

            curr_var = np.var(
                list(self.calc_distribution(population).values()))
            wandb_log = {"best_of_run_error": best_of_run_error, "population_error":
                         pop_err[0]/pop_err[1], "seq_length": current_iter, "threshold": threshold, "variance": curr_var,
                         "avg_size": avg_size, "good programs": len(programs), "runtime": time.time() - start_time}

            # increase seq length
            if inrease_each_n_steps > 0 and ((gen + 1) % inrease_each_n_steps == 0 or pop_err[0]/pop_err[1] > 0.95):
                if len(programs) == 0:
                    print('No programs found.. stop..')
                    return GPResult(population, fitnesses, self.grammar)
                # log best error tasks and reload population
                population = best_population
                if self.log_results:
                    log_solved_tasks(
                        {"tasks": len(tasks), "hits": best_population_err[0]}, current_iter, gen)
                else:
                    print(
                        f"Solved { best_population_err[0] } of {len(tasks)} tasks for sequence length {current_iter}")
                best_population_acc = 0.0
                best_of_run_error = 1e20
                best_population_err = None
                current_iter += 1
                tasks = makeTasks(
                    parsed_data, randomChunkSize=False, fixedChunkSize=current_iter)

                if compress and len(self.grammar.invented) < max_n_invented:
                    start_time = time.time()
                    population = self.compress(tasks, programs, gen)
                    fitnesses = [self.fitness(ind, tasks)
                                 for ind in population]
                    errors = [self.error(ind, tasks) for ind in population]
                    wandb_log["Lib learning time"] = time.time() - start_time
                if calc_stats:
                    start_time = time.time()
                    epsilon = epsilon_start
                    p = self.calc_distribution(population)
                    wandb_log["calc_stats time"] = time.time() - start_time

            elif pop_err[0]/pop_err[1] >= threshold:
                if self.log_results:
                    log_solved_tasks(
                        {"tasks": len(tasks), "hits": pop_err[0]}, current_iter, gen)
                else:
                    print(
                        f"Solved { pop_err[0] } of {len(tasks)} tasks for sequence length {current_iter}")
                best_of_run_error = 1e20
                current_iter += 1
                tasks = makeTasks(
                    parsed_data, randomChunkSize=False, fixedChunkSize=current_iter)

                # we only run if less than  max_n_invented functions
                if compress and len(self.grammar.invented) < max_n_invented:
                    start_time = time.time()
                    population = self.compress(tasks, programs, gen)
                    fitnesses = [self.fitness(ind, tasks)
                                 for ind in population]
                    errors = [self.error(ind, tasks) for ind in population]
                    wandb_log["Lib learning time"] = time.time() - start_time
                if calc_stats:
                    start_time = time.time()
                    epsilon = epsilon_start
                    p = self.calc_distribution(population)
                    wandb_log["calc_stats time"] = time.time() - start_time
                if adapt_threshold > 0.0 and threshold > adapt_threshold:
                    threshold -= adapt_threshold
            else:
                if compress:
                    wandb_log["Lib learning time"] = 0.0
                if calc_stats:
                    wandb_log["calc_stats time"] = 0.0
            if self.log_results:
                wandb.log(wandb_log, step=gen)
            else:
                print(wandb_log)
        return GPResult(population, fitnesses, self.grammar)

    def calc_distribution(self, population):
        last_distribution = Counter()
        for p in population:
            last_distribution += Counter(p.distribution())

        all_primitives_count = sum(last_distribution.values())
        p = {k: p/all_primitives_count for k, p in last_distribution.items()}
        # print(p)
        return p


class GPStitch(GPAlgorithm):
    def compress(self, tasks, trees, gen):
        programs = list(set([t.get_program() for t in trees]))
        print(f'send {len(programs)} to the compressor')
        curr_i = len(self.grammar.invented)
        res = stitch_core.compress(
            programs, iterations=1, max_arity=3, previous_abstractions=curr_i)

        all_rewritten = res.rewritten
        for a in res.abstractions:
            try:
                added = self.grammar.add_invented(
                    f'fn_{curr_i}', a.body, a.arity)
                if added:
                    print(
                        'added', a, self.grammar.invented[-1].parameters(), self.grammar.invented[-1].returns())
                    curr_i += 1
                    if self.log_results:
                        t = wandb.Table(
                            columns=["body", "tp"], data=[[inv.body, str(inv.tp)] for inv in self.grammar.invented])
                        if t is not None:
                            wandb.log({'Invented Functions': t}, step=gen)
                    else:
                        print(
                            f"Invented functions: {[[inv.body, str(inv.tp)] for inv in self.grammar.invented]}")
            except:
                all_rewritten = [r for r in all_rewritten if a.body not in r]
                print('Error for adding invented:', a)
                continue

        # Remove all not correctly rewritten programs, can happen since Stitch has no type concept currently.
        rewritten_pop = []
        for r in all_rewritten:
            try:
                t = TreeRoot.from_program(r, self.grammar)
                rewritten_pop.append(t)
            except:
                traceback.print_exc()
                continue
        print(f'population after rewritten {len(rewritten_pop)}')

        # Test if progams also execute on a task, if not discard them
        population = []
        for p in rewritten_pop:
            try:
                p.test_program(tasks[0])
                population.append(p)
            except:
                traceback.print_exc()
                continue
        print(f'population after testing {len(population)}')

        # Afterwards fill up the missing population with random programs
        while len(population) < self.pop_size:
            try:
                t = TreeRoot(self.grammar)
                t.build_random_tree(['taction'])
                population.append(t)
            except:
                continue
        print(f'population after random init {len(population)}')
        return population
