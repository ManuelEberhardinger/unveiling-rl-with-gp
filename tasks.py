import pandas as pd 
from dsl import *

def all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def parseData(taskData, groupby='run_id', verbose=False):
    columns = ['process', 'obs', 'obs direction',
               'action', 'reward', 'done', 'run_id']

    df = pd.DataFrame(taskData, columns=columns)
    df = df.drop(['process'], axis=1)
    df.action = df.action.apply(lambda x: x[0])
    df.obs = df.obs.apply(lambda x: np.flip(x[0] * 10, 1))
    df.insert(0, 'run_id', df.pop('run_id'))

    group = df.groupby('run_id')

    groups_to_consider = []

    for key in group.groups.keys():
        g = group.get_group(key)
        if verbose:
            print(f'group {key}')
        if not g[g.reward > 0.0].count().all():
            if verbose:
                print(f'no reward..')
        else:
            reward = g[g.reward > 0.0].reward.iloc[0]
            if reward < 0.85:
                if verbose:
                    print(f'skip {key} because reward is to small')
                continue
            if verbose:
                print(f'needed {g.shape[0]} steps. Reward: {reward}')
            groups_to_consider.append(key)
    group = group.filter(lambda x: x.run_id.mean() in groups_to_consider)
    print(group.shape)
    return group.groupby(groupby)

class Task():
    def __init__(self, name, tp, examples):
        self.name = name
        self.tp = tp
        self.examples = examples


def makeTasks(data, rand_min=10, rand_max=50, randomChunkSize=True, fixedChunkSize=None):
    assert randomChunkSize or (not randomChunkSize and fixedChunkSize)
    keys = data.groups.keys()
    print('keys:', len(keys))
    tasks = []
    for key in keys:
        to_imitate = data.get_group(key)
        if randomChunkSize:
            chunkSize = random.randint(rand_min, rand_max)
        else:
            chunkSize = fixedChunkSize
        examples = []
        part = 0
        for _, row in to_imitate.iterrows():
            input_ex = (row.obs.astype(int).tolist(),
                        int(row['obs direction'],))
            output_ex = int(row.action)
            examples.append((input_ex, output_ex))

            if chunkSize > 0 and chunkSize <= len(examples):
                # we check that the chosen actions are not all the same
                # otherwise it is too easy to find a program if all actions/output examples are the same
                # this results in programs such as (lambda (lambda forward-action))
                all_chosen_actions = list(zip(*examples))[1]
                if not all_equal(all_chosen_actions):
                    tasks.append(Task(f'perfect maze {key} size {chunkSize} part {part}',
                                 [tmap, tinpdirection, taction], examples))
                    part += 1
                    # we reset examples and add new chunkSize taskss
                    examples = []
                    if randomChunkSize:
                        chunkSize = random.randint(rand_min, rand_max)
                    else:
                        chunkSize = fixedChunkSize

        if len(examples) > 3:
            all_chosen_actions = list(zip(*examples))[1]
            if not all_equal(all_chosen_actions):
                tasks.append(Task(f'perfect maze {key} size {chunkSize} part {part}',
                             [tmap, tinpdirection, taction], examples))


    print(f'Created {len(tasks)} tasks with {fixedChunkSize} chunk size')
    return tasks


def get_max_seq_len(group_df):
    max_len = 0

    for key, item in group_df:
        seq_len = group_df.get_group(key).shape[0]
        if seq_len > max_len:
            max_len = seq_len

    return max_len
