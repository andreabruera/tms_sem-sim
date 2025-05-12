import numpy
import os

def aggregate_subjects(sims):
    copy = {k : {'all' : dict()} for k in sims.keys()}
    for k, v in sims.items():
        for _, lst in v.items():
            for ws, sim in lst:
                if ws not in copy[k]['all'].keys():
                    copy[k]['all'][ws] = [sim]
                else:
                    copy[k]['all'][ws].append(sim)
    copy = {k : {_ : [(ws, numpy.average(sims)) for ws, sims in v.items()] for _, v in __.items()} for k, __ in copy.items()}
    return copy

def write_trials(args, trials_set):
    folder = os.path.join('trials', args.lang)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, '{}#trials.tsv'.format(args.dataset)), 'w') as o:
        for t in trials_set:
            o.write('\t'.join(t))
            o.write('\n')

def reorganize_sims(sims):
    full_sims = dict()
    for n, n_data in sims.items():
        full_sims[n] = dict()
        counter = 0
        for s, ws, rt in n_data:
            if s not in full_sims[n].keys():
                full_sims[n][s] = list()
            full_sims[n][s].append((ws, rt))
    return full_sims

def collect_info(full_sims):
    labels = set(full_sims.keys())
    subjects = set([s for subs in full_sims.values() for s in subs.keys()])
    trials = set([len(set([ws[0] for ws in s])) for subs in full_sims.values() for s in subs.values()])
    print('labels: ')
    print(labels)
    print('subjects: ')
    print(subjects)
    print('trials: ')
    print(trials)
