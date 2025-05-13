import argparse
import fasttext
import numpy
import os
import pickle
import random
import scipy
import sklearn

from scipy import spatial
from sklearn import linear_model, metrics
from tqdm import tqdm

from behav_loaders import read_italian_anew, read_german_behav
from tms_loaders import read_it_social_quantity_tms, read_it_distr_learn_tms, read_de_pmtg_production_tms, read_phil, read_de_sem_phon_tms
from utf_utils import transform_german_word, transform_italian_word

def load_dataset(args):
    if 'de_behav' in args.dataset:
        data, vocab = read_german_behav(args)
    if 'it_anew' in args.dataset:
        data, vocab = read_italian_anew(args)
    if 'pmtg-prod' in args.dataset:
        data, vocab = read_de_pmtg_production_tms(args)
    if 'sem-phon' in args.dataset:
        data, vocab = read_de_sem_phon_tms(args)
    if 'distr-learn' in args.dataset:
        data, vocab = read_it_distr_learn_tms(args)
    if 'sound-action' in args.dataset:
        data, vocab = read_phil(args)
    if 'social-quantity' in args.dataset:
        data, vocab = read_it_social_quantity_tms(args)
    return vocab, data

def test_precomputed_model(args, model_name, datasets, datasets_vocab, freqs):
    model_sims, model_vocab = read_sims(args, model_name)
    missing_words = list(set(datasets_vocab).difference(set(model_vocab)))
    missing_trials = [c[0] for k, v in datasets.items() for k_two, v_two in v.items() for c in v_two if False in [True if w not in missing_words else False for w in c[0]]]
    #import pdb; pdb.set_trace()
    print('\nmissing words:\n{}'.format(missing_words))
    print('\nmissing trials (across all subjects and conditions): {}\n'.format(len(missing_trials)))
    datasets = {k : {k_two : [c for c in v_two if False not in [True if w not in missing_words else False for w in c[0]]] for k_two, v_two in v.items()} for k, v in datasets.items()}
    test_model(args, model_name, model_sims, datasets, freqs)

def read_sims(args, model_name):
    sims = dict()
    model_vocab = list()
    folder = os.path.join('dissimilarities', args.lang, model_name,)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, '{}#dissimilarities.tsv'.format(args.dataset))) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            words = tuple(line[0].split(','))
            model_vocab.extend(words)
            sim = float(line[1])
            sims[words] = sim
    return sims, model_vocab

def test_model(args, case, sims, datasets, freqs):
    if args.stat_approach != 'simple':
        datasets = bootstrapper(args, datasets, freqs, sims)
    else:
        datasets = {k : [v] for k, v in datasets.items()}
    ### time shortcut 1:
    ### we pre-load all sim tests
    print('now pre-checking words...')
    all_sims_data = dict()
    to_be_computed = set()
    with tqdm() as counter:
        for dataset_name, dataset in datasets.items():
            sub_corr = dict()
            corr = list()
            ### bootstrapping/iterations should be hard-coded now...
            for iter_dataset in tqdm(dataset):
                iter_corrs = list()
                #print(len(iter_dataset.keys()))
                for s, s_data in iter_dataset.items():
                    curr_corr = compute_corr(args, sims, s_data)
                    if curr_corr == None:
                        print('error with {}'.format([args.lang, case, dataset_name]))
                        continue
                    iter_corrs.append(curr_corr)
                    try:
                        sub_corr[s].append(curr_corr)
                    except KeyError:
                        sub_corr[s] = [curr_corr]
                if args.approach in ['rsa', 'correlation']:
                    if args.stat_approach == 'simple':
                        corr.extend(iter_corrs)
                    else:
                        iter_corr = numpy.average(iter_corrs)
                        corr.append(iter_corr)

            print('\n')
            print('{} model'.format(case))
            print('correlation with {} dataset:'.format(dataset_name))
            print(numpy.nanmean(corr))
            #if len(missing_words) > 0:
            #    print('missing words: {}'.format(missing_words))
            write_res(args, case, dataset_name, corr,)
            write_sub_res(args, case, dataset_name, sub_corr,)

def compute_corr(args, model_sims, test_sims):
    assert len(test_sims) > 0
    real = list()
    pred = list()
    for ws, v in test_sims:
        real.append(v)
        if type(ws[0]) == tuple:
            pred_ws = list()
            for w in ws[0]:
                try:
                    pred_ws.append(model_sims[(w, ws[1])])
                except KeyError:
                    continue
            assert len(pred_ws) > 0
            pred.append(numpy.average(pred_ws))
        else:
            pred.append(model_sims[ws])
    corr = scipy.stats.spearmanr(real, pred).statistic
    return corr

def rt(args, case, model, vocab, datasets, present_words, trans_from_en):
    if args.stat_approach != 'simple':
        datasets = bootstrapper(args, datasets, freqs, sims)
    else:
        datasets = {k : [v] for k, v in datasets.items()}
    for dataset_name, dataset in datasets.items():
        corr = list()
        ### bootstrapping/iterations should be hard-coded now...
        for iter_dataset in tqdm(dataset):
            iter_corrs = list()
            for s, s_data in iter_dataset.items():
                curr_corr = numpy.average([v[1] for v in s_data])
                if curr_corr == None:
                    print('error with {}'.format([args.lang, case, dataset_name]))
                    continue
                iter_corrs.append(curr_corr)
            if args.stat_approach == 'simple':
                corr.extend(iter_corrs)
            else:
                iter_corr = numpy.average(iter_corrs)
                corr.append(iter_corr)

        write_res(args, case, dataset_name, corr, )

def bootstrapper(args, full_data, freqs, sims):
    ### labels
    labels = list(full_data.keys())
    all_subjects = {k : list(v.keys()) for k, v in full_data.items()}
    all_trials = {k : [len(vs) for vs in v.values()] for k, v in full_data.items()}
    tms_datasets = [
                   'de_pmtg-prod',
                   'de_sem-phon', 
                   'it_distr-learn',
                   'de_sound-action',
                   'it_social-quantity',
                   ]
    behav_datasets = [
                'de_behav',
                'it_anew',
            ]
    ### fixed number of subjects and trials
    n_iter_sub = 20
    n_iter_trials = 25
    for k, v in all_subjects.items():
        if len(v) < n_iter_sub:
            print('max number of subjects for {}: {}'.format(k, len(v)))
    for k, vs in all_trials.items():
        smaller_ns = [_ for _ in vs if _ <n_iter_trials]
        if len(smaller_ns) > 0:
            print('insufficient number of trials for {} of subjects: {}'.format(len(smaller_ns)/len(vs), smaller_ns))
    ### here we create 10000
    boot_data = {l : list() for l in labels}
    if args.stat_approach == 'residualize':
        print('residualizing...')
    for _ in tqdm(range(1000)):
        iter_subs = {l : sorted(random.sample(subjects, k=min(n_iter_sub, len(subjects)))) for l, subjects in all_subjects.items()}
        iter_data_idxs = {l : 
                           {s : random.sample(
                                             range(len(full_data[l][s])), 
                                             k=min(n_iter_trials, len(full_data[l][s])),
                                             ) for s in iter_subs[l]}
                                             for l in labels}
        iter_data = {l : {s : [(full_data[l][s][k][0], full_data[l][s][k][1]) for k in iter_data_idxs[l][s]] for s in iter_subs[l]} for l in labels}
        ### residualization
        if args.stat_approach == 'residualize':
            new_iter_data = {k : {v : list() for v in _.keys()} for k, _ in iter_data.items()}
            struct_train_data = {l : {s : [(full_data[l][s][k][0], full_data[l][s][k][1]) for k in range(len(full_data[l][s])) if k not in iter_data_idxs[l][s]] for s in iter_subs[l]} for l in labels}
            flat_train_data = [
                               (
                                l, 
                                s, 
                                k, 
                                rt, 
                                ) for l, l_res in struct_train_data.items() \
                                        for s, s_res in l_res.items() \
                                        for k, rt in s_res
                                        ]
            flat_test_data = [
                              (
                               l, 
                               s, 
                               k, 
                               rt, 
                               ) for l, l_res in iter_data.items() \
                                       for s, s_res in l_res.items() \
                                       for k, rt in s_res]
            model = sklearn.linear_model.LinearRegression()
            if 'sound' not in args.dataset:
                ### adding missing cases
                for ts in [flat_train_data, flat_test_data]:
                    for t in ts:
                        for ex in t[2]:
                            try:
                                freqs[ex]
                            except KeyError:
                                freqs[ex] = 1.
                model.fit(
                          ### input
                          [
                           [
                            ### word(s) length
                            len(t[2][0]),
                            len(t[2][1]),
                            ### word(s) frequency
                            numpy.log10(freqs[t[2][0]]), 
                            numpy.log10(freqs[t[2][1]]), 
                            ] for t in flat_train_data],
                          ###target
                          [
                           [
                            ### rt
                            t[3], 
                            ] for t in flat_train_data],
                          )
                preds = model.predict(
                                      [[
                                        ### word(s) length
                                        len(t[2][0]),
                                        len(t[2][1]),
                                        ### word(s) frequency
                                        numpy.log10(freqs[t[2][0]]), 
                                        numpy.log10(freqs[t[2][1]]), 
                                        ] for t in flat_test_data]
                                      )
            else:
                ### considering only one word
                idx = 1
                model.fit(
                          ### input
                          [
                           [
                            ### word length
                            len(t[2][idx]),
                            ### word frequency
                            numpy.log10(freqs[t[2][idx]]),
                            ] for t in flat_train_data],
                          ###target
                          [
                           [
                            ### rt
                            t[3],
                            ] for t in flat_train_data],
                          )
                preds = model.predict(
                                      [[
                                        ### word length
                                        len(t[2][idx]),
                                        ### word frequency
                                        numpy.log10(freqs[t[2][idx]]),
                                        ] for t in flat_test_data]
                                      )
            residuals = [
                         (
                          real[0], 
                          real[1], 
                          real[2], 
                          real[3]-pred[0], 
                                  ) for real, pred in zip(flat_test_data, preds)]
            for l, s, k, r in residuals:
                try:
                    new_iter_data[l][s].append((k, r))
                except KeyError:
                    new_iter_data[l][s] = [(k, r)]
            del iter_data
            iter_data = {k : v for k, v in new_iter_data.items()}
        for l, l_data in iter_data.items():
            boot_data[l].append(l_data)
    return boot_data

def check_args(args):
    ### checking language if in first part of the name
    if '_' in args.dataset:
        assert args.dataset.split('_')[0] == args.lang
    if 'behav' in args.dataset:
        if args.stat_approach == 'residualize':
            raise RuntimeError()
    ### for sound action there is only one word for length/frequency
    if 'sound' in args.dataset:
        if 'visual-' in args.model or 'produced-' in args.model or 'one-' in args.model or 'two-' in args.model:
            raise RuntimeError()
    ### for non-production dataset, there is no distinction between visual and produced words
    if 'visual' in args.model or 'produced' in args.model:
        if 'pmtg' not in args.dataset and 'sem' not in args.dataset:
            raise RuntimeError()
        if 'one-' in args.model or 'two-' in args.model:
            if 'pmtg' in args.dataset or 'sem' in args.dataset or 'sound' in args.dataset or args.modality != 'tms':
                raise RuntimeError()

def write_sub_res(args, case, dataset_name, corr):
    corpus_fold = case.split('_')[1] if 'ppmi' in case else case
    details = '_'.join(case.split('_')[2:]) if 'ppmi' in case else case
    out_folder = os.path.join(
                              'test_sub_results',
                              args.approach,
                              args.stat_approach,
                              args.evaluation,
                              args.lang, 
                              corpus_fold, 
                              details,
                              )
    os.makedirs(out_folder, exist_ok=True)
    out_f = os.path.join(out_folder, '{}.tsv'.format(dataset_name))
    with open(out_f, 'w') as o:
        o.write('{}\t{}\t{}\t'.format(args.lang, case, dataset_name))
        for s, cs in corr.items():
            c = numpy.average(cs)
            o.write('{}\t'.format(c))
    print(out_f)

def write_res(args, case, dataset_name, corr, trust=True):
    corpus_fold = case.split('_')[1] if 'ppmi' in case else case
    details = '_'.join(case.split('_')[2:]) if 'ppmi' in case else case
    out_folder = os.path.join(
                              'test_results',
                              args.approach,
                              args.stat_approach,
                              args.evaluation,
                              args.lang, 
                              corpus_fold, 
                              details,
                              )
    os.makedirs(out_folder, exist_ok=True)
    out_f = os.path.join(out_folder, '{}.tsv'.format(dataset_name))
    with open(out_f, 'w') as o:
        o.write('{}\t{}\t{}\t'.format(args.lang, case, dataset_name))
        for c in corr:
            o.write('{}\t'.format(c))
    print(out_f)

def args(similarity=False):
    parser = argparse.ArgumentParser()
    corpora_choices = [
                       'visual-word-length', 
                       'one-word-length', 
                       'two-word-length', 
                       'produced-word-length', 
                       'overall-word-length', 
                       'visual-old20', 
                       'produced-old20',
                       'overall-old20', 
                       ]
    llms = [
         'gpt2-small',
         'gpt2-small-iso',
         'llama-3b',
         'llama-3b-iso',
         ]
    for llm in llms:
        if '3b' in llm:
            m = 28
        else:
            m = 12
        for l in range(m):
            corpora_choices.append('{}_layer-{}'.format(llm, l))
        corpora_choices.append('{}_surprisal'.format(llm))
    for corpus in [
                   'wac',
                   ]:
        corpora_choices.append('{}-ppmi-vecs'.format(corpus))
        for mode in [
                     'neg-raw-abs-prob',
                     'neg-log10-abs-prob',
                     'one-neg-log10-abs-prob',
                     'two-neg-log10-abs-prob',
                     'visual-neg-log10-abs-prob',
                     'produced-neg-log10-abs-prob',
                     'overall-neg-log10-abs-prob',
                     'neg-sym-raw-cond-prob',
                     'neg-fwd-raw-cond-prob',
                     'neg-sym-log10-cond-prob',
                     'surprisal',
                     ]:
            corpora_choices.append('{}-{}'.format(corpus, mode))
    parser.add_argument(
                        '--model',
                        choices=[
                                 'response_times',
                                 'fasttext',
                                 ] + corpora_choices,
                        required=True,
                        )
    parser.add_argument(
                        '--lang',
                        choices=[
                                 'de',
                                 'it',
                                 ],
                        required=True
                        )
    parser.add_argument(
                        '--dataset',
                        choices=[
                                ### behav
                                'de_behav',
                                'it_anew',
                                ### tms
                                'de_pmtg-prod',
                                'de_sem-phon',
                                'it_distr-learn',
                                'de_sound-action',
                                'it_social-quantity',
                                ],
                        required=True,
                        )
    parser.add_argument(
                     '--modality', 
                     choices=[
                              'behav', 
                              'tms', 
                              ],
                     required=True,
                     )
    if not similarity:
        parser.add_argument(
                            '--stat_approach',
                            choices=[
                                     'simple', 
                                     'bootstrap', 
                                     'residualize',
                                     ],
                            required=True,
                            )
        parser.add_argument(
                            '--approach',
                            choices=[
                                     'correlation', 
                                     ],
                            required=True,
                            )
        parser.add_argument(
                            '--evaluation',
                            choices=[
                                     'spearman', 
                                     ],
                            required=True,
                            )
    args = parser.parse_args()
    if similarity:
        args.stat_approach = ''
    check_args(args)

    return args
