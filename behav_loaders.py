import numpy
import os

from tqdm import tqdm

from loaders_utils import collect_info, reorganize_sims, write_trials
from utf_utils import transform_german_word

def read_italian_anew(args):
    ### lexical decition times
    sims = dict()
    test_vocab = set()
    for task in [
                  'lexical-decision', 
                  'word-naming', 
                  ]:
        full_case = 'it_anew-{}_{}#all-words'.format(task, args.stat_approach)
        sims[full_case] = dict()
        with open(os.path.join('data', 'behav', 'it_anew', 'it_anew-{}.tsv'.format(task))) as i:
            for l_i, l in enumerate(i):
                line = l.replace(',', '.').split('\t')
                #print(line)
                if l_i == 0:
                    header = [w.strip() for w in line]
                assert len(line) == len(header)
                if task == 'lexical-decision':
                    if line[header.index('Stimulus_Type')] != 'Word':
                        continue
                if line[header.index('Accuracy')] != '1':
                    continue
                word = line[header.index('Ita_Word')].lower()
                if '_' in word:
                    continue
                sub = line[header.index('Subject')]
                ### compressing...
                sub = 'all'
                if sub not in sims[full_case].keys():
                    sims[full_case][sub] = dict()
                #test_vocab = test_vocab.union(set([word, word.capitalize()]))
                test_vocab = test_vocab.union(set([word]))
                try:
                    word_rt = float(line[header.index('RTs')])
                except ValueError:
                    print([sub, word])
                    continue
                try:
                    sims[full_case][sub][word].append(word_rt)
                except KeyError:
                    sims[full_case][sub][word] = [word_rt]
    final_sims = dict()
    triples = set()
    with tqdm() as cntr:
        for case, case_r in sims.items():
            final_sims[case] = dict()
            for sub, measures in case_r.items():
                final_sims[case][sub] = list()
                for k_one_i, k_one in enumerate(sorted(measures.keys())):
                    for k_two_i, k_two in enumerate(sorted(measures.keys())):
                        if k_two_i <= k_one_i:
                            continue
                        key = tuple(sorted([k_one, k_two]))
                        final_sims[case][sub].append((key, abs(numpy.average(measures[k_one])-numpy.average(measures[k_two]))))
                        ### order tends to matter...
                        triples.add((k_one, k_two))
                        triples.add((k_two, k_one))
                        cntr.update(1)
    write_trials(args, triples)
    return final_sims, test_vocab

def read_german_behav(args):
    ### lexical decition times
    sims = {'de_word-naming_{}'.format(args.stat_approach) : {'all' : list()}, 'de_lexical-decision_{}'.format(args.stat_approach) : {'all' : list()}}
    test_vocab = set()
    triples = set()
    for case in sims.keys(): 
        short_case = case.split('_')[1]
        measures = dict()
        with open(os.path.join('data', 'behav', 'DeveL', 'devel_{}_de.tsv'.format(short_case))) as i:
            for l_i, l in enumerate(i):
                line = l.replace(',', '.').strip().split('\t')
                if l_i == 0:
                    header = [w for w in line]
                    marker = 'rt' if 'lex' in case else 'on'
                    ya = header.index('{}.ya.m'.format(marker))
                    oa = header.index('{}.oa.m'.format(marker))
                    continue
                if len(line) != len(header):
                    print(line)
                    continue
                word = line[0]
                #versions = transform_german_word(word)
                versions = [word]
                test_vocab = test_vocab.union(versions)
                word_rt = float(float(line[ya])+float(line[oa]))/2
                measures[word] = word_rt
        for k_one_i, k_one in enumerate(sorted(measures.keys())):
            for k_two_i, k_two in enumerate(sorted(measures.keys())):
                if k_two_i <= k_one_i:
                    continue
                key = tuple(sorted([k_one, k_two]))
                sims[case]['all'].append((key, abs(measures[k_one]-measures[k_two])))
                ### order tends to matter...
                triples.add((k_one, k_two))
                triples.add((k_two, k_one))
    write_trials(args, triples)
    return sims, test_vocab
