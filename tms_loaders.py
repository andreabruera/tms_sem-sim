import matplotlib
import numpy
import os

from matplotlib import pyplot

from loaders_utils import collect_info, reorganize_sims, write_trials
from utf_utils import transform_german_word, transform_italian_word

def read_de_pmtg_production_tms(args):
    print('\nPiai et al. - picture naming with interference\n')
    lines = list()
    missing = 0
    with open(os.path.join(
                           'data',
                           'tms',
                           'de_pmtg-production.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.split('\t')
            if l_i == 0:
                header = [w.strip() for w in line]
                continue
            line = [w.strip() for w in line]
            if line[header.index('accuracy')] == '0':
                missing += 1
                continue
            lines.append([w.strip() for w in line])
    print('missing words: {}'.format(missing))
    stims = set([l[header.index('stimulation')] for l in lines])
    conds = {
             #'u' : 'unrelated',
             #'r' : 'related',
             #'ur' : 'all-but-same',
             'urt' : 'all',
             }
    all_sims = dict()
    test_vocab = set()
    triples = set()
    for name, cond in conds.items():
        for stim in stims:
            #print(name)
            key = 'de_pmtg-production_{}#{}-{}'.format(args.stat_approach, cond, stim)
            current_cond = [l for l in lines if l[header.index('condition')].strip() in name and \
                                                l[header.index('stimulation')] == stim and \
                                                l[header.index('response')] not in ['0', 'NA'] and \
                                                l[header.index('rt')] not in ['0', 'NA']
                                                ]
            rts = [float(l[header.index('rt')]) for l in current_cond]
            ### ORIGINAL EXCLUSION CRITERIA
            to_remove = [l_i for l_i, l in enumerate(rts) if l>=3]
            print('removed according to original criteria {} trials'.format(len(to_remove)))
            current_cond = [l for l_i, l in enumerate(current_cond) if l_i not in to_remove]
            rts = [float(l[header.index('rt')]) for l in current_cond]
            log_rts = [numpy.log10(float(l[header.index('rt')])) for l in current_cond]
            #print(rts)
            subjects = [int(l[header.index('sbj')]) for l in current_cond]
            #vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('picture')].split('.')[0])]
            vocab_w_ones = [l[header.index('picture')].split('.')[0] for l in current_cond]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [l[header.index('distractor')] for l in current_cond]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            ### picture -> word
            #w_ones = [l[header.index('picture')].split('.')[0] for l in current_cond]
            #w_twos = [l[header.index('distractor')].strip() for l in current_cond]
            ### word -> picture
            w_ones = [l[header.index('distractor')].strip() for l in current_cond]
            w_twos = [l[header.index('picture')].split('.')[0] for l in current_cond]
            all_sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
            triples = triples.union(set([v[1] for v in all_sims[key]]))
    final_sims = reorganize_sims(all_sims)
    collect_info(final_sims)
    write_trials(args, triples)

    return final_sims, test_vocab

### Dataset #2: Klaus & Hartwigsen

def read_de_sem_phon_tms(args):
    print('\nKlaus & Hartwigsen - Semantic production\n')
    sims = dict()
    test_vocab = set()
    lines = list()
    na_lines = list()
    missing = 0
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_tms_sem-phon_ifg.tsv')
                           ) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if 'NA' in line:
                #print(line)
                if 'sem' in line:
                    na_lines.append(line)
                continue
            if l_i == 0:
                header = [w for w in line]
                continue
            #assert len(line)==len(header)
            if '' in line:
                continue
            if len(line) < len(header)-1:
                print('skipping line: {}'.format(line))
                continue
            ### removing trailing spaces
            line = [w.strip() for w in line]
            if line[header.index('ERR')] == '1':
                missing += 1
                continue
            lines.append(line)
    print('missing words: {}'.format(missing))
    print('sem trials containing a NA: {}'.format(len(na_lines)))
    ###
    conditions = set([l[header.index('stim')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    #print(tasks)
    full_sims = dict()
    triples = set()
    #for c, name in conditions.items():
    for t in tasks:
        if 'sem' not in t:
            continue
        for c in conditions:
            name = 'de_sem-phon_{}#{}-{}'.format(args.stat_approach, t, c)
            #print(name)
            ###One participant was replaced due to an overall mean error rate of 41.8% - sub 3
            #current_cond = [l for l in lines if l[header.index('stim')] in c and int(l[header.index('subj')])!=3]
            current_cond = [l for l in lines if l[header.index('stim')] in name and l[header.index('task')] in t and int(l[header.index('subj')])!=3]
                    #and l[header.index('utterance')]!='NA']
            tasks = [l[header.index('task')] for l in current_cond]
            assert len(set(tasks)) == 1
            subjects = [int(l[header.index('subj')]) for l in current_cond]
            assert len(set(subjects)) == 24
            #print(subjects)
            rts = [float(l[header.index('RT')]) for l in current_cond]
            to_remove = list()
            ### ORIGINAL EXCLUSION CRITERIA
            ### removing trials more/less than 3 SDs
            for sub in set(subjects):
                avg = numpy.average([rts[_] for _, s in enumerate(subjects) if s==sub])
                std = numpy.std([rts[_] for _, s in enumerate(subjects) if s==sub])
                sub_remove = [l_i for l_i, l in enumerate(rts) if (l>=avg+(3*std) or l<=avg-(3*std)) and subjects[l_i]==sub]
                to_remove.extend(sub_remove)
            print('removed according to original criteria {} trials'.format(len(to_remove)))
            current_cond = [l for l_i, l in enumerate(current_cond) if l_i not in to_remove]
            subjects = [int(l[header.index('subj')]) for l in current_cond]
            rts = [float(l[header.index('RT')]) for l in current_cond]

            log_rts = [numpy.log10(float(l[header.index('RT')])) for l in current_cond]
            #vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('item')].split('.')[0])]
            vocab_w_ones = [l[header.index('item')].split('.')[0] for l in current_cond]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            #vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('utterance')])]
            vocab_w_twos = [l[header.index('utterance')] for l in current_cond]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            ### image -> utterance
            w_ones = [l[header.index('item')].split('.')[0] for l in current_cond]
            w_twos = [l[header.index('utterance')] for l in current_cond]
            sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
            triples = triples.union(set([v[1] for v in sims[name]]))
    full_sims = reorganize_sims(sims)
    collect_info(full_sims)
    write_trials(args, triples)
    #print(triples)

    return full_sims, test_vocab

### Dataset #3: Gatti et al.

def read_it_distr_learn_tms(args):
    print('\nGatti et al. - Semantic relatedness judgment\n')
    lines = list()
    with open(os.path.join(
                           'data',
                           'tms',
                           'italian_tms_cereb.tsv')) as i:
        missing = 0
        for l_i, l in enumerate(i):
            line = [w.strip() for w in l.strip().split('\t')]
            if l_i == 0:
                header = [w for w in line]
                continue
            if line[header.index('accuracy')] == '0':
                #print(line)
                missing += 1
                continue
            lines.append(line)
    print('missing words: {}'.format(missing))
    conds = set([l[header.index('condition')] for l in lines])
    all_sims = dict()
    all_full_sims = dict()
    related_sims = dict()
    related_full_sims = dict()
    unrelated_sims = dict()
    unrelated_full_sims = dict()
    test_vocab = set()
    triples = set()
    for name in conds:
        for m_i, marker in enumerate(['1', '0', 'all']):
            if m_i < 2:
                current_cond = [l for l in lines if l[header.index('condition')]==name and l[header.index('Meaningful')]==marker]
            else:
                current_cond = [l for l in lines if l[header.index('condition')]==name]
            log_rts = [numpy.log10(float(l[header.index('RTs')].replace(',', '.'))) for l in current_cond]
            rts = [float(l[header.index('RTs')].replace(',', '.')) for l in current_cond]
            subjects = [int(l[header.index('Subject')]) for l in current_cond]
            ### noun -> adj
            w_ones = [l[header.index('noun')].lower() for l in current_cond]
            w_twos = [l[header.index('adj')].lower() for l in current_cond]
            #vocab_w_ones = [w for ws in w_ones for w in [ws, ws.capitalize()]]
            vocab_w_ones = [w for w in w_ones]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            #vocab_w_twos = [w for ws in w_twos for w in [ws, ws.capitalize()]]
            vocab_w_twos = [w for w in w_twos]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            if m_i == 0:
                related_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                triples = triples.union(set([v[1] for v in related_sims[name]]))
            elif m_i == 1:
                unrelated_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                triples = triples.union(set([v[1] for v in unrelated_sims[name]]))
            elif m_i == 2:
                all_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                triples = triples.union(set([v[1] for v in all_sims[name]]))
    related_full_sims = reorganize_sims(related_sims)
    unrelated_full_sims = reorganize_sims(unrelated_sims)
    all_full_sims = reorganize_sims(all_sims)

    final_sims = {'it_distr-learn_{}#all-trials_{}'.format(args.stat_approach, k) : v for k, v in all_full_sims.items()}
    if args.stat_approach not in ['residualize', 'bootstrap']:
        for k, v in related_full_sims.items():
            final_sims['it_distr-learn_{}#related-trials_{}'.format(args.stat_approach, k)] = v
        for k, v in unrelated_full_sims.items():
            final_sims['it_distr-learn_{}#unrelated-trials_{}'.format(args.stat_approach, k)] = v
    collect_info(final_sims)
    write_trials(args, triples)
    
    return final_sims, test_vocab

### Dataset #4: Kuhnke et al.

def read_phil(args):
    print('\nKuhnke et al. - Semantic feature judgment\n')
    ### reading dataset
    lines = list()
    proto = dict()
    triples = set()
    tasks = set()
    conditions = set()
    with open(os.path.join(
                           'data', 
                           'tms', 
                          'de_tms_sound-action_dataset.tsv',
                           )
                           ) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            ### keeping only correct trials
            acc = int(line[header.index('accuracy')])
            if acc == 0:
                continue
            w = line[header.index('word')]
            cat = line[header.index('category')]
            cond = line[header.index('condition')]
            conditions.add(cond)
            task = line[header.index('task')]
            tasks.add(task)
            try:
                proto[cat].add(w)
            except KeyError:
                proto[cat] = set([w])
            lines.append(line)
    proto_modes = [
             #'both-pos-all',
             'both-together-pos-all',
             #'both-neg-all',
             'matched-excl-all',
             #'matched-incl-all',
             #'opposite-excl-all',
             ]
    sims = dict()
    test_vocab = set()
    ### everything together
    for proto_mode in proto_modes:
        for c in conditions:
            for t in tasks:
                key = '{}_{}#{}_{}_{}'.format(args.dataset, args.stat_approach, t, c, proto_mode)
                ### separate tasks
                current_cond = [l for l in lines if l[header.index('task')]==t and l[header.index('condition')]==c]
                subjects = [int(l[header.index('subject')]) for l in current_cond]
                log_rts = [float(l[header.index('log10_rt')]) for l in current_cond]
                #rts = [float(l[header.index('raw_rt')]) for l in current_cond]
                if proto_mode == 'matched-excl-all':
                    if t == 'A':
                        proto_key = ['_A']
                    elif t == 'S':
                        proto_key = ['S_']
                elif proto_mode == 'opposite-excl-all':
                    if t == 'A':
                        proto_key = ['S_']
                    elif t == 'S':
                        proto_key = ['_A']
                if proto_mode == 'matched-incl-all':
                    if t == 'A':
                        proto_key = ['_A', 'SA']
                    elif t == 'S':
                        proto_key = ['S_', 'SA']
                elif proto_mode == 'both-together-pos-all':
                    proto_key = ['SA']
                elif proto_mode == 'both-separate-pos-all':
                    proto_key = ['_A', 'S_']
                elif proto_mode == 'both-neg-all':
                    proto_key = ['__',]
                w_ones = [tuple([w for k in proto_key for w in proto[k]]) for l in current_cond]
                all_w_ones = [tuple([w]) for ws in w_ones for w in ws]
                test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
                vocab_w_twos = [l[header.index('word')] for l in current_cond]
                w_twos = [l[header.index('word')] for l in current_cond]
                test_vocab = test_vocab.union(set(vocab_w_twos))
                sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                for p_ones, t in zip(w_ones, w_twos):
                    for p_one in p_ones:
                        triples.add((p_one, t))
    full_sims = reorganize_sims(sims)
    collect_info(full_sims)
    write_trials(args, triples)

    return full_sims, test_vocab

### dataset 5: Catricalà et al.

def read_it_social_quantity_tms(args):
    print('\nCatricalà et al. - Semantic priming')
    lines = list()
    it_mapper = dict()
    prototypes = {'s' : set(), 'q' : set()}
    exclusions = dict()
    accuracies = dict()
    with open(os.path.join(
                           'data',
                           'tms',
                           'it_tms_social-quant.tsv')) as i:
        missing = 0
        for l_i, l in enumerate(i):
            line = [w.strip() for w in l.strip().split('\t')]
            if l_i == 0:
                header = [w for w in line]
                continue
            if line[header.index('accuracy')] == '0':
                #print(line)
                missing += 1
                continue
            sub = line[header.index('subject')]
            if sub not in exclusions.keys():
                exclusions[sub] = dict()
            acc = int(line[header.index('accuracy')])
            if sub not in accuracies.keys():
                accuracies[sub] = list()
            accuracies[sub].append(acc)
            rt = line[header.index('response_time')]
            cond = line[header.index('condition')]
            cat = line[header.index('target_category')]
            if cat not in exclusions[sub].keys():
                exclusions[sub][cat] = {'cong' : list(), 'incong' : list()}
            w = line[header.index('target')]
            p = line[header.index('prime')]
            if cat[0] == p[0]:
                ### congruent
                it_mapper[cat] = p
                if len(cond) == 2:
                    exclusions[sub][cat]['cong'].append(float(rt))
            else:
                if len(cond) == 2:
                    exclusions[sub][cat]['incong'].append(float(rt))
            prototypes[cat[0]].add(w)
            lines.append(line)
    sub_to_remove = [k for k, v in accuracies.items() if numpy.average(v)<=0.5]
    #import pdb; pdb.set_trace()
    excluded = dict()
    for s, s_data in exclusions.items():
        for c, c_data in s_data.items():
            if len(c_data['cong']) == 0 and len(c_data['incong']) == 0:
                continue
            if numpy.average(c_data['cong']) > numpy.average(c_data['incong']):
                try:
                    excluded[c].append(s)
                except KeyError:
                    excluded[c] = [s]
    ### removing incomplete subject...
    for k in excluded.keys():
        excluded[k].append('23')
    print('excluded subjects following original paper: {}'.format(excluded))
    prototypes = {k[0] : tuple([w for w in v]) for k, v in prototypes.items()}
    print('missing words: {}'.format(missing))
    conds = set([l[header.index('condition')] for l in lines])
    print(conds)
    all_sims = dict()
    test_vocab = set()
    triples = set()
    for cong in [
                 #'congruent', 'incongruent', 
                 'all',
                 ]:
        if cong == 'congruent':
            cong_lines = [l for l in lines if l[header.index('target_category')][0]==l[header.index('prime')][0]]
        elif cong == 'incongruent':
            cong_lines = [l for l in lines if l[header.index('target_category')][0]!=l[header.index('prime')][0]]
        elif cong == 'all':
            cong_lines = [l for l in lines]
        for name in conds:
            for marker in [
                           'social', 
                           'quantity', 
                           #'all',
                           ]:
                test_vocab.add(marker)
                primes = ['prime-cat']
                for prime in primes:
                    ### in congruent cases primes and targets are the same
                    if 'target' in prime and cong == 'congruent':
                        continue
                    ### removing excluded subjects
                    if marker != 'all':
                        impossible_lines = [l for l in cong_lines if l[header.index('subject')] in excluded[marker[0]]]
                        #print('removed {} lines'.format(len(impossible_lines)))
                        assert len(impossible_lines)>0
                        possible_lines = [l for l in cong_lines if l[header.index('subject')] not in excluded[marker[0]]]
                    else:
                        possible_lines = [l for l in cong_lines]
                    '''
                    ### not excluding subjects
                    possible_lines = [l for l in cong_lines]
                    '''
                    key = 'it_social-quantity_{}#{}-{}-trials-{}_{}'.format(args.stat_approach, marker, cong, prime, name)
                    if marker != 'all':
                        current_cond = [l for l in possible_lines if l[header.index('condition')]==name and l[header.index('target_category')]==marker[0]]
                    else:
                        current_cond = [l for l in possible_lines if l[header.index('condition')]==name]
                    rts = [float(l[header.index('response_time')].replace(',', '.')) for l in current_cond]
                    subjects = [int(l[header.index('subject')]) for l in current_cond]
                    to_remove = list()
                    ### ORIGINAL EXCLUSION CRITERIA
                    ### removing trials more/less than 3 SDs
                    for sub in set(subjects):
                        avg = numpy.average([rts[_] for _, s in enumerate(subjects) if s==sub])
                        std = numpy.std([rts[_] for _, s in enumerate(subjects) if s==sub])
                        sub_remove = [l_i for l_i, l in enumerate(rts) if (l>=avg+(3*std) or l<=avg-(3*std)) and subjects[l_i]==sub]
                        to_remove.extend(sub_remove)
                    print('removed according to original criteria {} trials'.format(len(to_remove)))
                    current_cond = [l for l_i, l in enumerate(current_cond) if l_i not in to_remove]
                    log_rts = [numpy.log10(float(l[header.index('response_time')].replace(',', '.'))) for l in current_cond]
                    subjects = [int(l[header.index('subject')]) for l in current_cond]
                    rts = [float(l[header.index('response_time')].replace(',', '.')) for l in current_cond]
                    if 'prime' in prime:
                        # ones = primes, twos = targets
                        if prime == 'prime-cat':
                            ### prime -> target
                            w_ones = [l[header.index('prime')].lower() for l in current_cond]
                            vocab_w_ones = [w for ws in w_ones for w in transform_italian_word(ws)] 
                        elif prime == 'prime-proto':
                            w_ones = [prototypes[l[header.index('prime')][0]] for l in current_cond]
                            vocab_w_ones = [w for ws in w_ones for wz in ws for w in transform_italian_word(wz)] 
                        w_twos = [l[header.index('target')].lower() for l in current_cond]
                    elif 'target' in prime:
                        w_ones = [l[header.index('target')].lower() for l in current_cond]
                        vocab_w_ones = [w for ws in w_ones for w in transform_italian_word(ws)] 
                        ### inverting one and twos: ones = targets, twos = required choice
                        if prime == 'target-cat':
                            ### prime -> target
                            w_twos = [it_mapper[l[header.index('target_category')]] for l in current_cond]
                            vocab_w_twos = [w for ws in w_ones for w in transform_italian_word(ws)] 
                        elif prime == 'target-proto':
                            w_twos = [prototypes[it_mapper[l[header.index('target_category')]][0]] for l in current_cond]
                            vocab_w_twos = [w for ws in w_ones for wz in ws for w in transform_italian_word(wz)] 
                        elif prime == 'opposite-target-cat':
                            ### prime -> opposite target
                            w_twos = [it_mapper[[k for k in it_mapper.keys() if k!=l[header.index('target_category')]][0]] for l in current_cond]
                            vocab_w_twos = [w for ws in w_ones for w in transform_italian_word(ws)] 
                    test_vocab = test_vocab.union(set(vocab_w_ones))
                    vocab_w_twos = [w for ws in w_twos for w in transform_italian_word(ws)] 
                    test_vocab = test_vocab.union(set(vocab_w_twos))
                    #print(prime)
                    #print(set(vocab_w_twos))
                    print('available subjects per key: {}'.format([key, len(set(subjects))]))
                    all_sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                    triples = triples.union(set([v[1] for v in all_sims[key]]))
    final_sims = reorganize_sims(all_sims)
    collect_info(final_sims)

    ### we just return differences
    write_trials(args, triples)
    
    return final_sims, test_vocab
