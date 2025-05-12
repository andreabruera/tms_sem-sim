import numpy
import os
import pickle
import scipy

from scipy import spatial, stats
from tqdm import tqdm

from count_utils import build_coocs_vecs, build_frequency_vecs, build_ppmi_vecs, load_count_coocs
from test_utils import args, load_dataset, rt, test_model
from extraction_utils import check_present_words, load_static_model, load_context_model, load_context_surpr

from utf_utils import transform_german_word

def compute_sim(args, model, pair):
    sims = list()
    if args.lang == 'de':
        ones = transform_german_word(pair[0])
        twos = transform_german_word(pair[1])
    else:
        ones = [pair[0]]
        twos = [pair[1]]
    for ver_one in ones:
        for ver_two in twos:
            if 'old20' in args.model:
                ver_one = ver_one.lower()
                ver_two = ver_two.lower()
            elif args.lang == 'de' and ('cc100' in args.model or 'wiki' in args.model):
                ver_one = ver_one.lower()
                ver_two = ver_two.lower()
            sim = extract_pair(args, model, (ver_one, ver_two))
            if sim != 'nan':
                sims.append(sim)
    if len(sims) == 0:
        #print(pair)
        return 'nan'
    else:
        ### surprisal is a bit more complicated
        if 'surprisal' in args.model:
            if 'pt' not in args.model:
                sim = -numpy.log2(1+sum(sims))
            else:
                sim = numpy.average(sims)
        elif 'prob' in args.model:
            sim = sum(sims)
            if 'log' in args.model:
                #print(sim)
                sim = -numpy.log10(1+sim)
            else:
                sim = -sim
        else:
            sim = numpy.average(sims)

        return sim

def extract_pair(args, model, pair):
    ### conditional probability
    if 'surprisal' in args.model:
        sims = list()
        for o, t in [(0, 1), (1, 0)]:
            try:
                sim = model[pair[o]][pair[t]]
            except KeyError:
                if 'pt' not in args.model:
                    sim = 0.
                else:
                    continue
            sims.append(sim)
        try:
            assert len(sims) >= 1
            sim = numpy.average(sims)
        except AssertionError:
            sim = 'nan'
    else:
        try:
            vals = [model[k] for k in pair]
            ### the more frequent, the shorter rt
            if 'social' not in args.dataset and \
               'distr' not in args.dataset and \
               'sem' not in args.dataset and \
               'sound' not in args.dataset and \
               'pmtg' not in args.dataset:
                if 'prob' in args.model:
                    sim = abs(vals[0]-vals[1])
                elif 'old20' in args.model or 'length' in args.model:
                    sim = abs(vals[0]-vals[1])
                else:
                    sim = scipy.spatial.distance.cosine(vals[0], vals[1])
            else:
                if 'prob' in args.model or 'length' in args.model or 'old20' in args.model:
                    if 'sound' in args.dataset:
                        sim = vals[1]
                    else:
                        if 'pmtg' in args.dataset:
                            if 'visual' in args.model:
                                sim = vals[0]
                            elif 'produced' in args.model:
                                sim = vals[1]
                            elif 'overall' in args.model:
                                sim = sum(vals)
                        elif 'sem' in args.dataset:
                            if 'visual' in args.model:
                                sim = vals[0]
                            elif 'produced' in args.model:
                                sim = vals[1]
                            elif 'overall' in args.model:
                                sim = sum(vals)
                        else:
                            if 'one' in args.model:
                                sim = vals[0]
                            elif 'two' in args.model:
                                sim = vals[1]
                            else:
                                sim = sum(vals)
                else:
                    sim = scipy.spatial.distance.cosine(vals[0], vals[1])
        except KeyError:
            sim = 'nan'
    return sim

def compute_sims(args, model_name, model, pairs):
    sims = list()
    for p in pairs:
        assert len(p) in [2, 3]
        ### social-quantity
        if len(p) == 3:
            diff = list()
            for _ in range(2):
                pair = (p[_], p[2])
                sim = compute_sim(args, model, pair)
                if sim != 'nan':
                    diff.append(sim)
            if len(diff) == 2:
                sim = abs(diff[0]-diff[1])
                sims.append((p, sim))
        else:
            sim = compute_sim(args, model, p)
            if sim != 'nan':
                sims.append((p, sim))
    folder = os.path.join('dissimilarities', args.lang, model_name,)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, '{}#dissimilarities.tsv'.format(args.dataset)), 'w') as o:
        o.write('words\tdissimilarities\n')
        for ws, sim in sims:
            o.write(','.join(ws))
            o.write('\t{}\n'.format(sim))

def count_model(args, key, datasets, present_words, trans_from_en, coocs, vocab, row_words, ctx_words):
    trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
    model = {k : v for k, v in trans_pmi_vecs.items()}
    return model

def frequency_model(args, key, datasets, present_words, trans_from_en, freqs, vocab, row_words):
    trans_pmi_vecs = build_frequency_vecs(args, freqs, row_words, )
    model = {k : v for k, v in trans_pmi_vecs.items()}
    return model

def coocs_model(args, key, datasets, present_words, trans_from_en, coocs, vocab, row_words):
    trans_pmi_vecs = build_coocs_vecs(args, coocs, row_words, vocab)
    model = {k : v for k, v in trans_pmi_vecs.items()}
    return model

numpy.seterr(all='raise')

args = args(similarity=True)
orig_dataset = str(args.dataset)
### creating trial files
datasets = [
            'de_pmtg-prod',
            'de_sem-phon',
            'it_distr-learn',
            'de_sound-action',
            'it_social-quantity',
            'de_behav',
            'it_anew',
            ]
for dataset in datasets:
    args.dataset = dataset
    rows, datasets = load_dataset(args, trans_from_en)

trans_from_en = {}

args.dataset = orig_dataset
rows, datasets = load_dataset(args, trans_from_en)

pairs = list()
folder = os.path.join('trials', args.lang)
assert os.path.exists(folder)
with open(os.path.join(folder, '{}#trials.tsv'.format(args.dataset))) as i:
    for l in i:
        line = l.strip().split('\t')
        pairs.append(line)

### for static models, we only test once
static_models = [
                 'fasttext',
                 ]
top_freqs = [
                          100, 
                          200, 
                          500, 
                          750,
                          1000, 
                          2500, 
                          5000, 
                          7500,
                          10000, 
                          12500, 
                          15000, 
                          17500,
                          20000, 
                          25000,
                          30000,
                          35000,
                          40000,
                          45000,
                          50000,
                          60000,
                          70000,
                          80000,
                          90000,
                          100000,
                          150000,
                          200000,
                          250000,
                          300000,
                          350000,
                          400000,
                          450000,
                          500000,
                          ]
if 'length' in args.model:
    model = {k : len(k) for k in rows}
    compute_sims(args, args.model, model, pairs)
elif 'old20' in args.model:
    with open(os.path.join('..', '..', 'counts', args.lang, 'wac', '{}_wac_10_min-uncased_OLD20.pkl'.format(args.lang)), 'rb') as i:
        model = pickle.load(i)
    compute_sims(args, args.model, model, pairs)
elif args.model in static_models:
    model, vocab = load_static_model(args)
    compute_sims(args, args.model, model, pairs)
elif 'llama' in args.model or 'pt' in args.model:
    if 'surpr' not in args.model:
        model, vocab = load_context_model(args)
    else:
        model, vocab = load_context_surpr(args)
    compute_sims(args, args.model, model, pairs)
### for count models, we test with a lot of different possibilities
else:
    vocab, coocs, freqs, pos = load_count_coocs(args)
    ### keeping row words that are actually available
    row_words = [w for w in rows if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0]
    present_words = check_present_words(args, row_words, list(vocab.keys()))
    print('\nnumber of missing: {}\n'.format(1-len(present_words)/len(rows)))
    print(rows.difference(set(present_words)))
    if 'abs-prob' in args.model:
        model = frequency_model(args, args.model, datasets, present_words, trans_from_en, freqs, vocab, row_words,)
        compute_sims(args, args.model, model, pairs)
    elif 'surprisal' in args.model or 'cond-prob' in args.model:
        model = coocs_model(args, args.model, datasets, present_words, trans_from_en, coocs, vocab, row_words,)
        compute_sims(args, args.model, model, pairs)
    else:
        #
        ### top-n frequencies
        #
        for row_mode in [
                         '_', 
                         ]:
            for selection_mode in [
                                   'top', 
                                   ]: 
                for vocab_type in [
                                   'abs_freq',
                                   ]:
                    filt_freqs = {w : f for w, f in freqs.items() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
                    sorted_freqs = [w[0] for w in sorted(filt_freqs.items(), key=lambda item: item[1], reverse=True)]
                    for freq in tqdm(
                                     top_freqs
                                      ):
                        key = 'ppmi_{}_{}_{}{}_{}_words'.format(
                                                       args.model, 
                                                       vocab_type,
                                                       selection_mode, 
                                                       row_mode, 
                                                       freq,
                                                       )
                        if freq > len(vocab.keys()):
                            print('too many words requested, skipping!')
                            continue
                        if selection_mode == 'top':
                            if row_mode == 'rowincol':
                                ctx_words = set([w for w in sorted_freqs[:freq]]+row_words)
                            else:
                                ctx_words = [w for w in sorted_freqs[:freq]]
                        else:
                            random.seed(12)
                            idxs = random.sample(range(len(sorted_freqs)), k=min(freq, len(sorted_freqs)))
                            if row_mode == 'rowincol':
                                ctx_words = set([sorted_freqs[i] for i in idxs]+row_words)
                            else:
                                ctx_words = [sorted_freqs[i] for i in idxs]
                        ### using the basic required vocab for all tests as a basis set of words
                        model = count_model(args, key, datasets, present_words, trans_from_en, coocs, vocab, row_words, ctx_words)
                        compute_sims(args, key, model, pairs)
