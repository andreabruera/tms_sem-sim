import os
import pickle

from tqdm import tqdm

from count_utils import load_wac_freqs
from test_utils import args, load_dataset, rt, test_model, test_precomputed_model

args = args()
trans_from_en = {}

wac_freqs = load_wac_freqs(args) 

rows, datasets = load_dataset(args, trans_from_en)

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

if args.model == 'response_times':
    model = dict()
    vocab = [w for w in rows]
    present_words = [w for w in rows]
    rt(
       args, 
       args.model,
       model, 
       vocab, 
       datasets, 
       present_words,
       trans_from_en,
       )
elif 'length' in args.model:
    test_precomputed_model(args, args.model, datasets, rows, wac_freqs)
elif args.model in static_models:
    test_precomputed_model(args, args.model, datasets, rows, wac_freqs)
elif 'llama' in args.model or 'pt' in args.model:
    test_precomputed_model(args, args.model, datasets, rows, wac_freqs)
### for count models, we test with a lot of different possibilities
else:
    ### frequency
    if 'abs-prob' in args.model:
        test_precomputed_model(args, args.model, datasets, rows, wac_freqs)
    ### surprisal
    elif 'surprisal' in args.model or 'cond-prob' in args.model:
        test_precomputed_model(args, args.model, datasets, rows, wac_freqs)
    else:
        for freq in tqdm(
                         top_freqs
                          ):
            for row_mode in [
                             '_', 
                             ]:
                for selection_mode in [
                                       'top', 
                                       ]: 
                    for vocab_type in [
                                       'abs_freq',
                                       ]:
                        key = 'ppmi_{}_{}_{}{}_{}_words'.format(args.model, vocab_type, selection_mode, row_mode, freq)
                        test_precomputed_model(args, key, datasets, rows, wac_freqs)
