import argparse
import random
import numpy
import os

from lang_models_utils import ContextualizedModelCard
from lang_models_utils import extract_surpr, read_pairs

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    type=str, 
                    required=True,
                     choices=[
                             'gpt2-small',
                             'llama-3b',
                             ], 
                    help = 'Specifies where the vectors are stored')
parser.add_argument('--lang', 
                    type=str, 
                    required=True,
                     choices=[
                         'it', 'de',
                             ], 
                     )
parser.add_argument('--corpus', 
                    type=str, 
                    default='wac',
                     choices=[
                         'wac',
                             ], 
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
args = parser.parse_args()

cases = read_pairs(args)

current_model = ContextualizedModelCard(args, causal=True)

entity_vectors = extract_surpr(
                               args, 
                               current_model,
                               cases,
                               )
out_f = os.path.join('llm_surprisals', args.lang, args.model)
os.makedirs(out_f, exist_ok=True)

print(current_model.n_layers, current_model.required_shape, )
for case, surprs in entity_vectors.items():
    with open(os.path.join(out_f, '{}_{}.tsv'.format(case, args.model)), 'w') as o:
        o.write('word_one\tword_two\tsurprisal_w_two\tentropy_w_one\n')
        for ws, surpr in surprs.items():
            w_one = ws.split()[0]
            w_two = ws.split(']')[1]
            o.write('{}\t{}\t{}\t{}\n'.format(w_one, w_two, surpr[0], surpr[1]))
