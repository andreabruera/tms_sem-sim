import argparse
import random
import numpy
import os

from lang_models_utils import ContextualizedModelCard
from lang_models_utils import extract_vectors, read_all_sentences, read_pairs

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
                         'it', 'de'
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

#ws, replication_sentences = read_all_sentences(args)
#ws = {k : list() for k in ws}
cases = read_pairs(args)
ws = {k : list() for _ in cases.values() for __ in _ for k in __.split(' [SEP] ')}

current_model = ContextualizedModelCard(args)

entity_vectors, entity_sentences = extract_vectors(
                                                   args, 
                                                   current_model,
                                                   ws,
                                                   isolated=True,
                                                   )
out_f = os.path.join('llm_models', 'llm_vectors', args.lang, args.corpus, '{}-iso'.format(args.model))
os.makedirs(out_f, exist_ok=True)

print(current_model.n_layers, current_model.required_shape, )
for k, vecs in entity_vectors.items():
    vec = numpy.average(vecs[:min(10, len(vecs))], axis=0)
    ### vectors
    assert vec.shape == (current_model.n_layers, current_model.required_shape, )
    with open(os.path.join(out_f, '{}_{}.tsv'.format(k, args.model)), 'w') as o:
        o.write('word\tlayer\tvector\n')
        for layer in range(vec.shape[0]):
            layer_vec = vec[layer]
            o.write('{}\t{}\t'.format(k, layer))
            for dim in layer_vec:
                o.write('{}\t'.format(dim))
            o.write('\n')
