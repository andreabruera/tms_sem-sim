import argparse
import random
import numpy
import os

from lang_models_utils import ContextualizedModelCard
from lang_models_utils import extract_vectors, read_all_sentences

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
args = parser.parse_args()

replication_sentences = read_all_sentences(args)

current_model = ContextualizedModelCard(args)

entity_vectors, entity_sentences = extract_vectors(
                                                   args, 
                                                   current_model,
                                                   replication_sentences,
                                                   )
out_f = os.path.join('llm_vectors', args.lang, args.corpus, args.model)
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
