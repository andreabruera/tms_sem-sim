import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
                     '--lang', 
                     choices=['de', 'it'],
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
args = parser.parse_args()
elif args.modality == 'tms':
    datasets = [
            ### tms
            'de_pmtg-prod',
            'de_sem-phon',
            'it_distr-learn',
            'de_sound-action',
            'it_social-quantity',
            ]
elif args.modality == 'behav':
    datasets = [
                ### behav
                'de_behav',
                'it_anew',
                ]
final_datasets = list()
for d in datasets:
    if d[2] == '_':
        if d[:3] == '{}_'.format(args.lang):
            final_datasets.append(d)
    else:
        final_datasets.append(d)

for dataset in final_datasets:
    corpora_choices = list()
    for corpus in [
                   'wac',
                   ]:
        corpora_choices.append('{}-ppmi-vecs'.format(corpus))
        for mode in [
                     'surprisal',
                     'one-neg-log10-abs-prob',
                     'two-neg-log10-abs-prob',
                     'visual-neg-log10-abs-prob',
                     'produced-neg-log10-abs-prob',
                     'overall-neg-log10-abs-prob',
                     ]:
            corpora_choices.append('{}-{}'.format(corpus, mode))
    choices = list()
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
        if 'iso' not in llm:
            choices.append('{}_surprisal'.format(llm))
        if 'iso' in llm:
            for l in range(m):
                choices.append('{}_layer-{}'.format(llm, l))
    choices= corpora_choices + choices + [
             'fasttext',
             'response_times',
             'visual-word-length',
             'one-word-length',
             'two-word-length',
             'overall-word-length',
             'produced-word-length',
             ]
    for model in choices:
        os.system(
                  'python3 extract_similarities.py '\
                  '--lang {} '\
                  '--model {} '\
                  '--modality {} '\
                  '--dataset {}'.format(
                              args.lang, 
                              model, 
                              args.modality,
                              dataset, 
                              )
                  )
