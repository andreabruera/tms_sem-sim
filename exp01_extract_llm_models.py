import argparse
import os

from test_utils import load_dataset

parser = argparse.ArgumentParser()
args = parser.parse_args()

### creating trial files
rel_datasets = [
            'de_pmtg-prod',
            'de_sem-phon',
            'it_distr-learn',
            'de_sound-action',
            'it_social-quantity',
            'de_behav',
            'it_anew',
            ]
for dataset in rel_datasets:
    args.dataset = dataset
    args.stat_approach = 'bootstrap'
    args.lang = dataset[:2]
    rows, datasets = load_dataset(args)

langs = ['de', 'it']
models = [
         'gpt2-small',
         'llama-3b',
         ]
for lang in langs:
    for model in models:
        for dataset in rel_datasets:
            '''
            os.system(
                  'python3 extract_LLMs.py '\
                  '--lang {} '\
                  '--model {} '\
                  '--modality {} '\
                  '--dataset {}'.format(
                              lang, 
                              model, 
                              dataset, 
                              )
                  )
            '''
            os.system(
                  'python3 extract_iso_LLMs.py '\
                  '--lang {} '\
                  '--model {} '\
                  '--dataset {}'.format(
                              lang, 
                              model, 
                              dataset, 
                              )
                  )
            os.system(
                  'python3 extract_surprisal.py '\
                  '--lang {} '\
                  '--model {} '\
                  '--dataset {}'.format(
                              lang, 
                              model, 
                              dataset, 
                              )
                  )
