import matplotlib
import mne
import numpy
import os
import pingouin
import random
import re
import scipy

from matplotlib import colormaps, font_manager, pyplot
from mne import stats
from scipy import stats

from plot_utils import font_setup, perm_against_zero, permutation_two_samples, read_effsizes, set_colors
from tqdm import tqdm

### effect_sizes
lo_e, mid_e, hi_e = read_effsizes(mode='cog_neuro')
print(lo_e)

font_folder = '../../fonts'
font_setup(font_folder)

for mode in [
             'residualize', 
             'bootstrap',
             ]:
    for corpus in [
                   #'opensubs', 
                   'wac', 
                   #'cc100',
                   ]:

        results = dict()

        for root, direc, fz in os.walk(
                                  os.path.join(
                                      'test_results',
                                      )):
            for f in fz:
                parts = root.split('/')
                approach = parts[1]
                if approach != 'correlation':
                    continue
                stat_approach = parts[2]
                if stat_approach != mode:
                    continue
                evaluation = parts[3]
                if evaluation != 'spearman':
                    continue
                lang = parts[4]
                n = '100000'
                if 'gpt2-small_surprisal' in root:
                    model = 'GPT2\nsurprisal'
                    pass
                elif corpus in root and n in root:
                    model = '{}\nPPMI'.format(corpus)
                    pass
                elif corpus in root and 'abs-prob' in root:
                    if 'one-' in root:
                        model = 'Word\nfrequency\n(neg.\nfirst w.)'
                    elif 'two-' in root:
                        model = 'Word\nfrequency\n(neg.\nsecond w.)'
                    elif 'visual-' in root:
                        model = 'Word\nfrequency\n(neg.\nvisual)'
                    elif 'produced-' in root:
                        model = 'Word\nfrequency\n(neg.\nuttered)'
                    elif 'overall-' in root:
                        model = 'Word\nfrequency\n(neg. sum)'
                    else:
                        continue
                    print(root)
                    print(model.replace('\n', ' '))
                    pass
                elif 'word-length' in root:
                    if 'one-' in root:
                        model = 'Word\nlength\n(first w.)'
                    elif 'two-' in root:
                        model = 'Word\nlength\n(second w.)'
                    elif 'visual-' in root:
                        model = 'Word\nlength\n(visual)'
                    elif 'produced-' in root:
                        model = 'Word\nlength\n(uttered)'
                    elif 'overall-' in root:
                        model = 'Word\nlength\n(sum)'
                    else:
                        continue
                    pass
                else:
                    continue
                with open(os.path.join(root, f)) as i:
                    for l in i:
                        line = l.strip().split('\t')
                        lang = line[0]
                        if lang not in results.keys():
                            results[lang] = dict()
                        old_model = line[1]
                        all_task = line[2]
                        ### modality
                        assert all_task[:3] == '{}_'.format(lang)
                        task = all_task[3:].split('#')[0].split('_')[0]
                        if 'sem' in task or 'pmtg' in task:
                            splitter = '-'
                        else:
                            splitter = '_'
                        if 'sound' in task:
                            cond_idx = -2
                        else:
                            cond_idx = -1
                        case = all_task.split('#')[-1].split(splitter)[0]
                        cond = all_task.split('#')[-1].split(splitter)[cond_idx]
                        cond = '{}{}'.format(cond[0].lower(), cond[1:])
                        if cond == 'cedx':
                            cond = 'rCereb'
                        if cond == 'cz':
                            cond = 'vertex'
                        if cond not in ['sham', 'vertex']:
                            cond = 'TMS\n{}'.format(cond)
                        if 'distr-learn' in all_task:
                            pass
                        elif 'pmtg-prod' in all_task:
                            if '-but-' in all_task:
                                continue
                            pass
                        elif 'sem-phon' in all_task:
                            pass
                        elif 'sound-act' in all_task:
                            if 'together-pos-all' not in all_task:
                                continue
                            if 'all_all' in all_task:
                                continue
                            if 'detailed' in all_task:
                                continue
                            case = '{}-{}'.format(case, all_task.split('_')[-1])
                            pass
                        elif 'social' in all_task:
                            if 'prime-cat' not in all_task:
                                continue
                            if 'cong' in all_task:
                                continue
                            pass
                        else:
                            continue
                        if task not in results[lang].keys():
                            results[lang][task] = dict()
                        if case not in results[lang][task].keys():
                            results[lang][task][case] = dict()
                        if cond not in results[lang][task][case].keys():
                            results[lang][task][case][cond] = dict()
                        non_nan_res = [v if v!='nan' else 0. for v in line[3:]]
                        res = numpy.array(non_nan_res, dtype=numpy.float32)
                        results[lang][task][case][cond][model] = res[:1000]
        colors = {
                  '{}\nPPMI'.format(corpus) : ('navy', 'royalblue', 'lightsteelblue',),
                  'GPT2\nvectors' : ('lightskyblue', 'lightblue', 'paleturquoise'),
                  '{}\nsurprisal'.format(corpus) : ('mediumorchid', 'thistle', 'plum'),
                  'GPT2\nsurprisal' : ('mediumvioletred', 'pink', 'palevioletred'),
                  'Word\nfrequency' : ('sienna', 'peru', 'sandybrown'),
                  'Word\nlength' : ('gray', 'darkgray', 'lightgray'),
                  }

        out_f = os.path.join('plots', '00', mode, corpus)
        os.makedirs(out_f, exist_ok=True)

        for lang, l_results in results.items():
            for task, t_results in l_results.items():
                for case, c_results in t_results.items():
                    ### getting ready to write things down...
                    lines = list()
                    gen_line = [mode, corpus, lang, task, case]
                    curr_fold = os.path.join(out_f, lang, task, case)
                    os.makedirs(curr_fold, exist_ok=True)
                    conds = sorted(c_results.keys(), reverse=True)
                    models = set([m for _ in c_results.values() for m in _.keys()])
                    no_tms_cond = [c for c in conds if 'ver' in c or 'sh' in c][0]
                    models = set([m for _ in c_results.values() for m in _.keys()])
                    best_model = sorted(
                                      [(c_results[no_tms_cond][m], m) for m in models if 'length' not in m and 'frequency' not in m], 
                                      key=lambda item : numpy.average(item[0]),
                                      reverse=True,
                                      )[0][1]
                    sorted_models = [
                                    '{}\nPPMI'.format(corpus), 
                                    ] +\
                                    ['GPT2\nsurprisal',] +\
                                    [m for m in models if 'freq' in m and ('visual' in m or 'first' in m)] +\
                                    [m for m in models if 'freq' in m and ('utter' in m or 'second' in m)] +\
                                    [m for m in models if 'freq' in m and ('sum' in m)] +\
                                    [m for m in models if 'ength' in m and ('visual' in m or 'first' in m)] +\
                                    [m for m in models if 'ength' in m and ('utter' in m or 'second' in m)] +\
                                    [m for m in models if 'ength' in m and ('sum' in m)]
                    print(sorted_models)
                    try:
                        if 'together' not in case:
                            assert len(sorted_models) > 6
                        else:
                            assert len(sorted_models) == 4

                    except AssertionError:
                        print(case)
                        print(models)
                        continue
                    xs = list(range(len(sorted_models)))
                    if len(conds) == 2:
                        corrections = list(numpy.linspace(-.33, .33, len(conds)))
                        txt_corrections = list(numpy.linspace(-.4, .4, len(conds)))
                        m_sc = 2000
                        t_s = 20
                    else:
                        corrections = list(numpy.linspace(-.5, .5, len(conds)))
                        txt_corrections = list(numpy.linspace(-.55, .55, len(conds)))
                        m_sc = 1400
                        t_s = 15
                    if 'together' in case:
                        figsize=(15, 10)
                    else:
                        figsize=(20, 10)
                    fig, ax = pyplot.subplots(constrained_layout=True, figsize=figsize)
                    x_shift = 0
                    xticks = list()
                    counter = -1
                    ps = list()
                    for m_i, m in enumerate(sorted_models):
                        if 'Resp' in m:
                            gen_avg = numpy.average([v for _ in c_results.values() for __ in _.values() for v in __])
                            gen_std = numpy.std([v for _ in c_results.values() for __ in _.values() for v in __])
                        counter += 1
                        for c_i, c in enumerate(conds):
                            spec_line = [v for v in gen_line]
                            spec_line.extend([m, c])
                            if 'length' in m:
                                color=colors['Word\nlength'][c_i]
                            elif 'freq' in m:
                                color=colors['Word\nfrequency'][c_i]
                            else:
                                color=colors[m][c_i]
                            xticks.append((counter, m))
                            
                            if len(conds) == 2:
                                w = 0.6
                            else:
                                w = 0.45
                            if c_i == 0:
                                comps = list()
                                for other_i, other in enumerate(conds[1:]):
                                    two = c_results[other][m]
                                    t_val, p_val, fake_distr, ci = permutation_two_samples(c_results[c][m], two)
                                    print([m, c, other, p_val, t_val])
                                    ps.append((m, (c, other), p_val, t_val))
                                    comps.append('{}_{}_{}_{}_{}_{}@{}'.format(c, other, ci[0], ci[1], p_val, t_val, ','.join([str(v) for v in fake_distr])))
                                spec_line.append('#'.join(comps))
                            else:
                                spec_line.append('na')
                            ### simple p-value
                            t, p, ci = perm_against_zero(c_results[c][m])
                            #print(p)
                            ps.append((m, c, p))
                            spec_line.extend([p, t, ci])
                            ### bar
                            ax.bar(
                                   m_i+corrections[c_i]+x_shift, 
                                   numpy.average(c_results[c][m]),
                                   width=w,
                                   color=color,
                                   edgecolor='gray',
                                   zorder=2.
                                   )
                            ax.errorbar(
                                   m_i+corrections[c_i]+x_shift, 
                                   numpy.average(c_results[c][m]),
                                   yerr=numpy.std(c_results[c][m]),
                                   color='black', 
                                   capsize=5,
                                   zorder=3.
                                   )
                            spec_line.extend([m_i+corrections[c_i]+x_shift, m_i+txt_corrections[c_i]+x_shift])
                            spec_line.append(numpy.average(c_results[c][m]))
                            spec_line.append(','.join([str(val) for val in c_results[c][m]]))
                            lines.append(spec_line)
                            if len(c_results[c][m]) == 1000:
                                alpha = 0.2
                            elif len(c_results[c][m]) == 10000:
                                alpha = 0.02
                            ax.scatter(
                                   [m_i+corrections[c_i]+x_shift+(random.randrange(-m_sc, m_sc)*0.0001) for rand in range(len(c_results[c][m]))], 
                                   c_results[c][m],
                                   color=color,
                                   edgecolor='white',
                                   alpha=alpha,
                                   zorder=2.5
                                   )
                            ax.text(
                                   m_i+txt_corrections[c_i]+x_shift, 
                                   -.05,
                                   s=c,
                                   fontsize=t_s,
                                   ha='center',
                                   va='center',
                                   )
                        x_shift += 1
                        counter += 1
                        if m_i in [0, 1, 4]:
                            x_shift += 1
                            counter += 1
                    ax.set_ylim(bottom=-.08, top=.38)
                    ax.hlines(xmin=-.8, xmax=len(sorted_models)+x_shift-2.2, color='black', y=0)
                    ax.hlines(xmin=-.8, xmax=len(sorted_models)+x_shift-2.2, color='silver',alpha=0.5,linestyle='dashed', y=[y*0.01 for y in range(-5, 35, 5)], zorder=1)
                    pyplot.ylabel('Spearman correlation (RSA RT-model)', fontsize=23)
                    pyplot.xticks(
                                  [x[0] for x in xticks], 
                                  [x[1] for x in xticks],
                                  fontsize=25,
                                  fontweight='bold')
                    print(curr_fold)
                    pyplot.savefig(os.path.join(curr_fold, '{}_00.jpg'.format(case)), dpi=300)
                    ### writing to file
                    with open(os.path.join(curr_fold, '{}_00.txt'.format(case)), 'w') as o:
                        o.write('mode\tcorpus\tlang\ttask\tcase\tmodel\tcondition\t')
                        o.write('comparisons\t')
                        o.write('p_val_raw\tt_val\tci_constant\tx_bar\tx_label\tperms_avg\tperms\n')
                        for lin in lines:
                            for li in lin:
                                o.write('{}\t'.format(str(li).replace('\n', '$')))
                            o.write('\n')
### running correction for multiple comparisons
corr_f = os.path.join('plots', 'stats_after_correction')
os.makedirs(corr_f, exist_ok=True)
for corpus in [
               'wac',
               ]:
    final_corr = list()
    final_to_be_corr = list()
    results = dict()
    fakes = list()
    for approach in [
                     'residualize', 
                     'bootstrap',
                     ]:
        fold = os.path.join('plots', '00', approach, corpus)
        for lang in ['de', 'it']:
            for root, direc, fz in os.walk(os.path.join(fold, lang)):
                for f in fz:
                    if 'txt' in f:
                        assert '00' in f
                        with open(os.path.join(root, f)) as i:
                            for l_i, l in enumerate(i):
                                line = l.strip().split('\t')
                                if l_i == 0:
                                    header = line.copy()
                                    continue
                                task = line[header.index('task')]
                                case = line[header.index('case')]
                                if 'sound' in task:
                                    full_task = '{}_{}_{}'.format(approach, task, case[0])
                                elif 'social' in task:
                                    full_task = '{}_{}_{}'.format(approach, task, case.split('-')[0])
                                else:
                                    full_task = '{}_{}'.format(approach, task)
                                if full_task not in results.keys():
                                    results[full_task] = list()
                                cond = line[header.index('condition')]
                                model = line[header.index('model')]
                                p_raw = float(line[header.index('p_val_raw')])
                                t_val = float(line[header.index('t_val')])
                                bar_y = float(line[header.index('perms_avg')])
                                bar_x = float(line[header.index('x_bar')])
                                ci_constant = float(line[header.index('ci_constant')])
                                label_x = float(line[header.index('x_label')])
                                perms = [float(val) for val in line[header.index('perms')].split(',')]
                                assert len(perms) in [1000, 10000]
                                comparisons = [val.split('@')[0].split('_') for val in line[header.index('comparisons')].split('#')]
                                curr_dict = {
                                             'cond' : cond,
                                             'case' : case,
                                             'model' : model,
                                             'p_raw' : p_raw,
                                             't_val' : t_val,
                                             'bar_y' : bar_y,
                                             'bar_x' : bar_x,
                                             'label_x' : label_x,
                                             'perms' : perms,
                                             'ci_constant' : ci_constant,
                                             'comparisons' : comparisons,
                                             }
                                results[full_task].append(curr_dict)
    print(results.keys())
    ### correcting p-values only once
    to_be_corr = list()
    counter = 0
    for k, v in results.items():
        for case in v:
            to_be_corr.append(
                              (
                                (k, 
                                 case['case'], 
                                 case['cond'], 
                                 case['model'],
                                 numpy.mean(case['perms']),
                                 case['ci_constant'],
                                 ), 
                               case['t_val'], 
                               case['p_raw']
                               )
                              )
            for cmpr in case['comparisons']:
                #print(cmpr)
                if cmpr != ['na']:
                    counter += 1
                    to_be_corr.append(
                                      (
                                        (k, 
                                         case['case'], 
                                         case['cond'], 
                                         case['model'], 
                                         cmpr[:4]
                                         ), 
                                        ### t-val
                                       float(cmpr[5]), 
                                        ### p-val
                                       float(cmpr[4])
                                       )
                                      )
    print(counter) 
    ### baseline
    corr = mne.stats.fdr_correction([v[2] for v in to_be_corr])[1]
    final_corr.extend(corr)
    final_to_be_corr.extend(to_be_corr)
    with open(os.path.join(corr_f, '{}_corrected_baseline_p-vals.tsv'.format(corpus)), 'w') as o:
        o.write('approach\tdataset\tcase\tcondition\tmodel\tci_min\tci_max\tperms_avg\tt_value\traw_p\tfdr_corrected_p\n')
        for d, cp in zip(final_to_be_corr, final_corr):
            if len(d[0]) == 6:
                approach = d[0][0].split('_')[0]
                task = d[0][0].split('_')[1]
                case = d[0][1]
                cond = d[0][2]
                model = d[0][3]
                avg = d[0][4]
                ci_constant = d[0][5]
                ci_min = float(avg)-float(ci_constant)
                ci_max = float(avg)+float(ci_constant)
                raw_p = d[2]
                t = d[1]
                o.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(approach, task, case, cond, model, ci_min, ci_max, avg, t, raw_p, cp))
    with open(os.path.join(corr_f, '{}_corrected_comparisons_p-vals.tsv'.format(corpus)), 'w') as o:
        o.write('approach\tdataset\tcase\tcondition_one\tcondition_two\tmodel\tci_min\tci_max\tt_value\traw_p\tfdr_corrected_p\n')
        for d, cp in zip(final_to_be_corr, final_corr):
            if len(d[0]) == 5:
                #print(d)
                approach = d[0][0].split('_')[0]
                task = d[0][0].split('_')[1]
                case = d[0][1]
                model = d[0][3]
                cond_one = d[0][4][0]
                cond_two = d[0][4][1]
                ci_min = d[0][4][2]
                ci_max = d[0][4][3]
                raw_p = d[2]
                t = d[1]
                o.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        approach, 
                        task, 
                        case, 
                        cond_one,
                        cond_two, 
                        model, 
                        ci_min, 
                        ci_max, 
                        t,
                        raw_p, 
                        cp)
                        )


font_folder = '../../fonts'
font_setup(font_folder)

for model_sel in [
                  'ppmi', 
                  'both', 
                  ]:
    for mode in [ 
                 'residualize',  
                 'bootstrap', 
                 ]: 
        for corpus in [ 
                       'wac',  
                       ]: 
            baseline = list() 
            with open(os.path.join(corr_f, '{}_corrected_baseline_p-vals.tsv'.format(corpus))) as i: 
                for l_i, l in enumerate(i): 
                    line = l.strip().split('\t') 
                    if l_i == 0: 
                        continue 
                    baseline.append(line) 
     
            comparisons = list() 
            with open(os.path.join(corr_f, '{}_corrected_comparisons_p-vals.tsv'.format(corpus))) as i: 
                for l_i, l in enumerate(i): 
                    line = l.strip().split('\t') 
                    if l_i == 0: 
                        continue 
                    comparisons.append(line)
            results = dict() 
     
            for root, direc, fz in os.walk( 
                                      os.path.join( 
                                          'test_results', 
                                          )): 
                for f in fz: 
                    parts = root.split('/') 
                    approach = parts[1] 
                    if approach != 'correlation': 
                        continue 
                    stat_approach = parts[2] 
                    if stat_approach != mode: 
                        continue 
                    evaluation = parts[3] 
                    if evaluation != 'spearman': 
                        continue 
                    lang = parts[4] 
                    n = '100000' 
                    if 'gpt2-small_surprisal' in root: 
                        model = 'GPT2\nsurprisal' 
                        pass 
                    elif corpus in root and n in root: 
                        model = '{}\nPPMI'.format(corpus) 
                        pass 
                    elif corpus in root and 'abs-prob' in root:
                        if 'one-' in root:
                            model = 'Word\nfrequency\n(neg.\nfirst w.)'
                        elif 'two-' in root:
                            model = 'Word\nfrequency\n(neg.\nsecond w.)'
                        elif 'visual-' in root:
                            model = 'Word\nfrequency\n(neg.\nvisual)'
                        elif 'produced-' in root:
                            model = 'Word\nfrequency\n(neg.\nuttered)'
                        elif 'overall-' in root:
                            model = 'Word\nfrequency\n(neg. sum)'
                        else:
                            continue
                        print(root)
                        print(model.replace('\n', ' '))
                        pass
                    elif 'word-length' in root:
                        if 'one-' in root:
                            model = 'Word\nlength\n(first w.)'
                        elif 'two-' in root:
                            model = 'Word\nlength\n(second w.)'
                        elif 'visual-' in root:
                            model = 'Word\nlength\n(visual)'
                        elif 'produced-' in root:
                            model = 'Word\nlength\n(uttered)'
                        elif 'overall-' in root:
                            model = 'Word\nlength\n(sum)'
                        else:
                            continue
                        pass
                    else: 
                        continue 
                    with open(os.path.join(root, f)) as i: 
                        for l in i: 
                            line = l.strip().split('\t') 
                            lang = line[0] 
                            if lang not in results.keys(): 
                                results[lang] = dict() 
                            old_model = line[1] 
                            all_task = line[2] 
                            ### modality 
                            assert all_task[:3] == '{}_'.format(lang) 
                            task = all_task[3:].split('#')[0].split('_')[0] 
                            if 'sem' in task or 'pmtg' in task: 
                                splitter = '-' 
                            else: 
                                splitter = '_' 
                            if 'sound' in task: 
                                cond_idx = -2 
                            else: 
                                cond_idx = -1 
                            case = all_task.split('#')[-1].split(splitter)[0] 
                            cond = all_task.split('#')[-1].split(splitter)[cond_idx] 
                            cond = '{}{}'.format(cond[0].lower(), cond[1:]) 
                            if cond == 'cedx': 
                                cond = 'rCereb' 
                            if cond == 'cz': 
                                cond = 'vertex' 
                            if cond not in ['sham', 'vertex']: 
                                cond = 'TMS\n{}'.format(cond) 
                            if 'distr-learn' in all_task: 
                                pass 
                            elif 'pmtg-prod' in all_task: 
                                if '-but-' in all_task: 
                                    continue 
                                pass 
                            elif 'sem-phon' in all_task: 
                                pass 
                            elif 'sound-act' in all_task:
                                if 'together-pos-all' not in all_task: 
                                    continue 
                                if 'all_all' in all_task: 
                                    continue 
                                if 'detailed' in all_task: 
                                    continue 
                                case = '{}-{}'.format(case, all_task.split('_')[-1]) 
                                pass 
                            elif 'social' in all_task: 
                                if 'prime-cat' not in all_task: 
                                    continue 
                                if 'cong' in all_task: 
                                    continue 
                                pass 
                            else: 
                                continue 
                            if task not in results[lang].keys(): 
                                results[lang][task] = dict() 
                            if case not in results[lang][task].keys(): 
                                results[lang][task][case] = dict()
                            if cond not in results[lang][task][case].keys(): 
                                results[lang][task][case][cond] = dict() 
                            non_nan_res = [v if v!='nan' else 0. for v in line[3:]] 
                            res = numpy.array(non_nan_res, dtype=numpy.float32) 
                            results[lang][task][case][cond][model] = res[:1000] 
            colors = { 
                      '{}\nPPMI'.format(corpus) : ('seagreen', 'mediumaquamarine', 'mediumseagreen'), 
                      'GPT2\nvectors' : ('lightskyblue', 'lightblue', 'paleturquoise'), 
                      '{}\nsurprisal'.format(corpus) : ('mediumorchid', 'thistle', 'plum'), 
                      'GPT2\nsurprisal' : ('mediumvioletred', 'pink', 'palevioletred'), 
                      'Word\nfrequency' : ('sienna', 'peru', 'sandybrown'), 
                      'Word\nlength' : ('gray', 'darkgray', 'lightgray'), 
                      } 
     
            out_f = os.path.join('plots', 'corrected', model_sel, mode, corpus) 
            os.makedirs(out_f, exist_ok=True) 
     
            for lang, l_results in results.items(): 
                for task, t_results in l_results.items(): 
                    for case, c_results in t_results.items(): 
                        ### getting ready to write things down... 
                        lines = list() 
                        gen_line = [mode, corpus, lang, task, case] 
                        curr_fold = os.path.join(out_f, lang, task, case) 
                        os.makedirs(curr_fold, exist_ok=True) 
                        conds = sorted(c_results.keys(), reverse=True) 
                        models = set([m for _ in c_results.values() for m in _.keys()]) 
                        print(models)
                        no_tms_cond = [c for c in conds if 'ver' in c or 'sh' in c][0] 
                        models = set([m for _ in c_results.values() for m in _.keys()]) 
                        best_model = sorted( 
                                          [(c_results[no_tms_cond][m], m) for m in models if 'length' not in m and 'frequency' not in m],  
                                          key=lambda item : numpy.average(item[0]), 
                                          reverse=True,
                                          )[0][1] 
                        print('best: {}'.format(best_model))
                        if model_sel == 'ppmi':
                            sorted_models = ['{}\nPPMI'.format(corpus)] +\
                                        [m for m in models if 'freq' in m and 'sum' in m] +\
                                        [m for m in models if 'ength' in m and 'sum' in m]
                        elif model_sel == 'both':
                            sorted_models = [
                                    '{}\nPPMI'.format(corpus), 
                                    ] +\
                                    ['GPT2\nsurprisal',] +\
                                    [m for m in models if 'freq' in m and ('visual' in m or 'first' in m)] +\
                                    [m for m in models if 'freq' in m and ('utter' in m or 'second' in m)] +\
                                    [m for m in models if 'freq' in m and ('sum' in m)] +\
                                    [m for m in models if 'ength' in m and ('visual' in m or 'first' in m)] +\
                                    [m for m in models if 'ength' in m and ('utter' in m or 'second' in m)] +\
                                    [m for m in models if 'ength' in m and ('sum' in m)]
                        try: 
                            assert len(sorted_models) in [3, 4, 6, 8] 
                        except AssertionError: 
                            print(case) 
                            print(models) 
                            continue 
                        xs = list(range(len(sorted_models))) 
                        if len(conds) == 2: 
                            corrections = list(numpy.linspace(-.33, .33, len(conds))) 
                            txt_corrections = list(numpy.linspace(-.4, .4, len(conds))) 
                            m_sc = 2000 
                            t_s = 20 
                        else: 
                            corrections = list(numpy.linspace(-.5, .5, len(conds))) 
                            txt_corrections = list(numpy.linspace(-.55, .55, len(conds))) 
                            m_sc = 1400 
                            t_s = 15 
                        fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10)) 
                        x_shift = 0 
                        xticks = list() 
                        counter = -1 
                        ps = list() 
                        for m_i, m in enumerate(sorted_models): 
                            if 'Resp' in m: 
                                gen_avg = numpy.average([v for _ in c_results.values() for __ in _.values() for v in __]) 
                                gen_std = numpy.std([v for _ in c_results.values() for __ in _.values() for v in __]) 
                            counter += 1 
                            for c_i, c in enumerate(conds): 
                                spec_line = [v for v in gen_line] 
                                spec_line.extend([m, c]) 
                                if 'length' in m:
                                    color=colors['Word\nlength'][c_i]
                                elif 'freq' in m:
                                    color=colors['Word\nfrequency'][c_i]
                                else:
                                    color=colors[m][c_i]
                                xticks.append((counter, m)) 
                                 
                                if len(conds) == 2: 
                                    w = 0.6 
                                else: 
                                    w = 0.45 
                                if c_i == 0: 
                                    comps = list() 
                                    for other_i, other in enumerate(conds[1:]): 
                                        two = c_results[other][m] 
                                        ### collecting corrected pvalue 
                                        for line in comparisons: 
                                            if line[0] != mode: 
                                                continue 
                                            if line[1] != task: 
                                                continue 
                                            if line[2] != case: 
                                                continue 
                                            if line[3] != c.replace('\n', '$'): 
                                                continue 
                                            if line[4] != other.replace('\n', '$'): 
                                                continue 
                                            if line[5] != m.replace('\n', '$'): 
                                                continue 
                                            print(line)
                                            t_val = float(line[-3]) 
                                            raw_p = float(line[-2])
                                            p_val = float(line[-1]) 
                                            print(line)
                                        ps.append((m, (c, other), p_val, t_val)) 
                                        fake_distr = ['na'] 
                                        comps.append('{}_{}_{}_{}_{}@{}'.format(c, other, raw_p, p_val, t_val, ','.join([str(v) for v in fake_distr]))) 
                                    spec_line.append('#'.join(comps)) 
                                else: 
                                    spec_line.append('na') 
                                for line in baseline: 
                                    if line[0] != mode: 
                                        continue 
                                    if line[1] != task: 
                                        continue 
                                    if line[2] != case: 
                                        continue 
                                    if line[3] != c.replace('\n', '$'): 
                                        continue 
                                    if line[4] != m.replace('\n', '$'): 
                                        continue 
                                    t = float(line[-3]) 
                                    raw_p = float(line[-2])
                                    p = float(line[-1]) 
                                ps.append((m, c, p, t)) 
                                spec_line.extend([raw_p, p, t]) 
                                ### bar 
                                ax.bar( 
                                       m_i+corrections[c_i]+x_shift,  
                                       numpy.average(c_results[c][m]), 
                                       width=w, 
                                       color=color, 
                                       edgecolor='gray', 
                                       zorder=2. 
                                       ) 
                                ax.bar( 
                                       m_i+corrections[c_i]+x_shift,  
                                       numpy.average(c_results[c][m]), 
                                       width=w, 
                                       fill=False,
                                       edgecolor='dimgray', 
                                       zorder=3. 
                                       ) 
                                ax.errorbar( 
                                       m_i+corrections[c_i]+x_shift,  
                                       numpy.average(c_results[c][m]), 
                                       yerr=numpy.std(c_results[c][m]), 
                                       color='dimgray',  
                                       capsize=5, 
                                       zorder=3. 
                                       ) 
                                spec_line.extend([m_i+corrections[c_i]+x_shift, m_i+txt_corrections[c_i]+x_shift]) 
                                spec_line.append(numpy.average(c_results[c][m])) 
                                spec_line.append(','.join([str(val) for val in c_results[c][m]])) 
                                spec_line.append(c_i) 
                                lines.append(spec_line) 
                                if len(c_results[c][m]) == 1000: 
                                    alpha = 0.2 
                                elif len(c_results[c][m]) == 10000: 
                                    alpha = 0.02 
                                ax.scatter( 
                                       [m_i+corrections[c_i]+x_shift+(random.randrange(-m_sc, m_sc)*0.0001) for rand in range(len(c_results[c][m]))],  
                                       c_results[c][m], 
                                       color=color, 
                                       edgecolor='white', 
                                       alpha=alpha, 
                                       zorder=2.5 
                                       ) 
                                ax.text( 
                                       m_i+txt_corrections[c_i]+x_shift,  
                                       -.08, 
                                       s=c, 
                                       fontsize=t_s, 
                                       ha='center', 
                                       va='center', 
                                       ) 
                            x_shift += 1 
                            counter += 1 
                            if m_i in [11, 33]: 
                                x_shift += 1 
                                counter += 1 
                        nonshaded = list()
                        ### absolute p-values 
                        corr_ps = mne.stats.fdr_correction([v[2] for v in ps])[1] 
                        corr_ps = [(ps[i][0], ps[i][1], ps[i][2], ps[i][3]) for i, p in enumerate(corr_ps) if type(ps[i][1]!=tuple)] 
                        x_shift = 0 
                        counter = -1 
                        for m_i, m in enumerate(sorted_models): 
                            counter += 1 
                            for c_i, c in enumerate(conds): 
                                for pm, pc, pp, t in corr_ps: 
                                    if pm == m and pc == c: 
                                        if abs(t)>lo_e:
                                            print('FOTTUTO D {}'.format(t))
                                        if pp < 0.005 and abs(t)>lo_e: 
                                            print(pp) 
                                            ax.scatter( 
                                                   m_i+corrections[c_i]+x_shift-.075,  
                                                   0.01, 
                                                   color='black', 
                                                   edgecolor='white', 
                                                   zorder=3., 
                                                   marker='*', 
                                                   s=300 
                                                   ) 
                                            ax.scatter( 
                                                   m_i+corrections[c_i]+x_shift+.075,  
                                                   0.01, 
                                                   color='black', 
                                                   edgecolor='white', 
                                                   zorder=3., 
                                                   marker='*', 
                                                   s=300 
                                                   ) 
                                        elif (str(pp)[:4] == '0.05' or pp<0.05) and abs(t)>lo_e: 
                                            print(pp) 
                                            ax.scatter( 
                                                   m_i+corrections[c_i]+x_shift,  
                                                   0.01, 
                                                   color='black', 
                                                   edgecolor='white', 
                                                   zorder=3., 
                                                   marker='*', 
                                                   s=400 
                                                   ) 
                            x_shift += 1 
                            counter += 1 
                            if m_i in [11, 33]: 
                                x_shift += 1 
                                counter += 1 
                        print('nonshaded {}'.format(nonshaded))
                        ### relative p-values 
                        corr_ps = mne.stats.fdr_correction([v[2] for v in ps])[1] 
                        corr_ps = [(ps[i][0], ps[i][1], ps[i][2], ps[i][3]) for i, p in enumerate(corr_ps) if type(ps[i][1])==tuple] 
                        x_shift = 0 
                        counter = -1 
                        for m_i, m in enumerate(sorted_models): 
                            counter += 1 
                            for c_i, c in enumerate(conds): 
                                for pm, pc, pp, t in corr_ps: 
                                    alpha = 1.
                                    if pm == m and pc[0] == c: 
                                        for other_i, other in enumerate(conds[1:]): 
                                            if pc[1] == other: 
                                                if 'length' in m:
                                                    color=colors['Word\nlength'][other_i+1]
                                                elif 'freq' in m:
                                                    color=colors['Word\nfrequency'][other_i+1]
                                                else:
                                                    color=colors[m][c_i]
                                                if pp < 0.05 and abs(t)>lo_e: 
                                                    ax.vlines( 
                                                             ymin=0.41-(other_i*0.03),  
                                                              ymax=0.41-(other_i*0.03)-0.01,  
                                                              x=m_i+corrections[c_i]+x_shift, 
                                                              color=color, 
                                                              alpha=alpha,
                                                              zorder=3.,
                                                              linewidth=5. 
                                                              ) 
                                                    ax.vlines( 
                                                              ymin=0.41-(other_i*0.03),  
                                                              ymax=0.41-(other_i*0.03)-0.01,  
                                                              x=m_i+corrections[c_i+1+other_i]+x_shift, 
                                                              color=color,
                                                              alpha=alpha,
                                                              zorder=3.,
                                                              linewidth=5. 
                                                              ) 
                                                    ax.hlines( 
                                                              xmin=m_i+corrections[c_i]+x_shift-0.025,  
                                                              xmax=m_i+corrections[c_i+1+other_i]+x_shift+0.03,  
                                                              y=0.41-(other_i*0.03), 
                                                              color=color,
                                                              zorder=3.,
                                                              alpha=alpha,
                                                              linewidth=5. 
                                                              ) 
                                                xmin=m_i+corrections[c_i]+x_shift  
                                                xmax=m_i+corrections[c_i+1+other_i]+x_shift  
                                                middle = xmin + ((xmax-xmin)*.5) 
                                                if pp < 0.005 and abs(t)>lo_e: 
                                                    if abs(t) > hi_e:
                                                        ax.text(
                                                            middle-0.275, 
                                                           y=0.42-(other_i*0.03), 
                                                           s='L',
                                                           fontweight='bold',
                                                           fontsize=15,
                                                           color=color,
                                                           ha='center',
                                                           va='center',
                                                           )
                                                    elif abs(t) > mid_e:
                                                        ax.text(
                                                            middle-0.275, 
                                                           y=0.42-(other_i*0.03), 
                                                           s='M',
                                                           fontweight='bold',
                                                           fontsize=15,
                                                           ha='center',
                                                           va='center',
                                                            color=color, 
                                                           )
                                                    elif abs(t) > lo_e:
                                                        ax.text(
                                                            middle-0.275, 
                                                           y=0.42-(other_i*0.03), 
                                                           s='S',
                                                           fontweight='bold',
                                                           fontsize=15,
                                                           ha='center',
                                                           va='center',
                                                            color=color, 
                                                           )
                                                    print(pp) 
                                                    ax.scatter( 
                                                               middle-0.075, 
                                                               y=0.42-(other_i*0.03), 
                                                               color=color, 
                                                               edgecolor='gray', 
                                                               zorder=3., 
                                                               marker='*',
                                                               alpha=alpha,
                                                               s=300 
                                                               ) 
                                                    ax.scatter( 
                                                               middle+0.075, 
                                                               y=0.42-(other_i*0.03), 
                                                               color=color, 
                                                               edgecolor='gray', 
                                                               alpha=alpha,
                                                               zorder=3., 
                                                               marker='*', 
                                                               s=300 
                                                               ) 
                                                elif pp < 0.05 and abs(t)>lo_e: 
                                                    if abs(t) > hi_e:
                                                        ax.text(
                                                           middle-0.15,
                                                           y=0.42-(other_i*0.03), 
                                                           s='L',
                                                           fontweight='bold',
                                                           fontsize=15,
                                                           ha='center',
                                                           va='center',
                                                           color=color,
                                                           )
                                                    elif abs(t) > mid_e:
                                                        ax.text(
                                                            middle-0.15,
                                                           y=0.42-(other_i*0.03), 
                                                           s='M',
                                                           color=color,
                                                           fontweight='bold',
                                                           fontsize=15,
                                                           ha='center',
                                                           va='center',
                                                           )
                                                    elif abs(t) > lo_e:
                                                        ax.text(
                                                            middle-0.15,
                                                           y=0.42-(other_i*0.03), 
                                                           s='S',
                                                           color=color,
                                                           fontweight='bold',
                                                           fontsize=15,
                                                           ha='center',
                                                           va='center',
                                                           )
                                                    print(pp) 
                                                    ax.scatter( 
                                                               middle, 
                                                               y=0.42-(other_i*0.03), 
                                                               color=color, 
                                                               edgecolor='gray', 
                                                               alpha=alpha,
                                                               zorder=3., 
                                                               marker='*', 
                                                               s=300 
                                                               ) 
                            x_shift += 1 
                            counter += 1 
                            if m_i in [11, 33]: 
                                x_shift += 1 
                                counter += 1 
                        ax.set_ylim(bottom=-.1, top=.43) 
                        ax.hlines(xmin=-.8, xmax=len(sorted_models)+x_shift-1.2, color='black', y=0) 
                        ax.hlines(xmin=-.8, xmax=len(sorted_models)+x_shift-1.2, color='silver',alpha=0.5,linestyle='dashed', y=[y*0.01 for y in range(-5, 40, 5)], zorder=1) 
                        pyplot.ylabel('Spearman correlation (RT-model)', fontsize=23) 
                        pyplot.xticks( 
                                      [x[0] for x in xticks],  
                                      [x[1] for x in xticks], 
                                      fontsize=25, 
                                      fontweight='bold') 
                        print(curr_fold) 
                        pyplot.savefig(os.path.join(curr_fold, '{}.jpg'.format(case)), dpi=300) 
                        pyplot.savefig(os.path.join(curr_fold, '{}.svg'.format(case)),) 
                        pyplot.clf() 
                        pyplot.close() 
                        ### writing to file
                        with open(os.path.join(curr_fold, '{}.txt'.format(case)), 'w') as o:
                            o.write('mode\tcorpus\tlang\ttask\tcase\tmodel\tcondition\t')
                            o.write('comparisons\t')
                            o.write('raw_p\tfdr_corrected_p_val\tt_val\tx_bar\tx_label\tperms_avg\tperms\trel_idx\n')
                            for lin in lines:
                                for li in lin:
                                    o.write('{}\t'.format(str(li).replace('\n', '$')))
                                o.write('\n')

font_folder = '../../fonts'
font_setup(font_folder)

plot_info = {
             'bootstrap' : [
                  {
                  'figsize' : (16, 10),
                  'correction' : (.7, 1.),
                  'jump' : 6.5,
                  'font_size' : 16,
                  'title_size' : 25,
                  'width' : (0.4, 0.4),
                  'xticks' : [2., 7.5], 
                  'xlabels' : ['Picture naming w/\ninterference', 'Semantic production'],
                  },
                  {
                  'figsize' : (16, 10),
                  'correction' : (.7, .7, .7),
                  'jump' : 6,
                  'font_size' : 15,
                  'title_size' : 25,
                  'width' : (0.4, 0.4, 0.4),
                  'xticks' : [1.5, 7.5, 13.5], 
                  'xlabels' : ['Semantic relatedness\njudgement', 'Action feature\njudgement', 'Sound feature\njudgement'],
                  },
                  #{
                  #'figsize' : (8, 10),
                  #'correction' : (.7,),
                  #'jump' : 0.,
                  #'font_size' : 16,
                  #'title_size' : 25,
                  #'width' : (0.4,),
                  #'xticks' : [1.5], 
                  #'xlabels' : ['Semantic relatedness\njudgement'],
                  #},
                  {
                  'figsize' : (16, 10),
                  'correction' : (1., 1.),
                  'jump' : 8,
                  'font_size' : 15,
                  'title_size' : 25,
                  'width' : (0.4, 0.4),
                  'xticks' : [2., 10.], 
                  'xlabels' : ['Semantic priming\n(quantity)', 'Semantic priming\n(social)'],
                  },
                            ], 
             'residualize' : [
                  {
                  'figsize' : (16, 10),
                  'correction' : (.75, .9, .75, .75, .75, .9, .9),
                  'jump' : 2.,
                  'font_size' : 15,
                  'title_size' : 20,
                  'width' : (0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
                  'xticks' : [0., 2., 4., 6., 8., 10., 12.], 
                  'xlabels' : ['Picture\nnaming w/\ninterference', 'Semantic\nproduction','Semantic\nrelatedness\njudgement', 'Action\nfeature\njudgement', 'Sound\nfeature\njudgement', 'Semantic\npriming\n(quantity)', 'Semantic\npriming\n(social)'],
                  },
                             ],
             }

corpus = 'wac'
plots = {
         'bootstrap' : [
                        ('pmtg-production','sem-phon'),
                        ('distr-learn', 'sound-action_A', 'sound-action_S'),
                        ('social-quantity_quantity', 'social-quantity_social'),
                        ],
         'residualize' : [
                        ('pmtg-production', 'sem-phon',
                        'distr-learn',
                        'sound-action_A', 'sound-action_S',
                        'social-quantity_quantity', 'social-quantity_social',
                         ),
                          ],
         }

results = dict()
fold = os.path.join('plots', 'corrected', 'ppmi')
for root, direc, fz in os.walk(fold):
    for f in fz:
        if 'txt' not in f:
            continue
        splt_root = root.split('/')
        if 'wac' not in splt_root:
            continue
        dataset = splt_root[-2]
        if 'sound' in dataset:
            full_task = '{}_{}'.format(dataset, splt_root[-1][0])
        elif 'social' in dataset:
            full_task = '{}_{}'.format(dataset, splt_root[-1].split('-')[0])
        else:
            full_task = '{}'.format(dataset)
        approach = splt_root[3]
        if approach not in results.keys():
            results[approach] = dict()
        if full_task not in results[approach].keys():
            results[approach][full_task] = list()
        with open(os.path.join(root, f)) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                if l_i == 0:
                    header = line.copy()
                    #print(header)
                    continue
                ### just checking
                task = line[header.index('task')]
                case = line[header.index('case')]
                if 'sound' in task:
                    new_full_task = '{}_{}'.format(task, case[0])
                elif 'social' in task:
                    new_full_task = '{}_{}'.format(task, case.split('-')[0])
                else:
                    new_full_task = '{}'.format(task)
                assert new_full_task == full_task
                ### collecting results
                cond = line[header.index('condition')]
                model = line[header.index('model')]
                p_raw = float(line[header.index('raw_p')])
                p_fdr = float(line[header.index('fdr_corrected_p_val')])
                t_val = float(line[header.index('t_val')])
                bar_y = float(line[header.index('perms_avg')])
                bar_x = float(line[header.index('x_bar')])
                rel_idx = int(line[header.index('rel_idx')])
                label_x = float(line[header.index('x_label')])
                perms = [float(val) for val in line[header.index('perms')].split(',')]
                assert len(perms) in [1000, 10000]
                #print([model, cond])
                comparisons = [val.split('@')[0].split('_') for val in line[header.index('comparisons')].split('#')]
                curr_dict = {
                             'cond' : cond,
                             'model' : model,
                             'p_raw' : p_raw,
                             'p_fdr' : p_fdr,
                             't_val' : t_val,
                             'bar_y' : bar_y,
                             'bar_x' : bar_x,
                             'label_x' : label_x,
                             'rel_idx' : rel_idx,
                             'perms' : perms,
                             'comparisons' : comparisons,
                             }
                print(os.path.join(root, f))
                results[approach][full_task].append(curr_dict)
### just checking...
for a, a_r in results.items():
    for t, t_r in a_r.items():
        assert len(t_r) in [6, 9]
out_f = os.path.join('plots', 'main_text')
os.makedirs(out_f, exist_ok=True)
for plot, cases in plots.items():
    colors = set_colors(plot)
    plot_n = 0
    for datasets in cases:
        print(datasets)
        plot_n += 1
        figsize = plot_info[plot][plot_n-1]['figsize']
        out = os.path.join(out_f, 'R{:02}_{}.jpg'.format(plot_n, plot))
        #figsize=(13, 10)
        fig, ax = pyplot.subplots(figsize=figsize, constrained_layout=True)
        ax.set_ylim(bottom=-.09, top=.42) 
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        pyplot.ylabel('Spearman correlation (RT-model)', fontsize=23) 
        pyplot.yticks(fontsize=20)
        lgnds = list()
        comps = list()
        mapper = dict()
        xs = list()
        for start_i, dataset in enumerate(datasets):
            all_data = results[plot][dataset]
            w = plot_info[plot][plot_n-1]['width'][start_i]
            jump = plot_info[plot][plot_n-1]['jump']
            corr = plot_info[plot][plot_n-1]['correction'][start_i]
            t_s = plot_info[plot][plot_n-1]['font_size']
            title_size = plot_info[plot][plot_n-1]['title_size']
            m_sc = 1000
            start = start_i * jump
            for data in all_data:
                label = data['cond']
                model = data['model']
                if plot == 'residualize' and ('length' in model or 'frequency' in model):
                    continue
                p_abs = data['p_fdr']
                t_abs = data['t_val']
                perms = data['perms']
                rel_idx = data['rel_idx']
                color = colors[model][rel_idx]
                if rel_idx == 0:
                    if plot == 'residualize':
                        label_mapper = {
                                        'GPT2$surprisal' : 'GPT2 surprisal',
                                        'wac$PPMI' : 'Semantic dissimilarity (beyond word length and freq.)',
                                        'Word$frequency$(neg. sum)' : 'Word frequency (neg.)',
                                        'Word$length$(sum)': 'Word length',
                                        }
                    else:
                        label_mapper = {
                                        'GPT2$surprisal' : 'GPT2 surprisal',
                                        'wac$PPMI' : 'Semantic dissimilarity',
                                        #'Word$frequency' : 'Word frequency (neg.)',
                                        #'Word$length': 'Word length',
                                        'Word$frequency$(neg. sum)' : 'Word frequency (neg.)',
                                        'Word$length$(sum)': 'Word length',
                                        }
                    if 'distr' not in dataset:
                        if start_i in [0, 2] and ('length' in model or 'frequency' in model):
                            pass
                        else:
                            legend_label = label_mapper[model]
                            if legend_label not in lgnds:
                                lgnds.append(legend_label)
                                lgnd_w = 2.
                                ax.bar(0, 0, color=color, label=legend_label)
                    else:
                        legend_label = label_mapper[model]
                        if legend_label not in lgnds:
                            lgnds.append(legend_label)
                            lgnd_w = 0.6
                            ax.bar(0, 0, color=color, label=legend_label)
                x_bar = data['bar_x']*corr
                xs.append(x_bar+start)
                x_label = data['label_x']*corr
                mapper[(label, model, dataset)] = (x_bar+start, rel_idx)
                for cmpr in data['comparisons']:
                    if cmpr != ['na']:
                        print(cmpr)
                        p_comp = float(cmpr[-2])
                        t_comp = float(cmpr[-1])
                        if p_comp < 0.05 and abs(t_comp)>lo_e:
                            comps.append([
                                          label,
                                          model,
                                          dataset,
                                          cmpr[0], 
                                          cmpr[1], 
                                          p_comp,
                                          t_comp,
                                          ])
                ax.bar(
                       x_bar+(start), 
                       numpy.average(perms),
                       width=w, 
                       color=color, 
                       edgecolor='gray', 
                       zorder=1.5 
                       )
                ax.bar(
                       x_bar+(start), 
                       numpy.average(perms),
                       width=w, 
                       fill=False,
                       edgecolor='dimgray', 
                       zorder=2.5 
                       )
                ax.errorbar(
                       x_bar+(start), 
                       numpy.average(perms),
                       yerr=numpy.std(perms),
                       color='dimgray', 
                       capsize=5, 
                       zorder=3. 
                       )
                ax.scatter( 
                       [x_bar+start+(random.randrange(-m_sc, m_sc)*0.0001) for rand in range(len(perms))],  
                       perms, 
                       color=color, 
                       edgecolor='white', 
                       alpha=0.2, 
                       zorder=2. 
                       ) 
                if rel_idx == 0:
                    ax.text( 
                           x_label+start,  
                           -.02, 
                           s=label.replace('$', '\n'), 
                           fontsize=t_s, 
                            #fontweight='bold',
                           ha='center', 
                           va='center', 
                            )
                ### absolute p-value
                if p_abs < 0.005 and abs(t_abs)>lo_e:
                    '''
                    if abs(t_abs) > 0.8:
                        ax.text(
                           x_bar+start,  
                           0.035, 
                           s='H',
                           fontweight='bold',
                           fontsize=15,
                           color='black', 
                           ha='center',
                           va='center',
                           )
                    elif abs(t_abs) > 0.5:
                        ax.text(
                           x_bar+start,  
                           0.035, 
                           s='M',
                           fontweight='bold',
                           fontsize=15,
                           ha='center',
                           va='center',
                           color='black', 
                           )
                    '''
                    ax.scatter( 
                               x_bar+start-0.08,  
                               0.02, 
                               color='black', 
                               edgecolor='white', 
                               zorder=3., 
                               marker='*', 
                               s=200 
                               ) 
                    ax.scatter( 
                               x_bar+start+0.08,  
                               0.02, 
                               color='black', 
                               edgecolor='white', 
                               zorder=3., 
                               marker='*', 
                               s=200 
                               ) 
                elif (str(p_abs)[:4] == '0.05' or p_abs<0.05) and abs(t_abs)>lo_e:
                    '''
                    if abs(t_abs) > 0.8:
                        ax.text(
                           x_bar+start,  
                           0.035, 
                           s='H',
                           fontweight='bold',
                           fontsize=15,
                           color='black', 
                           ha='center',
                           va='center',
                           )
                    elif abs(t_abs) > 0.5:
                        ax.text(
                           x_bar+start,  
                           0.035, 
                           s='M',
                           fontweight='bold',
                           fontsize=15,
                           ha='center',
                           va='center',
                           color='black', 
                           )
                    '''
                    ax.scatter( 
                               x_bar+start,  
                               0.02, 
                               color='black', 
                               edgecolor='white', 
                               zorder=3., 
                               marker='*', 
                               s=200 
                               ) 
        ### comparative p-value
        for l, m, d, start, end, p, t in comps:
            st, st_idx = mapper[(start, m, d)]
            nd, nd_idx = mapper[(end, m, d)]
            ax.vlines( 
                     ymin=0.39-(nd_idx*0.03),  
                      ymax=0.39-(nd_idx*0.03)-0.01,  
                      x=st, 
                      color=colors[m][nd_idx], 
                      #alpha=alpha,
                      zorder=3.,
                      linewidth=5. 
                      ) 
            ax.vlines( 
                      ymin=0.39-(nd_idx*0.03),  
                      ymax=0.39-(nd_idx*0.03)-0.01,  
                      x=nd, 
                      color=colors[m][nd_idx],
                      #alpha=alpha,
                      zorder=3.,
                      linewidth=5. 
                      ) 
            ax.hlines( 
                      xmin=st-0.025,  
                      xmax=nd+0.03,  
                      y=0.39-(nd_idx*0.03), 
                      color=colors[m][nd_idx],
                      zorder=3.,
                      #alpha=alpha,
                      linewidth=5. 
                      ) 
            if p < 0.005:
                if abs(t) > hi_e:
                    ax.text(
                       st+((nd-st)*.5)-0.28,  
                       .3975-(nd_idx*0.03), 
                       s='L',
                       fontweight='bold',
                       fontsize=15,
                       color=colors[m][0], 
                       ha='center',
                       va='center',
                       )
                elif abs(t) > mid_e:
                    ax.text(
                       st+((nd-st)*.5)-0.28,  
                       .3975-(nd_idx*0.03), 
                       s='M',
                       fontweight='bold',
                       fontsize=15,
                       ha='center',
                       va='center',
                       color=colors[m][0], 
                       )
                elif abs(t) > lo_e:
                    ax.text(
                       st+((nd-st)*.5)-0.28,  
                       .3975-(nd_idx*0.03), 
                       s='S',
                       fontweight='bold',
                       fontsize=15,
                       ha='center',
                       va='center',
                       color=colors[m][0], 
                       )
                ax.scatter( 
                       st+((nd-st)*.5)-0.08,  
                       .3975-(nd_idx*0.03), 
                       color=colors[m][0], 
                       edgecolor='gray', 
                       zorder=3., 
                       marker='*', 
                       s=200 
                       ) 
                ax.scatter( 
                       st+((nd-st)*.5)+0.08,  
                       .3975-(nd_idx*0.03), 
                       color=colors[m][0], 
                       edgecolor='gray', 
                       zorder=3., 
                       marker='*', 
                       s=200 
                       ) 
            elif p < 0.05:
                if abs(t) > hi_e:
                    ax.text(
                       st+((nd-st)*.5)-0.2,  
                       .3975-(nd_idx*0.03), 
                       s='L',
                       fontweight='bold',
                       fontsize=15,
                       color=colors[m][0], 
                       ha='center',
                       va='center',
                       )
                elif abs(t) > mid_e:
                    ax.text(
                       st+((nd-st)*.5)-0.2,  
                       .3975-(nd_idx*0.03), 
                       s='M',
                       fontweight='bold',
                       fontsize=15,
                       ha='center',
                       va='center',
                       color=colors[m][0], 
                       )
                elif abs(t) > lo_e:
                    ax.text(
                       st+((nd-st)*.5)-0.2,  
                       .3975-(nd_idx*0.03), 
                       s='S',
                       fontweight='bold',
                       fontsize=15,
                       ha='center',
                       va='center',
                       color=colors[m][0], 
                       )
                ax.scatter( 
                       st+((nd-st)*.5),  
                       .3975-(nd_idx*0.03), 
                       color=colors[m][0], 
                       edgecolor='gray', 
                       zorder=3., 
                       marker='*', 
                       s=200 
                       ) 
        ax.hlines(
                  xmin=min(xs), 
                  xmax=max(xs), 
                  color='black', 
                  y=0,
                  ) 
        ax.hlines(
                  xmin=min(xs), 
                  xmax=max(xs), 
                  color='silver',
                  alpha=0.5,
                  linestyle='dashed', 
                  y=[y*0.01 for y in range(-5, 40, 5)], 
                  zorder=1,
                  ) 
        ax.legend(ncol=4, loc=9, columnspacing=lgnd_w, fontsize=17)
        ax.set_xticks(
                      plot_info[plot][plot_n-1]['xticks'],
                      plot_info[plot][plot_n-1]['xlabels'],
                      fontsize = title_size,
                      fontweight = 'bold',
                      linespacing=1.5,
                      #fontstyle='italic',
                      )
        ax.tick_params(axis='x',pad=30)
        pyplot.savefig(out, dpi=600)
        pyplot.clf()
        pyplot.close()
