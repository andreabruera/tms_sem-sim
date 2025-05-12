import matplotlib
import numpy
import pingouin
import random
import scipy

from matplotlib import font_manager
from tqdm import tqdm

def set_colors(plot):
    colors = { 
          'wac$surprisal' : (
                             'mediumorchid', 
                             'thistle', 
                             'plum',
                             ), 
          'GPT2$surprisal' : (
                             'mediumvioletred', 
                             'pink', 
                             'palevioletred',
                             ), 
          'Word$frequency$(neg. sum)' : (
                             'sienna', 
                             'peru', 
                             'sandybrown',
                             ), 
          'Word$length$(sum)' : (
                            'gray', 
                            'darkgray', 
                            'lightgray',
                            ), 
          } 
    if plot != 'residualize':
        colors['wac$PPMI'] = (
                          'seagreen', 
                          'mediumaquamarine', 
                          'mediumseagreen',
                          ) 
    else:
        colors['wac$PPMI'] = (
                            'navy',
                            'lightsteelblue', 
                            #'lightblue', 
                             'paleturquoise',
                             )
    return colors

def font_setup(font_folder):
    ### Font setup
    # Using Helvetica as a font
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

def read_effsizes(mode):
    if mode == 'classic':
        lo_e = 0.2
        mid_e = 0.5
        hi_e = 0.8
    elif mode == 'cog_neuro':
        ### taken from
        ### Empirical assessment of published effect sizes and 
        ### power in the recent cognitive neuroscience and psychology literature
        ### Denes Szucs ,John P. A. Ioannidis, PLOS ONE 2018
        lo_e = 0.637
        mid_e = 0.932
        hi_e = 1.458
    return lo_e, mid_e, hi_e

def perm_against_zero(distr):
    one = (sum([1 for _ in distr if _<0.])+1)/(len(distr)+1)
    two = (sum([1 for _ in distr if _>0.])+1)/(len(distr)+1)
    half_p = min(one, two)
    p = half_p * 2
    ci_constant = 1.96 * (numpy.std(distr)/numpy.sqrt(len(distr)))
    t = numpy.average(distr)/numpy.std(distr)
    return t, p, ci_constant

def permutation_one_sample(one, one_sided=True):
    one = one.tolist()
    assert len(one) in [1000,10000]
    ### permutation test
    if one_sided:
        real_diff = numpy.average(one)
    else:
        real_diff = abs(numpy.average(one))
    fake_distr = list()
    fakes = 1000
    for _ in tqdm(range(fakes)):
        fake_one = [o*random.choice([-1, +1]) for o in one]
        if one_sided:
            fake_diff = numpy.average(fake_one)
        else:
            fake_diff = abs(numpy.average(fake_one))
        fake_distr.append(fake_diff)
    ### p-value
    p_val = (sum([1 for _ in fake_distr if _>real_diff])+1)/(fakes+1)
    ### t-value
    # adapted from https://matthew-brett.github.io/cfd2020/permutation/permutation_and_t_test.html
    pers_errors = [v-numpy.average(one) for v in one]
    place_errors = [v-numpy.average(two) for v in fake_distr]
    all_errors = pers_errors + place_errors
    est_error_sd = numpy.sqrt(sum([er**2 for er in all_errors]) / (len(one) + len(fake_distr) - 2))
    sampling_sd_estimate = est_error_sd * numpy.sqrt(1 / len(one) + 1 / len(fake_distr))
    t_val = real_diff/sampling_sd_estimate

    return t_val, p_val

def permutation_two_samples(one, two):
    one = one.tolist()
    assert len(one) in [10000, 1000]
    two = two.tolist()
    assert len(two) in [1000, 10000]
    diff_one = numpy.average(one)/numpy.std(one)
    diff_two = numpy.average(two)/numpy.std(two)
    real_diff = abs(diff_one-diff_two)
    t_val = pingouin.compute_effsize(one, two, eftype='cohen', paired=True)
    ci = pingouin.compute_esci(stat=t_val, nx=1000, ny=1000, eftype='cohen', paired=True)
    ci_constant = ci[1]-real_diff
    fake_distr = list()
    fakes = 1000
    for _ in tqdm(range(fakes)):
        fake = random.sample(one+two, k=len(one+two))
        fake_one = fake[:int(len(fake)*.5)]
        fake_two = fake[int(len(fake)*.5):]
        fake_diff_one = numpy.average(fake_one)/numpy.std(fake_one)
        fake_diff_two = numpy.average(fake_two)/numpy.std(fake_two)
        fake_diff = abs(fake_diff_one-fake_diff_two)
        fake_distr.append(fake_diff)
    ### p-value
    p_val = (sum([1 for _ in fake_distr if _>real_diff])+1)/(fakes+1)

    return t_val, p_val, fake_distr, ci
