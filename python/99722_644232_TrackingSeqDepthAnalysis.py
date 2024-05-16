import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

get_ipython().magic('matplotlib inline')

from matplotlib import rc
# Enable full LaTeX support in plot text. Requires a full-fledged LaTeX installation
# on your system, accessible via PATH.
rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (16, 6)
matplotlib.rcParams.update({'font.size': 16})

out_dir = '../fig'

def gen_plots(root, part, eval_completeness):
    # if 'eval_completeness' is false, eval accuracy.
    # TODO maybe also compute results for dynamic parts.
    file_pattern = 'k-99999-kitti-tracking-sequence-{sequence_id:04d}--offset-0-'                       'depth-precomputed-{depth}-voxelsize-0.0500-max-depth-m-20.00-'                     '{fusion}-NO-direct-ref-with-fusion-weights-{part}.csv'
    base = os.path.join(root, file_pattern)
    res = {}
    res_completeness = {}
    
    sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fusions = ['NO-dynamic', 'dynamic-mode']
    metrics = ['input', 'fusion']
    depths = ['dispnet', 'elas']
    
    res = {}
    res_completeness = {}
    
    # Not a proper header. The real one is in the thesis tex, since it's nested
    # and pointless to generate from code.
    for depth in depths:
        for fusion in fusions:
            for metric in metrics:
                dyn_str = 'Dyn' if fusion == 'dynamic-mode' else 'No-dyn'
                if metric == 'input' and fusion == 'NO-dynamic':
                    print('Input-{} {} & '.format(depth, dyn_str), end='')
                    
                if metric == 'fusion':
                    print('Fusion-{} {} &'.format(depth, dyn_str), end='')
    print()
    
    acc_perc_agg = {}
    completeness_agg = {}
    
    # Yowza, that's a lot of loop nesting!
    for sequence_id in sequences:
        seq_count = -1
        
        print('{:02d} &'.format(sequence_id), end='')
        for depth in depths:
            best_key = None
            best_score = -1.0
            
            for fusion in fusions:
                fname = base.format(sequence_id=sequence_id, depth=depth, 
                                        fusion=fusion, part=part)
                df = pd.read_csv(fname)
            
                for metric in metrics:
                    key = "{}-{}-{}-{:02d}".format(metric, depth, fusion, sequence_id)
                    cross_seq_key = "{}-{}-{}".format(metric, depth, fusion)

                    # Do not count frames with no pixels in them. This would distort the 
                    # dynamic reconstruction metrics due to frames containing no objects.
                    ok = (df['{}-total-3.00-kitti'.format(metric)] != 0)

                    err = df['{}-error-3.00-kitti'.format(metric)][ok]
                    tot = df['{}-total-3.00-kitti'.format(metric)][ok]
                    mis = df['{}-missing-3.00-kitti'.format(metric)][ok]
                    cor = df['{}-correct-3.00-kitti'.format(metric)][ok]
                    mis_sep = df['{}-missing-separate-3.00-kitti'.format(metric)][ok]

                    acc_perc = cor / (tot - mis)
                    # When evaluating dynamic parts, sometimes we encounter cases with
                    # e.g., very distant cars where tot == mis.
                    acc_perc = acc_perc[~np.isnan(acc_perc)]
                    completeness = 1.0 - (mis_sep / tot)
                    
                    if cross_seq_key not in acc_perc:
                        acc_perc_agg[cross_seq_key] = []
                        completeness_agg[cross_seq_key] = []
                        
                    acc_perc_agg[cross_seq_key] += acc_perc.tolist()
                    completeness_agg[cross_seq_key] += completeness.tolist()
                    
                    if eval_completeness:
                        res[key] = completeness
                    else:
                        res[key] = acc_perc
                    
                    
                    mean_acc_perc = acc_perc.mean()
                    mean_com_perc = completeness.mean()
                    
                    # The input should be the same in dynamic and non-dynamic mode.
                    if not (metric == 'input' and fusion == 'dynamic-mode'):
                        if eval_completeness:
                            # Compute and display completeness
                            if mean_com_perc > best_score:
                                best_score = mean_com_perc
                                best_key = key
                        else:
                            # Compute and display accuracy
                            if mean_acc_perc > best_score:
                                best_score = mean_acc_perc
                                best_key = key
                    
                    if -1 == seq_count:
                        seq_count = len(df)
                    elif seq_count != len(df):
                        print("Warning: inconsistent lengths for sequence {:04d}".format(sequence_id))
                        print(sequence_id, depth, fusion, metric, len(df))
                  
            for fusion in fusions:
                for metric in metrics:
                    key = "{}-{}-{}-{:02d}".format(metric, depth, fusion, sequence_id)

                    if not (metric == 'input' and fusion == 'dynamic-mode'):
                        if res[key].mean() is np.nan:
                            # No data for the dynamic parts when doing standard fusion!
                            assert(fusion == 'NO-dynamic')
                            continue
                        elif key == best_key:
                            print(r'\textbf{{{:.4f}}}'.format(res[key].mean()), end='')
                        else:
                            print(r'        {:.4f}   '.format(res[key].mean()), end='')
                            
                        if not (metric == 'fusion' and fusion == 'dynamic-mode'):
                            print('& ', end='')
                     
            if depth == depths[0]:
                print('&', end='\n    ')
            
        print(r'\\')
        
    print("\n\n")
    for metric in metrics:
        for depth in depths:
            for fusion in fusions:
                key = "{}-{}-{}".format(metric, depth, fusion)
                acc_perc = acc_perc_agg[key]
                completeness = completeness_agg[key]
                print(key, len(acc_perc), len(completeness))
                print("Mean accuracy: {}, Mean completeness: {}".format(np.mean(acc_perc), np.mean(completeness)))

#                 box_colors.append(colors[depth][metric])
#                 columns.append(key)
#                 box_positions.append(box_offset)
    
gen_plots('../csv/tracking-res/', 'static-depth-result', eval_completeness=False)
# gen_plots('../csv/tracking-res/', 'dynamic-depth-result')

def gen_baseline_data(root, part, eval_completeness):
    print("TODO(andrei): Same as above but for the InfiniTAM baseline.")

