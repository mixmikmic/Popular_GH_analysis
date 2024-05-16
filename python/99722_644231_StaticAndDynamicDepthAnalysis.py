import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

get_ipython().magic('matplotlib inline')

from matplotlib import rc

# Enables full LaTeX support in the plot text.
# Requires a full-fledged LaTeX installation on your system, accessible via PATH.
rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (16, 5)
matplotlib.rcParams.update({'font.size': 20})

def gen_plots(root, part, out_dir, **kw):
    file_pattern = 'k-99999-kitti-odometry-{sequence_id:02d}-offset-0-depth-precomputed-{depth}-'                    'voxelsize-0.0500-max-depth-m-20.00-dynamic-mode-NO-direct-ref-'                    'with-fusion-weights-{part}.csv'
    base = os.path.join(root, file_pattern)
    save_to_disk = kw.get('save_to_disk', True)
    
    sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metrics = ['input', 'fusion']
    depths = ['elas', 'dispnet']
    
    # Load the data and prepare some positioning and color info
    # (Since we're doing a nontrivial plot, which means we need to do a lot of
    #  stuff manually.)
    box_positions = []
    box_colors = []
    columns = []
    box_offset = 0.0
    INNER_GAP = 0.75
    SEQUENCE_GAP = 1.0
    GROUP_SIZE = len(depths) * len(metrics)
    
    colors = {
        'elas': {
            'input': 'C0',
            'fusion': 'C1',
        },
        'dispnet': {
            'input': 'C2',
            'fusion': 'C3'
        }
    }
    
    def setup_xaxis_legend(ax, **kw):
        bp_np = np.array(box_positions)
        alt_ticks = bp_np[np.arange(len(bp_np)) % GROUP_SIZE == 0] + (INNER_GAP*(GROUP_SIZE-1.0)/2.0)
        ax.set_xticks(alt_ticks)
        ax.set_xticklabels("{:02d}".format(sid) for sid in sequences)
        ax.set_xlabel("Sequence")

        ax.set_ylim([0.0, 1.0])

        for patch, color in zip(boxplot['medians'], box_colors):
            patch.set_color(color)    

        for patch, color in zip(boxplot['boxes'], box_colors):
            patch.set_color(color)

        # Ugly, but required since every box has two whiskers and two caps...
        for idx, (whisker, cap) in enumerate(zip(boxplot['whiskers'], boxplot['caps'])):
            cap.set_color(box_colors[idx%(2*GROUP_SIZE) // 2])
            whisker.set_color(box_colors[idx%(2*GROUP_SIZE) // 2])   

        # Dummies for showing the appropriate legend
        ax.plot([0.0], [-1000], label="ELAS input", color=colors['elas']['input'])
        ax.plot([0.0], [-1000], label="ELAS fused", color=colors['elas']['fusion'])
        ax.plot([0.0], [-1000], label="DispNet input", color=colors['dispnet']['input'])
        ax.plot([0.0], [-1000], label="DispNet fused", color=colors['dispnet']['fusion'])
        ax.legend(loc=kw.get('legendloc', 'lower left'))

        ax.grid('off')
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)
        
    def save_fig(f, fname):
        print("Saving figure to [{}]... ".format(fname), end='')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        f.savefig(os.path.join(out_dir, fname + '.png'), bbox_inches='tight')
        f.savefig(os.path.join(out_dir, fname + '.eps'), bbox_inches='tight')
        print("\rSaved figure to [{}].    ".format(fname))
        
    def compute_metrics(dataframe, metric):               
        # Do not count frames with no pixels in them. This would distort the 
        # dynamic reconstruction metrics due to frames containing no objects.
        ok = (dataframe['{}-total-3.00-kitti'.format(metric)] != 0)

        err = dataframe['{}-error-3.00-kitti'.format(metric)][ok]
        tot = dataframe['{}-total-3.00-kitti'.format(metric)][ok]
        mis = dataframe['{}-missing-3.00-kitti'.format(metric)][ok]
        cor = dataframe['{}-correct-3.00-kitti'.format(metric)][ok]
        mis_sep = dataframe['{}-missing-separate-3.00-kitti'.format(metric)][ok]

        acc_perc = cor / (tot - mis)
        completeness = 1.0 - (mis_sep / tot)
        
        return acc_perc, completeness
    
    def setup_agg_plot(ax, boxplot):
        # Aesthetic crap
        ax.set_ylim([0.4, 1.01])
        plt.minorticks_on()
        plt.xticks(rotation=45, ha='right')

        for patch in boxplot['medians']:
            patch.set_color('black')
        for patch in boxplot['boxes']:
            patch.set_color('black')
        for patch in boxplot['whiskers']:
            patch.set_color('black')

        ax.set_xticklabels(["ELAS input", "ELAS fused", "DispNet input", "DispNet fused"])
        ax.grid('off')
        ax.yaxis.grid(True, linestyle='-', which='major', color='gray', alpha=0.75)
        ax.yaxis.grid(True, linestyle='-', which='minor', color='lightgrey', alpha=0.75)
    
    res = {}
    res_completeness = {}
    
    # Aggregated for all the sequences.
    res_acc_agg = {}
    res_completeness_agg = {}
    
    for sequence_id in sequences:
        for depth in depths:
            # Part dictates what we are evaluating: dynamic or static parts
            fname = base.format(sequence_id=sequence_id, depth=depth, part=part)
            df = pd.read_csv(fname)
#             print("{} frames in sequence #{}-{}".format(len(df), sequence_id, depth))
            
            for metric in metrics:
                key = "{}-{}-{:02d}".format(metric, depth, sequence_id)
                agg_key = "{}-{}".format(metric, depth)
                if not agg_key in res_acc_agg:
                    res_acc_agg[agg_key] = []
                    res_completeness_agg[agg_key] = []
                
                acc_perc, completeness = compute_metrics(df, metric)
                res[key] = acc_perc
                res_completeness[key] = completeness
                res_acc_agg[agg_key] = res_acc_agg[agg_key] + acc_perc.tolist()
                res_completeness_agg[agg_key] = res_completeness_agg[agg_key] + completeness.tolist()
                
                box_colors.append(colors[depth][metric])
                
                columns.append(key)
                box_positions.append(box_offset)
                box_offset += INNER_GAP
            
        box_offset += SEQUENCE_GAP
                
#     res_acc_all = [entry for (key, sublist) in res.items() for entry in sublist]
        
    print("Data read & aggregated OK.")
    
    print("Agg meta-stats:")
    for k, v in res_acc_agg.items():
        print(k, len(v))
    
    
    ################################################################################
    # Accuracy plots
    ################################################################################
    res_df = pd.DataFrame(res)    
    FIG_SIZE = (16, 6)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    (ax, boxplot) = res_df.boxplot(columns, showfliers=False, 
                                   return_type='both',
                                   widths=0.50, 
                                   ax=ax, 
#                                    patch_artist=True,  # Enable fill
                                   positions=box_positions)
    setup_xaxis_legend(ax)
    ax.set_ylabel("Accuracy", labelpad=15)
    ax.set_ylim([0.3, 1.01])
    if save_to_disk:
        save_fig(fig, 'odo-acc-{}'.format(part))
    
    ################################################################################
    # Aggregate accuracy plots
    ################################################################################
    res_acc_agg_df = pd.DataFrame(res_acc_agg)
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(1, 1, 1)
    agg_cols = ['input-elas', 'fusion-elas', 'input-dispnet', 'fusion-dispnet']
    (ax, boxplot) = res_acc_agg_df.boxplot(agg_cols, showfliers=False,
                                           return_type='both',
                                           widths=0.25,
                                           ax=ax)
    ax.set_ylabel("Accuracy", labelpad=15)
    setup_agg_plot(ax, boxplot)

    print("Textual results: ")
    for col in agg_cols:
        print(col, ":", res_acc_agg_df[col].mean())
    
    if save_to_disk:
        save_fig(fig, 'odo-acc-agg-{}'.format(part))
    
    ################################################################################
    # Completeness plots
    ################################################################################
    res_completeness_df = pd.DataFrame(res_completeness)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    
    (ax, boxplot) = res_completeness_df.boxplot(columns, showfliers=False, 
                                               return_type='both',
                                               widths=0.50, 
                                               ax=ax, 
            #                                    patch_artist=True,  # Enable fill
                                               positions=box_positions)
    
    setup_xaxis_legend(ax)
    ax.set_ylim([0.3, 1.01])
    ax.set_ylabel("Completeness")
    
    if save_to_disk:
        save_fig(fig, 'odo-completeness-{}'.format(part))
        
    ################################################################################
    # Aggregate completeness plots
    ################################################################################
    res_completeness_agg_df = pd.DataFrame(res_completeness_agg)
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(1, 1, 1)
    agg_cols = ['input-elas', 'fusion-elas', 'input-dispnet', 'fusion-dispnet']
    (ax, boxplot) = res_completeness_agg_df.boxplot(agg_cols, showfliers=False,
                                                    return_type='both',
                                                    widths=0.25,
                                                    ax=ax)
    ax.set_ylabel("Completeness", labelpad=15)
    setup_agg_plot(ax, boxplot)

    print("Textual results: ")
    for col in agg_cols:
        print(col, ":", res_completeness_agg_df[col].mean())
    
    if save_to_disk:
        save_fig(fig, 'odo-completeness-agg-{}'.format(part))
    
                
        
save = True
out_dir = '../fig'
gen_plots('../csv/odo-res', 'static-depth-result', out_dir, save_to_disk=save)
gen_plots('../csv/odo-res', 'dynamic-depth-result', out_dir, save_to_disk=save)





