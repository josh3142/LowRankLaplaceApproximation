import os
import glob
import re
from typing import List, Literal

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from color_map import get_color

# %%
#  hyperparameters
# for loading
p_methods = ['lowrank-kron', 'lowrank-diag', 'subset-diag','subset-magnitude', 'subset-swag',
             'lowrankoptimal-loadfile']
psi_ref = 'loadfile'
# for plotting
markersizes = [3,4,5.5]
markertypes = {'lowrank': 'o', 'subset': 's', 'lowrankoptimal': '^'}
delta_x_list = [-0.3,0.0, 0.3]
fontsize = 18

# %%
# to format labels in the plots (removes underscores)
def remove_underscores(str_list: List[str]) -> List[str]:
    formatted_str_list = []
    for item in str_list:
        formatted_str_list.append(item.replace('_',' '))
    return formatted_str_list

# %%
# collect keywords from files
pathname = os.path.join('results','mnist_c','cnn_small')
seed_folders = glob.glob(os.path.join(pathname, 'seed*'))
corruption_files = [os.path.basename(file) for file in glob.glob(os.path.join(pathname,
                                          os.path.basename(seed_folders[-1]),
                                          'Metrics_lowrank-kron_Psiloadfile_c-*'))]
corruptions = [re.search('_c-(.*)\.pt$',file).group(1) for file in corruption_files]

# %%
# templates for filenames
def metrics_file(
    seed_folder: str,
    p_method: str,
    psi_ref: str,
    corruption: str
) -> str:
    return os.path.join(seed_folder,
            f'Metrics_{p_method}_Psi{psi_ref}_c-{corruption}.pt')


def nll_file(
    seed_folder: str,
    p_method: str,
    psi_ref: str,
    corruption: str
) -> str:
    return os.path.join(seed_folder,
            f'nll_{p_method}_Psi{psi_ref}_c-{corruption}.pt')

def plot_file(metric: Literal['rel_error', 'trace', 'nll']) -> str:
    return os.path.join(pathname, f'{metric}.png')

# %%
# load s list
with open(
    metrics_file(seed_folders[0], p_methods[0], psi_ref, corruptions[0]),
    'rb'
) as f:
    s_list = torch.load(f)['s_list']

# %% 
# collect results
rel_error_results = {}
trace_results = {}
nll_results = {}
for i, s in enumerate(s_list):
    rel_error_results[s] = {}
    trace_results[s] = {}
    nll_results[s] = {}
    for corruption in corruptions:
        rel_error_results[s][corruption] = {}
        trace_results[s][corruption] = {}
        nll_results[s][corruption] = {}
        for p_method in p_methods:
            rel_error_results[s][corruption][p_method] = []
            trace_results[s][corruption][p_method] = []
            nll_results[s][corruption][p_method] = []
            for seed_folder in seed_folders:
                with open(
                    metrics_file(seed_folder, p_method, psi_ref, corruption), 
                    'rb'
                ) as f:
                    metrics = torch.load(f)
                rel_error_results[s][corruption][p_method].append(metrics['rel_error'][i])
                trace_results[s][corruption][p_method].append(metrics['trace'][i])
                with open(
                    nll_file(seed_folder, p_method, psi_ref, corruption), 
                    'rb'
                ) as f:
                    nll_metrics = torch.load(f)
                nll_results[s][corruption][p_method].append(nll_metrics['nll'][i])
            
# %%
# plot results
mpl.rcParams['font.size'] = fontsize
print('relative error results')
plt.figure(1)
plt.clf()
for i, (s, markersize, delta_x) in enumerate(zip(s_list, markersizes, delta_x_list)):
    for p_method in p_methods:
        methodclass, ptype = p_method.split('-')
        mean_values = []
        std_values = []
        x = 1.0 * np.arange(len(corruptions)) + delta_x
        for corruption in corruptions:
            mean_values.append(np.mean(rel_error_results[s]
                [corruption][p_method])
            )
            std_values.append(np.std(rel_error_results[s]
            [corruption][p_method])
                /np.sqrt(len(trace_results[s][corruption][p_method]))
            )
        plt.errorbar(x=x, y=mean_values, yerr=std_values, color=get_color(ptype),
                    fmt=markertypes[methodclass], 
                    markersize=markersize,
                    alpha=.5)
    plt.xticks(x, remove_underscores(corruptions), rotation=90)
    plt.ylabel(r'Relative Error')
    plt.tight_layout()
plt.savefig(plot_file('rel_error'), bbox_inches='tight')

# %%
# plot results
mpl.rcParams['font.size'] = fontsize
print('trace results')
plt.figure(2)
plt.clf()
for i, (s, markersize, delta_x) in enumerate(zip(s_list, markersizes, delta_x_list)):
    for p_method in p_methods:
        methodclass, ptype = p_method.split('-')
        mean_values = []
        std_values = []
        x = 1.0 * np.arange(len(corruptions)) + delta_x
        for corruption in corruptions:
            mean_values.append(np.mean(trace_results[s]
                [corruption][p_method])
            )
            std_values.append(np.std(trace_results[s]
            [corruption][p_method])
                /np.sqrt(len(trace_results[s][corruption][p_method]))
            )
        plt.errorbar(x=x, y=mean_values, yerr=std_values, color=get_color(ptype),
                    fmt=markertypes[methodclass], 
                    markersize=markersize,
                    alpha=.5)
    plt.xticks(x, remove_underscores(corruptions), rotation=90)
    plt.ylabel(r'Trace')
    plt.yscale('log')
    plt.tight_layout()
plt.savefig(plot_file('trace'), bbox_inches='tight')

# %%
mpl.rcParams['font.size'] = fontsize
print('nll results')
plt.figure(2)
plt.clf()
for i, (s, markersize, delta_x) in enumerate(zip(s_list, markersizes, delta_x_list)):
    for p_method in p_methods:
        methodclass, ptype = p_method.split('-')
        mean_values = []
        std_values = []
        x = 1.0 * np.arange(len(corruptions)) + delta_x
        for corruption in corruptions:
            mean_values.append(np.mean(nll_results[s]
                [corruption][p_method])
            )
            std_values.append(np.std(nll_results[s]
            [corruption][p_method])
                /np.sqrt(len(nll_results[s][corruption][p_method]))
            )
        plt.errorbar(x=x, y=mean_values, yerr=std_values, color=get_color(ptype),
                    fmt=markertypes[methodclass], 
                    markersize=markersize,
                    alpha=.5)
    plt.xticks(x, remove_underscores(corruptions), rotation=90)
    plt.ylabel(r'NLL')
    plt.yscale('log')
    plt.tight_layout()
plt.savefig(plot_file('nll'), bbox_inches='tight')
