import os
import glob

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt

from color_map import get_color


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    # optional arguments
    file_prefix = getattr(cfg, 'file_prefix', 'Metrics')
    metric = getattr(cfg, 'metric', 'trace')
    psi_ref = getattr(cfg, 'psi_ref', 'ggnit')

    # mandatory arguments
    data_name = cfg.data.name
    pred_model_name = cfg.pred_model.name
    metric_collection = {}
    seed_folders = glob.glob(os.path.join(
        'results',
        data_name,
        pred_model_name,
        'seed*'
    ))
    for seed_folder in seed_folders:
        # loop through methods
        file_list = glob.glob(os.path.join(
            seed_folder,
            f'{file_prefix}_*_Psi{psi_ref}.pt'
        ))
        for file in file_list:
            with open(file, 'rb') as f:
                loaded_file = torch.load(f)
                values = np.array(loaded_file[metric])
            file_name = os.path.basename(file)
            method = file_name.split('_')[1]
            try:
                metric_collection[method]['values'].append(values)
            except KeyError:
                metric_collection[method] = {'values': [values,]}
            try:
                assert metric_collection[method]['s_list'] \
                    == loaded_file['s_list']
            except KeyError:
                metric_collection[method]['s_list'] = loaded_file['s_list']
            baseline_file = os.path.join(
                seed_folder,
                f'{file_prefix}_None_Psi{method}.pt',
            )
            try:
                with open(baseline_file, 'rb') as f:
                    baseline_value = np.array(torch.load(f)[metric])
                    try:
                        metric_collection[method]['baseline']['values']\
                            .append(baseline_value)
                    except KeyError:
                        metric_collection[method]['baseline']\
                            = {'values': [baseline_value,]}
            except FileNotFoundError:
                metric_collection[method]['baseline'] = None

            # try to load baseline

    for method in metric_collection.keys():
        values = np.stack(metric_collection[method]['values'], axis=0)
        metric_collection[method].update({
            'mean': np.mean(values, axis=0),
            'std': np.std(values, axis=0)/np.sqrt(len(values)),
        })
        if metric_collection[method]['baseline']:
            baseline_values = metric_collection[method]['baseline']['values']
            metric_collection[method]['baseline'].update({
                'mean': np.mean(baseline_values, axis=0).item(),
                'std': np.std(baseline_values, axis=0).item()/np.sqrt(len(values)),
            })

    # plot results
    for method in metric_collection.keys():
        color = get_color(method)
        s_list = metric_collection[method]['s_list']
        if len(s_list) > 1:
            plt.plot(
                s_list,
                metric_collection[method]['mean'],
                label=method,
                color=color,
                alpha=0.7,
            )
            plt.fill_between(
                s_list,
                metric_collection[method]['mean']
                - metric_collection[method]['std'],
                metric_collection[method]['mean']
                + metric_collection[method]['std'],
                alpha=.5,
                color=color,
            )
            if metric_collection[method]['baseline']:
                plt.axhline(
                    y=metric_collection[method]['baseline']['mean'],
                    linestyle='dashed',
                    color=color,
                    alpha=0.7,
                )
        else:
            plt.axhline(
                y=metric_collection[method]['mean'][0],
                linestyle='dotted',
                color=get_color(method),
                label=method,
                alpha=0.7,
            )
    plt.xlabel('s')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(os.path.join(
        'results',
        data_name,
        pred_model_name,
        f'plot_{metric}_{data_name}_{pred_model_name}.pdf'
        ))


if __name__ == '__main__':
    run_main()
