#!/usr/bin/env python3

import argparse
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 14

RCD = 'rcd'

PREFIXES = {
    'RCD': RCD,
}

DATA_CSV = 'data.csv'
TIME = 'time'
RECALL = 'recall'
PRECISION = 'precision'
CI_COUNT = 'tests'
F1_SCORE = 'f1_score'

GRAPH_LABELS = {
    TIME: 'Execution Time (sec)',
    PRECISION: 'PRECISION',
    RECALL: 'Recall',
    CI_COUNT: 'No. of CI Tests',
    F1_SCORE: 'F1 Score',
}

TITLES = {
    **GRAPH_LABELS,
    CI_COUNT: 'Average number of CI Tests executed',
    TIME: r'Execution Time for top-$k$',
    RECALL: r'top-$k$ Recall',
    RECALL: r'top-$k$ Precision'
}

COLORS = ['C0', 'C1', 'C2', 'C3']
MARKERS = itertools.cycle(['o', 's', '^', 'x', 'D', 'P'])
LINE_STYLES = itertools.cycle(['-', '--', '-.', ':', None])

# ============================= Private methods =============================

def _simple_line_plot(x, y, err=None, xlabel='', ylabel=''):
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14

    if err:
        l = ax.errorbar(x, y, yerr=err, marker=next(MARKERS), ls=next(LINE_STYLES))
        l[-1][0].set_linestyle(':')
    else:
        ax.plot(x, y, marker=next(MARKERS), ls=next(LINE_STYLES))

    ax.xaxis.label.set_fontsize(FONT_SIZE + 5)
    ax.yaxis.label.set_fontsize(FONT_SIZE + 5)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE + 2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()


def _line_plot(data, labels, err=None, xlabel='', ylabel='', title='', xc_limit=None, log_scale=False, legend_position=None):
    print(f'===================== {ylabel} =====================')
    print(f'mean = {data} | err = {err}')

    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14

    # legend_position = (0.01, 0.99)
    x = [str(x) for x in labels]

    for j, (l, v) in enumerate(data.items()):
        if err:
            l = ax.errorbar(x, v, yerr=err[l], label=l, marker=next(MARKERS), ls=next(LINE_STYLES))
            l[-1][0].set_linestyle('-.')
        else:
            ax.plot(x, v, label=l, marker=next(MARKERS), ls=next(LINE_STYLES))

        if log_scale:
            ax.set_yscale('log')

        if xc_limit is not None:
            ax.axvspan(str(xc_limit), str(np.max(labels)), alpha=0.1, color='darkorange')

    # ax.set_ylim([0, 2000])
    # ax.set_ylim([0, 30000])

    ax.xaxis.label.set_fontsize(FONT_SIZE + 5)
    ax.yaxis.label.set_fontsize(FONT_SIZE + 5)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE + 2)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

def _extract_field(df, field):
    l = dict()
    for key, value in PREFIXES.items():
        label = f'{value}_{field}'
        if label not in df.columns: continue

        l[key] = df[label].values.tolist()
    return l

def _multiple_attr(data, dir, save, attr, attr_label):
    _save_or_show = lambda name: plt.savefig(dir + name) if save else plt.show()

    temp = data.groupby(attr, as_index=False)
    mean = temp.mean()
    std = temp.std()
    labels = mean[attr].values.tolist()
    err = std / np.sqrt(len(data) / len(labels))

    for i in GRAPH_LABELS.keys():
        _line_plot(_extract_field(mean, i), labels,
                   err=_extract_field(err, i),
                   xlabel=attr_label, ylabel=GRAPH_LABELS[i],
                   log_scale=(i in []),
                   title=' '.join(i.split('_')).title())
        _save_or_show(f"{i}.pdf")
    _save_or_show(f'recall-precision.pdf')

# ============================= Public methods =============================

def multiple_nodes(data, dir, **kwargs):
    save = kwargs['save']
    _multiple_attr(data, dir, save, 'nodes', 'Nodes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates plots from experiment data')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the experiment data')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save to path, otherwise just show the plots')

    args = parser.parse_args()
    path = args.path
    save = args.save

    dir = path + '/'
    data = pd.read_csv(dir + DATA_CSV)
    multiple_nodes(data, dir, save=save)
