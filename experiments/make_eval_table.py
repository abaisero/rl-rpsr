#!/usr/bin/env python
import argparse

import pandas as pd


def main_table(args):
    names = 'pomdp', 'env_model', 'policy_model', 'eval_model', 'return'
    df = pd.read_csv(args.filename, sep=' ', header=None, names=names)
    df = df[df['env_model'] == 'rpsr']
    del df['env_model']

    df.rename(
        columns={
            'pomdp': 'Domain',
            'eval_model': 'Model',
            'policy_model': 'Policy',
        },
        inplace=True,
    )

    df['Domain'] = df['Domain'].str.replace('.pomdp', '')
    df['Domain'] = df['Domain'].str.replace('.POMDP', '')
    df['Domain'] = df['Domain'].str.replace('.95', '')

    df[['Policy', 'Model']] = df[['Policy', 'Model']].replace(
        {'random': 'Random', 'bsr': 'POMDP', 'psr': 'PSR', 'rpsr': 'R-PSR'},
    )

    grouped = df.groupby(['Domain', 'Model', 'Policy'])
    mean = grouped['return'].mean()
    std = grouped['return'].std()

    stats = pd.DataFrame({'mean': mean, 'std': std})
    stats = stats.agg(
        lambda data: f'{data["mean"]:.1f} \pm {data["std"]:.1f}', axis=1
    )

    stats = stats.unstack()
    stats = stats[['Random', 'POMDP', 'PSR', 'R-PSR']]
    stats.reset_index(inplace=True)

    pomdp = stats['Domain'].iloc[0]
    pomdp = pomdp.replace('.pomdp', '')
    pomdp = pomdp.replace('.POMDP', '')
    pomdp = pomdp.replace('.95', '')

    print(
        stats.to_latex(
            caption=f'Policy return estimates, \emph{{{pomdp}}}. For each policy (columns), $1000$ episodes of $100$ steps are evaluated by each model (rows).  Means and standard deviations shown as $\\mu\\pm\\sigma$.',
            label=f'tab:results:{pomdp}',
            index=False,
            header=[f'{{{col}}}' for col in stats.columns],
            escape=False,
            column_format='ll' + 4 * 'S[table-format=+3.1(1)]',
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    main_table(parser.parse_args())


if __name__ == '__main__':
    main()
