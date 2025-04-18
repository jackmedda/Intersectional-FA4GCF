import os
import pdb
import re
import math
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mpl_lines
import matplotlib.colors as mpl_colors
import matplotlib.ticker as mpl_tickers
import matplotlib.patches as mpl_patches
import matplotlib.transforms as mpl_trans
import matplotlib.legend_handler as mpl_legend_handlers


def update_plt_rc():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 30

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)


class HandlerEllipse(mpl_legend_handlers.HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = orig_handle.get_center()
        radius = orig_handle.get_radius()
        p = mpl_patches.Ellipse(
            xy=center, width=radius, height=radius
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


if __name__ == "__main__":
    current_file = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--plots_path_to_merge', '-pptm', default=os.path.join(current_file, 'plots'))
    parser.add_argument('--base_plots_path', '-bpp', default=os.path.join(current_file, 'merged_plots'))
    parser.add_argument('--exclude', '-ex', nargs='+', help='Exclude certain config ids', default=None)
    parser.add_argument('--psi_impact', action="store_true")

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    args.exclude = args.exclude or []
    print(args)

    sns.set_context("paper")
    update_plt_rc()
    out_path = os.path.join(args.base_plots_path, args.dataset)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    PVAL_001 = '^'
    PVAL_005 = '*'


    def pval_symbol(pval, n_groups=2):  
        if pval < (0.05 / (n_groups // 2)):
            return PVAL_005

        return ''

    def remove_str_pval(str_pval):
        return str_pval.replace('^', '').replace('*', '')

    def hl(val):
        return "\hl{" + val + "}"

    user_policies = ['ZN', 'FR', 'IR']
    item_policies = ['IP', 'IT', 'PR']
    user_item_policies = [f"{up}+{ip}" for up in user_policies for ip in item_policies]

    dataset_order = ['LFM1M', 'ML1M', 'ML1MD', 'KRECB', 'KRECS']
    models_order = ['HMLET', 'LightGCN', 'NGCF', 'SGL', 'XSimGCL']
    policies_order = ['Orig'] + user_policies + item_policies + user_item_policies
    group_attr_order = ['Gender | Age', 'Feat0 | Feat13']
    pert_type_order = ['Orig', 'Pert']

    dataset_map = {
        'lastfm-1m': 'LFM1M',
        'ml-1m': 'ML1M',
        'ml-1m_dense': 'ML1MD',
        'kuairec_big': 'KRECB',
        'kuairec_small_watch_ratio_1_inf': 'KRECS',
    }

    group_attr_map = {
        'gender|age': 'Gender | Age',
        'onehot_feat0|onehot_feat13': 'Feat0 | Feat13'
    }

    short_group_attr_map = {
        'Gender | Age': 'G | A',
        'Feat0 | Feat13': 'F0 | F13'
    }

    loaded_dfs = []
    group_dfs = []
    dfs_across_exps = {'consumer': [], 'provider': []}
    edge_perturbation_impact = {}
    plots_path = os.path.join(args.plots_path_to_merge, args.dataset)
    del_dist_giant_df_list = []
    for dirpath, dirnames, filenames in os.walk(plots_path):
        if filenames:
            for x in filenames:
                if x == 'DP_barplot.csv':
                    metadata = dirpath.split(args.dataset + os.sep)[1]
                    mod, s_attr, conf_pol = metadata.split(os.sep)
                    conf_id, policy = conf_pol.split('_')

                    metadata_map = {
                        'Dataset': args.dataset,
                        'Model': mod,
                        'GroupAttribute': s_attr
                    }
                    df = pd.read_csv(os.path.join(dirpath, x))

                    delta_col = df.columns[df.columns.str.contains('Delta')][0]
                    metric = delta_col.replace('$\Delta$', '')

                    df.rename(columns={delta_col: 'DP'}, inplace=True)
                    for key, val in metadata_map.items():
                        df[key] = val

                    rel_cols = ['Policy'] + list(metadata_map.keys())
                    loaded_dfs.append(df[rel_cols + ['Split', metric.upper(), 'DP', 'pvalue']])

                    g_df = df[df.columns[~df.columns.isin([delta_col, 'Split', 'DP', 'pvalue'])]].melt(rel_cols).rename(columns={
                        'variable': 'Metric', 'value': 'Value'
                    })
                    group_dfs.append(g_df)
                elif x == 'del_dist_giant.csv':
                    del_dist_giant_df_list.append(pd.read_csv(os.path.join(dirpath, x)))

    orig_pert_pval_data = []
    orig_pert_pval_cols = ['Dataset', 'Model', 'GroupAttribute', 'Policy', 'Split', 'Metric', 'P_value']
    all_dsets_path = os.path.dirname(plots_path)
    for dirpath, dirnames, filenames in os.walk(all_dsets_path):
        if filenames:
            for x in filenames:
                if x == 'orig_pert_pval_dict.pkl' and any(x in dirpath for x in dataset_map.keys()):
                    metadata = dirpath.split(all_dsets_path + os.sep)[1]
                    dset, mod, s_attr, conf_pol = metadata.split(os.sep)
                    conf_id, policy = conf_pol.split('_')

                    with open(os.path.join(dirpath, 'orig_pert_pval_dict.pkl'), 'rb') as f:
                        pval = pickle.load(f)
                        for spl, spl_pval in pval.items():
                            for metr, pval_value in spl_pval.items():
                                orig_pert_pval_data.append(
                                    [dset, mod, s_attr, policy, spl, metr, pval_value]
                                )

    orig_pert_pval_df = pd.DataFrame(orig_pert_pval_data, columns=orig_pert_pval_cols)

    cols_order = ["Dataset", "Model", "GroupAttribute", "Policy", "Split", "Metric", "Value", "pvalue"]
    first_df = pd.concat(loaded_dfs, ignore_index=True)
    first_df = first_df.melt(first_df.columns[~first_df.columns.isin([metric, 'DP'])], var_name='Metric', value_name='Value')
    first_df = first_df.drop_duplicates(subset=cols_order[:6])
    first_df = first_df[cols_order]

    first_df['Dataset'] = first_df['Dataset'].map(dataset_map)
    first_df['GroupAttribute'] = first_df['GroupAttribute'].map(group_attr_map)
    first_df.sort_values(cols_order[:5])

    orig_pert_pval_df['Dataset'] = orig_pert_pval_df['Dataset'].map(dataset_map)
    orig_pert_pval_df['GroupAttribute'] = orig_pert_pval_df['GroupAttribute'].map(group_attr_map)

    first_df['Value'] *= 100  # transform it to percentage
    first_df.to_csv(os.path.join(out_path, 'best_exp_raw_perc_values_table.csv'), index=False)

    first_merged_dfs_to_merge = []
    for dirpath, _, merged_csvs in os.walk(os.path.dirname(out_path)):
        if merged_csvs and '.ipynb_checkpoints' not in dirpath:
            for mc in merged_csvs:
                if mc == 'best_exp_raw_perc_values_table.csv' and any(x in dirpath for x in dataset_map.keys()):
                    first_merged_dfs_to_merge.append(pd.read_csv(os.path.join(dirpath, mc)))

    first_total_df = pd.concat(first_merged_dfs_to_merge, axis=0, ignore_index=True)
    first_total_df["Model"] = first_total_df["Model"].replace('SVD_GCN', 'SVD-GCN')
    first_total_df.to_csv(os.path.join(os.path.dirname(out_path), 'total_raw_perc_table.csv'), index=False)

    if args.psi_impact:
        if first_total_df.Policy.str.contains('IP+IR', regex=False).any():
            def remap_ipir_psi_values(val):
                psi_values = re.search('(?<=\().*(?=\))', val)[0]
                return f"IR+IP ({'+'.join(psi_values.split('+')[::-1])})"
            first_total_df['Policy'] = first_total_df['Policy'].map(
                lambda x: remap_ipir_psi_values(x) if 'IP+IR' in x else x
            )
        base_exp_ratios = '(0.35+0.2)'
        # base_exp_mask = first_df.Policy.str.contains('(0.35+0.2)', regex=False)
        # base_exp_df = first_df[base_exp_mask].reset_index(drop=True)
        # psi_impact_df = first_df[~base_exp_mask].reset_index(drop=True)

        first_total_df = first_total_df[first_total_df['Split'] == 'Test'].reset_index(drop=True)
        dp_key = '$\Delta$'
        first_total_df = first_total_df.replace('DP', dp_key)
        fixed_user_psi = first_total_df[first_total_df['Policy'].str.contains('(0.35+', regex=False)]
        fixed_item_psi = first_total_df[first_total_df['Policy'].str.contains('+0.2)', regex=False)]

        for i, (fixed_psi_df, varying_psi_type) in enumerate(
                zip([fixed_item_psi, fixed_user_psi], ['$\Psi_{\mathcal{U}}$', '$\Psi_{\mathcal{I}}$'])
        ):
            fixed_psi_df[varying_psi_type] = fixed_psi_df['Policy'].map(
                lambda p: p.split('(')[1].replace(')', '').split('+')[i]
            ).astype(float)
            fixed_psi_df['Policy'] = fixed_psi_df['Policy'].map(lambda p: p.split()[0])
            fixed_psi_df['Setting'] = fixed_psi_df[['Policy', 'Dataset', 'Model', 'GroupAttribute']].apply(
                lambda x: f'({",".join(x)})', axis=1
            )

            style_kws = dict(
                style='Metric',
                markers={dp_key: 'X', 'NDCG': 'P'},
                dashes={dp_key: (), 'NDCG': (2, 1)},
                errorbar=None,
                lw=5,
                markersize=30
            )

            # psi_dsets_order = ['LF1K', 'ML1M']
            psi_dsets_order = ['LF1M', 'ML1M']

            fixed_psi_df_gby = fixed_psi_df.groupby(['Policy', 'Dataset', 'Model', 'GroupAttribute'])
            for dset_i, ((psi_pol, psi_dset, psi_mod, psi_grattr), setting_psi_df) in enumerate(fixed_psi_df_gby):
                psi_out_path = os.path.join(
                    os.path.dirname(out_path), 'varying_psi', psi_dset, psi_mod, psi_grattr, psi_pol
                )
                os.makedirs(psi_out_path, exist_ok=True)

                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                colors = sns.color_palette('cividis', n_colors=4)
                ax_color, axx_color = colors[0], colors[-2]
                ax.margins(y=0.1)
                sns.lineplot(
                    x=varying_psi_type, y='Value', data=setting_psi_df[setting_psi_df['Metric'] == dp_key],
                    color=ax_color, ax=ax, **style_kws
                )
                ax.set_xlabel('')
                # ax.set_ylabel(f'{dset_psi_df["Setting"].iloc[0]}\n{dp_key}', color=ax_color)
                ax.set_ylabel(dp_key, color=ax_color)
                ax.tick_params(axis='y', labelcolor=ax_color)

                axx = ax.twinx()
                axx.margins(y=0.1)
                sns.lineplot(
                    x=varying_psi_type, y='Value', data=setting_psi_df[setting_psi_df['Metric'] == 'NDCG'],
                    color=axx_color, ax=axx, **style_kws
                )
                axx.set_xlabel('')
                axx.set_ylabel('NDCG', color=axx_color)
                axx.tick_params(axis='y', labelcolor=axx_color)

                ax.grid(axis='both', which='major', ls=':', color='k')
                ax.set_xticks(fixed_psi_df[varying_psi_type].unique())
                ax.xaxis.set_major_formatter(mpl_tickers.FuncFormatter(lambda x, pos: f"{int(x * 100)}%"))
                ax.yaxis.set_major_formatter(mpl_tickers.StrMethodFormatter("{x:.2f}"))
                axx.yaxis.set_major_formatter(mpl_tickers.StrMethodFormatter("{x:.2f}"))
                ax.yaxis.set_major_locator(mpl_tickers.LinearLocator(6))
                axx.yaxis.set_major_locator(mpl_tickers.LinearLocator(6))

                ax_handles, ax_labels = ax.get_legend_handles_labels()
                axx_handles, axx_labels = axx.get_legend_handles_labels()
                handles, labels = ax_handles + axx_handles, ax_labels + axx_labels
                ax.get_legend().remove()
                axx.get_legend().remove()

                fig.savefig(
                    os.path.join(psi_out_path, ('user' if i == 0 else 'item') + '_varying_psi_lineplot.png'),
                    bbox_inches='tight', pad_inches=0, dpi=300
                )
                plt.close(fig)

        figlegend = plt.figure(figsize=(len(labels),  1))
        figlegend.legend(
            handles, labels, loc='center', frameon=False, fontsize=12, ncol=len(labels),
            markerscale=0.7, handlelength=5, # handletextpad=4, columnspacing=2, borderpad=0.1
        )
        figlegend.savefig(
            os.path.join(os.path.dirname(out_path), 'legend_psi_impact.png'),
            dpi=300, bbox_inches="tight", pad_inches=0
        )

        exit()
    else:
        first_total_df['Policy'] = first_total_df['Policy'].str.replace('IP+IR', 'IR+IP')
        orig_pert_pval_df['Policy'] = orig_pert_pval_df['Policy'].str.replace('IP+IR', 'IR+IP')

    # Raw best settings utility and fairness
    first_best_settings = []
    first_best_settings_cols = ["Dataset", "Model", "GroupAttribute", "Policy"]
    first_total_gby = first_total_df.groupby(["Dataset", "Model", "Split", "Metric"])
    for sa in group_attr_order:
        for (dset, mod, spl, metr), first_total_group_df in first_total_gby:
            if spl == "Valid" and metr == "DP" and first_total_group_df["GroupAttribute"].str.contains(sa).any():
                first_total_sa_df = first_total_group_df[first_total_group_df["GroupAttribute"] == sa]
                first_total_sa_df = first_total_sa_df[first_total_sa_df["Policy"] != "Orig"]
                curr_best_setting = first_total_sa_df.iloc[first_total_sa_df.Value.argmin()]
                first_best_settings.append(curr_best_setting)

    first_best_settings_df = pd.concat(first_best_settings, axis=1).T
    first_bs_idx = first_best_settings_df.set_index(first_best_settings_cols)
    best_first_total_df = first_total_df.set_index(first_best_settings_cols).loc[first_bs_idx.index].reset_index()
    test_best_first_total_df = best_first_total_df[best_first_total_df["Split"] == "Test"]
    # test_best_pol_first_total_df = best_first_total_df.groupby(["Dataset", "Model", "GroupAttribute", "Metric"]).apply(
    #     lambda x: x.sort_values("Value", ascending=False).iloc[0]
    # ).reset_index(drop=True)

    test_bp_idx = test_best_first_total_df.set_index(first_best_settings_cols[:-1] + ["Split", "Metric"]).index
    test_orig_first_total_df = first_total_df[first_total_df["Policy"] == "Orig"].set_index(
        first_best_settings_cols[:-1] + ["Split", "Metric"]
    ).loc[test_bp_idx].reset_index()

    before_key, after_key = "Base", "Aug"
    first_best_pol_orig_df = pd.concat([test_best_first_total_df, test_orig_first_total_df], axis=0, ignore_index=True)
    first_best_pol_orig_df["Metric"] = first_best_pol_orig_df["Metric"].replace('DP', '$\Delta$')
    first_best_pol_orig_df["Status"] = first_best_pol_orig_df["Policy"].map(lambda p: before_key if p == "Orig" else after_key)
    first_best_pol_orig_df["GroupAttribute"] = first_best_pol_orig_df["GroupAttribute"].map(short_group_attr_map.__getitem__)

    test_orig_pert_pval_df = orig_pert_pval_df[orig_pert_pval_df["Split"] == "Test"]
    test_orig_pert_pval_df["Metric"] = test_orig_pert_pval_df["Metric"].replace('DP', '$\Delta$')
    test_orig_pert_pval_df["GroupAttribute"] = test_orig_pert_pval_df["GroupAttribute"].map(short_group_attr_map.__getitem__)

    pol_as_a_metric_df = first_best_pol_orig_df[(first_best_pol_orig_df["Status"] == "Aug") & (first_best_pol_orig_df["Metric"] == "NDCG")]
    pol_as_a_metric_df["Value"] = pol_as_a_metric_df["Policy"].map(lambda p: "{\\small \\emph{" + p + "}}")
    pol_as_a_metric_df["Metric"] = "PolAsMetric"

    # first_best_pol_orig_df = pd.concat([first_best_pol_orig_df, pol_as_a_metric_df], axis=0, ignore_index=True)
    pol_as_a_metric_df_pivot = pol_as_a_metric_df.pivot(
        index=["Dataset", "GroupAttribute", "Status"],
        columns=["Model", "Metric"],
        values="Value"
    )

    first_best_pol_orig_df_pivot = first_best_pol_orig_df.pivot(
        index=["Dataset", "GroupAttribute", "Status"],
        columns=["Model", "Metric"],
        values="Value"
    ).reindex(
        [before_key, after_key], axis=0, level=2
    ).reindex(
        ["NDCG", "$\Delta$"], axis=1, level=1
    )
    first_best_pol_orig_df_pivot.to_csv(os.path.join(os.path.dirname(out_path), "best_policy_compare_orig_dp_utility.csv"), sep='\t')


    def bold_row(row):
        metr_index = row.index.get_level_values(level=1)
        best_ndcg_idx = row[metr_index == "NDCG"].argmax() * 2
        best_dp_idx = row[metr_index == r"$\Delta$"].argmin() * 2 + 1
        row = row.apply(lambda x: f"{x:.2f}")
        row[best_ndcg_idx] = r"\textbf{" + row[best_ndcg_idx] + "}"
        row[best_dp_idx] = r"\textbf{" + row[best_dp_idx] + "}"
        return row


    def underline_improvements(col):
        float_col = col.str.replace(r"\textbf{", "").str.replace("}", "").astype(float)
        status_index = float_col.index.get_level_values(level=2)
        after_idx = float_col[status_index == after_key].values
        before_idx = float_col[status_index == before_key].values
        fn = "__gt__" if col.name[1] == "NDCG" else "__lt__"
        hl_mask = getattr(after_idx - before_idx, fn)(0)
        hl_idx = hl_mask.nonzero()[0] * 2 + 1
        col[hl_idx] = r"\underline{" + col[hl_idx] + "}"
        return col

    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.apply(bold_row, axis=1)
    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.apply(underline_improvements, axis=0)
    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.join(pol_as_a_metric_df_pivot).reindex(
        models_order, axis=1, level=0
    ).reindex(
        ["PolAsMetric", "NDCG", "$\Delta$"], axis=1, level=1
    )
    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.fillna('').rename(columns={'PolAsMetric': ''})

    first_best_pol_orig_df_pivot.index.names = [""] * len(first_best_pol_orig_df_pivot.index.names)
    first_best_pol_orig_df_pivot.columns.names = [""] * len(first_best_pol_orig_df_pivot.columns.names)

    pval_idx_columns = ["Dataset", "GroupAttribute", "Model", "Metric", "Policy"]
    test_orig_pert_pval_df["Model"] = test_orig_pert_pval_df["Model"].replace('SVD_GCN', 'SVD-GCN')
    test_orig_pert_pval_df_idx = test_orig_pert_pval_df.set_index(pval_idx_columns)
    for _, row in first_best_pol_orig_df_pivot.iterrows():
        row_idx = row.name
        for col_idx, _ in row.items():
            row_col_policy = first_best_pol_orig_df_pivot.loc[row_idx, (*col_idx[:1], '')].split('{')[-1].split('}')[0]
            if "Aug" in row_idx and '' not in col_idx:
                pval_str = pval_symbol(test_orig_pert_pval_df_idx.loc[(*row_idx[:2], *col_idx, row_col_policy), 'P_value'])
                if pval_str:
                    pval_str = f"$^{PVAL_005}$"
                row_col_val = first_best_pol_orig_df_pivot.loc[row_idx, col_idx]
                first_best_pol_orig_df_pivot.loc[row_idx, col_idx] = pval_str + row_col_val

    with open(os.path.join(os.path.dirname(out_path), "best_policy_compare_orig_dp_utility.tex"), "w") as tex_file:
        best_policy_tex_text = first_best_pol_orig_df_pivot.to_latex(
            column_format=">{\\raggedright}p{7.5mm}>{\\raggedright}p{4mm}l*{9}{|>{\\raggedright}p{2.5mm}rr}",
            multicolumn=True,
            multicolumn_format="c|",
            multirow=True,
            escape=False
        ).replace(
            "NDCG", r"NDCG $\uparrow$"
        ).replace(
            "$\Delta$", "$\Delta$ $\downarrow_0$"
        ).replace(
            "\multirow[t]", "\multirow[c]"
        )
        for dset in dataset_map.values():
            best_policy_tex_text = best_policy_tex_text.replace('{' + dset + '}', '{\\rotatebox[origin=c]{90}{' + dset + '}}')

        best_policy_tex_text = re.sub(r'(\s*&)+\s*\\\\\n', '', best_policy_tex_text)

        tex_file.write(best_policy_tex_text)

    # Mini Heatmaps
    heatmaps_path = os.path.join(os.path.dirname(out_path), 'mini_heatmaps')
    os.makedirs(heatmaps_path, exist_ok=True)

    MAX_VMAX = 7
    for spl in ["Valid", "Test"]:
        cmap = sns.color_palette("cividis", as_cmap=True)
        first_total_heat_df = first_total_df[(first_total_df["Split"] == spl) & (first_total_df["Metric"] == "DP")]
        first_t_hm_gby = first_total_heat_df.groupby(["Model", "Dataset", "GroupAttribute"])
        vmin, vmax = 0, min(MAX_VMAX, first_total_heat_df["Value"].max())
        norm = mpl_colors.Normalize(vmin, vmax)
        for mod in models_order:
            mod_hm_df = first_total_heat_df[first_total_heat_df["Model"] == mod]
            for dset in dataset_order:
                for s_attr in group_attr_order:
                    if (mod, dset, s_attr) not in first_t_hm_gby.groups:
                        continue

                    mini_heat_df = first_t_hm_gby.get_group((mod, dset, s_attr))

                    mini_heatmap_path = os.path.join(heatmaps_path, spl, mod)
                    os.makedirs(mini_heatmap_path, exist_ok=True)

                    orig_policy_dp = mini_heat_df.loc[mini_heat_df["Policy"] == "Orig", "Value"].item()
                    user_mh_df = mini_heat_df[mini_heat_df["Policy"].isin(user_policies)].set_index("Policy").reindex(
                        user_policies
                    )
                    item_mh_df = mini_heat_df[mini_heat_df["Policy"].isin(item_policies)].set_index("Policy").reindex(
                        item_policies
                    )
                    user_item_mh_df = mini_heat_df[mini_heat_df["Policy"].isin(user_item_policies)]

                    user_item_mh_df[['U', 'I']] = user_item_mh_df['Policy'].str.split('+', expand=True).values
                    user_item_hmap_data = user_item_mh_df.pivot(columns='I', index='U', values='Value').reindex(
                        user_policies, axis=0
                    ).reindex(
                        item_policies, axis=1
                    ).values
                    user_hmap_data = user_mh_df.loc[:, "Value"].values
                    item_hmap_data = item_mh_df.loc[:, "Value"].values

                    data = np.hstack([user_hmap_data.reshape([-1, 1]), user_item_hmap_data])
                    data = np.vstack([np.concatenate([[orig_policy_dp], item_hmap_data]).reshape([1, -1]), data])
                    data = pd.DataFrame(data, index=[''] + user_policies, columns=[''] + item_policies)

                    # annot = np.full_like(data, fill_value='', dtype='<U4')
                    greater_than_vmax_mask = data > MAX_VMAX
                    annot = data.applymap("{:.2f}".format).values
                    annot[0, 0] = "Base\n" + annot[0, 0]
                    annot[greater_than_vmax_mask] = f"> {MAX_VMAX}"

                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    sns.heatmap(
                        data, cmap=cmap, norm=norm, annot=annot, fmt="", square=True,
                        cbar=False, linewidths=.5, linecolor='white', ax=ax
                    )
                    ax.tick_params(length=0)
                    ax.xaxis.tick_top()
                    fig.savefig(
                        os.path.join(mini_heatmap_path, f"{dset}_{s_attr}_mini_heatmap.png"),
                        dpi=300, bbox_inches="tight", pad_inches=0
                    )
                    plt.close(fig)

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_fig, cbar_ax = plt.subplots(1, 1, figsize=(0.5, 10.8))
        colorbar = cbar_fig.colorbar(mappable, cax=cbar_ax, orientation="vertical")
        colorbar.ax.set_yticklabels([float(t.get_text()) for t in colorbar.ax.get_yticklabels()])
        cbar_fig.savefig(
            os.path.join(heatmaps_path, spl, "heatmaps_colorbar.png"), dpi=300, bbox_inches="tight", pad_inches=0
        )
        plt.close(cbar_fig)


    # Graph metrics merge with deleted edges distribution plot
    del_dist_giant_df = pd.concat(del_dist_giant_df_list, axis=0, ignore_index=True)
    del_dist_giant_df.to_csv(os.path.join(out_path, 'merge_del_dist_giant_df.csv'), index=False)

    del_dist_giant_dfs_to_merge = []
    for dirpath, _, merged_csvs in os.walk(os.path.dirname(out_path)):
        if merged_csvs and '.ipynb_checkpoints' not in dirpath:
            for mc in merged_csvs:
                if mc == 'merge_del_dist_giant_df.csv' and any(x in dirpath for x in dataset_map.keys()):
                    del_dist_giant_dfs_to_merge.append(pd.read_csv(os.path.join(dirpath, mc)))

    merged_del_dist_giant_df = pd.concat(del_dist_giant_dfs_to_merge, axis=0, ignore_index=True)

    rq3_confs = {
        "gender|age": ["ml-1m", "lastfm-1m", "ml-1m_dense"],
        "onehot_feat0|onehot_feat13": ["kuairec_big", "kuairec_small_watch_ratio_1_inf"],
        "Item": ["ml-1m", "lastfm-1m", "ml-1m_dense", "kuairec_big", "kuairec_small_watch_ratio_1_inf"],
    }
    short_gm = {"Degree": "DEG", "Density": "DTY", "Sparsity": "SP", "Reachability": "SP"}
    rq3_dsets = list(np.unique([d for conf in rq3_confs.values() for d in conf]))
    if (set(merged_del_dist_giant_df["Dataset"].unique()) & set(rq3_dsets)) == set(rq3_dsets):
        # spine_color = "red"

        merged_del_dist_giant_df["Dataset"] = merged_del_dist_giant_df["Dataset"].map(dataset_map)
        unique_quantiles = merged_del_dist_giant_df["Quartile"].unique()
        parts = len(unique_quantiles) * 2
        merged_del_dist_giant_df["Quartile"] = merged_del_dist_giant_df["Quartile"].map(
            lambda x: [f"Q{i + 1}" for i in range(parts)][int((x * parts - 1) // 2)]
        )
        giant_hue_col = "Demo Group"
        giant_y_col = "Del Edges Distribution"
        giant_x_col = "Quartile"
        giant_hue_col_order = {
            'gender|age': ['Older Males', 'Younger Males', 'Older Females', 'Younger Females'],
            'onehot_feat0|onehot_feat13': ['0|0', '0|1', '1|0', '1|1'],
            'Item': ['Item'],
        }

        fs_titles_labels = 26
        fs_ticks = 22

        for (giant_sa, giant_mod), giant_samod_df in merged_del_dist_giant_df.groupby(["Sens Attr", "Model"]):
            if giant_sa in rq3_confs:
                giant_dsets = rq3_confs[giant_sa]
                mapped_giant_dsets = [dataset_map[d] for d in giant_dsets]
                giant_samod_df = giant_samod_df[giant_samod_df["Dataset"].isin(mapped_giant_dsets)]
                giant_gms = giant_samod_df["Graph Metric"].unique()

                markers_map = dict(zip(giant_hue_col_order[giant_sa], ["d", "X", "*", "o"]))

                # adv_dsets_color_map = {
                #     "ML-1M": {"Males": spine_color, "Younger": spine_color},
                #     "FENG": {"Older": spine_color},
                #     "LFM-1K": {"Females": spine_color, "Older": spine_color},
                #     "INS": {"Males": spine_color, "Younger": spine_color},
                # }
                # if giant_mod == "NGCF":
                #     del adv_dsets_color_map["INS"]["Younger"]
                #     adv_dsets_color_map["INS"]["Older"] = spine_color

                giant_fig, giant_axs = plt.subplots(
                    len(giant_dsets), len(giant_gms), sharey='row', sharex=True,
                    figsize=(30, 3 * len(giant_dsets)), layout="constrained"
                )
                giant_fig.supylabel('Added Edges Distribution', fontsize=fs_titles_labels)
                giant_fig.supxlabel('Quartiles', ha='left', fontsize=fs_titles_labels)

                giant_df_gby = giant_samod_df.groupby(["Dataset", "Graph Metric"])
                for d_i, giant_dset in enumerate(mapped_giant_dsets):
                    for gm_i, giant_gm in enumerate(giant_gms):
                        giant_ax = giant_axs[d_i, gm_i]

                        if (giant_dset, giant_gm) in giant_df_gby.groups:
                            dset_g_plot_df = giant_df_gby.get_group((giant_dset, giant_gm))
                            dset_g_plot_df = dset_g_plot_df[[giant_x_col, giant_y_col, giant_hue_col]]
                            sns.lineplot(
                                x=giant_x_col,
                                y=giant_y_col,
                                data=dset_g_plot_df,
                                hue=giant_hue_col,
                                hue_order=giant_hue_col_order[giant_sa],
                                style=giant_hue_col,
                                style_order=giant_hue_col_order[giant_sa],
                                markers=markers_map,
                                markersize=25,
                                lw=2,
                                dashes=False,
                                legend="full",
                                alpha=0.8,
                                ax=giant_ax
                            )

                            if d_i == 0 and gm_i == 0:
                                _handles, _labels = giant_ax.get_legend_handles_labels()
                                giant_ax.get_legend().remove()
                                giant_legend = giant_fig.legend(
                                    _handles, _labels, loc="lower center", ncol=4,
                                    bbox_to_anchor=(0.5, 1.01, 0.05, 0.05),
                                    bbox_transform=giant_fig.transFigure,
                                    markerscale=1.3, prop={'size': fs_titles_labels},
                                    frameon=False
                                )
                                giant_legend.set_zorder(10)
                            else:
                                giant_ax.get_legend().remove()

                            if gm_i == 0:
                                giant_ax.yaxis.set_major_formatter(mpl_tickers.StrMethodFormatter("{x:.2f}"))
                                giant_ax.set_ylabel(giant_dset, fontsize=fs_titles_labels)
                            else:
                                giant_ax.set_ylabel('')

                            if d_i == len(giant_dsets) - 1:
                                giant_ax.set_xlabel(giant_ax.get_xlabel(), fontsize=fs_titles_labels)

                            giant_ax.grid(True, axis='both', ls=':')
                            giant_ax.set_xlabel('')
                        else:
                            giant_ax.set_axis_off()
                            giant_ax.text(
                                0.5, 0.5, 'Node Group NA', ha='center', va='center',
                                fontdict=dict(fontsize=20), transform=giant_ax.transAxes
                            )

                        if d_i == 0:
                            giant_ax.set_title(
                                f"{giant_gm} ({short_gm[giant_gm]})" if giant_gm != 'Reachability' \
                                                                     else 'Intra-Group Distance (IGD)',
                                fontsize=fs_titles_labels
                            )

                        giant_ax.tick_params(which='major', labelsize=fs_ticks)
                        if gm_i > 0:
                            giant_ax.tick_params(length=0, labelleft=False)
                        if d_i < len(giant_dsets) - 1:
                            giant_ax.tick_params(length=0, labelbottom=False)

                        # for spine in giant_ax.spines.values():
                        #     spine.set_edgecolor(adv_dsets_color_map[giant_dset].get(giant_g, 'black'))

                # giant_fig.subfigs[0].subplots_adjust(
                #     left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.1, hspace=0.08
                # )
                # giant_fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
                giant_fig.savefig(
                    os.path.join(os.path.dirname(out_path), f"{giant_sa}_{giant_mod}_{'_'.join(rq3_dsets)}_del_dist_plot_per_gm.png"),
                    bbox_inches="tight", pad_inches=0, dpi=250
                )
