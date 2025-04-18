import os
import re
import sys
import pickle
import argparse

import scipy
import sklearn
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.feature_selection as sk_feats

current_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file, os.pardir))

import fa4gcf.utils as utils
import fa4gcf.evaluation as evaluation
from fa4gcf.config import Config
from fa4gcf.utils.case_study import (
    pref_data_from_checkpoint,
    extract_metrics_from_perturbed_edges
)


def compute_graph_metrics_analysis_tables_data(_sens_gmdf,
                                               _gm_analysis_tables_data: dict,
                                               gm_rel_info,
                                               graph_metrics,
                                               _iterations=100,
                                               kl_eps=1e-8):  # avoids NaN
    # https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable
    # Paper that states how select number of neighbors, repetitions and usage of median
    # we regroup again by sensitive attribute because `Item` was added as label
    for gm_sens_attr, gm_sattr_df in _sens_gmdf.groupby("Sens Attr"):
        gm_sattr_df_norm = gm_sattr_df.copy(deep=True)
        gm_sattr_df_norm['Degree'] = sklearn.preprocessing.MinMaxScaler().fit_transform(
            gm_sattr_df_norm.loc[:, ['Degree']].to_numpy()
        ).squeeze()
        gm_sattr_df_norm['Del Edges Count Scaled'] = sklearn.preprocessing.MinMaxScaler().fit_transform(
            gm_sattr_df_norm.loc[:, ['Del Edges Count']].to_numpy()
        ).squeeze()
        for gm_dg, gm_dgdf in gm_sattr_df_norm.groupby("Demo Group"):
            mi_res = np.zeros((_iterations, len(graph_metrics)), dtype=float)
            wd_res = [np.inf] * len(graph_metrics)
            kl_res = [0] * len(graph_metrics)
            dcor_res = [None] * len(graph_metrics)
            dcor_pval = [None] * len(graph_metrics)
            gm_del_dist = [None] * len(graph_metrics)

            n_del_edges_scaled = gm_dgdf.loc[:, 'Del Edges Count Scaled'].to_numpy()
            for gm_i, gm in enumerate(graph_metrics):
                gm_dg_data = gm_dgdf.loc[:, gm].to_numpy()
                gm_dg_data = gm_dg_data.astype(n_del_edges_scaled.dtype)

                if "wd" in _gm_analysis_tables_data:
                    wd_res[gm_i] = scipy.stats.wasserstein_distance(gm_dg_data, n_del_edges_scaled)
                if "kl" in _gm_analysis_tables_data:
                    kl_res[gm_i] = scipy.stats.entropy(
                        n_del_edges_scaled + kl_eps, gm_dg_data + kl_eps, base=2
                    )
                # if "dcor" in _gm_analysis_tables_data:
                #     dcor_res[gm_i] = dcor.distance_correlation(gm_dg_data, n_del_edges_scaled)
                #     dcor_pval[gm_i] = dcor.independence.distance_covariance_test(
                #         gm_dg_data, n_del_edges_scaled, num_resamples=10
                #     ).pvalue

                if "del_dist" in _gm_analysis_tables_data:
                    quantiles = np.array_split(gm_dgdf.sort_values(gm), 20)
                    gm_del_dist[gm_i] = [q['Del Edges Count'].sum() / de_count for q in quantiles]

            if "mi" in _gm_analysis_tables_data:
                for mi_i in range(_iterations):
                    mi_res[mi_i] = sk_feats.mutual_info_regression(
                        gm_dgdf.loc[:, graph_metrics].values,
                        n_del_edges_scaled,
                        n_neighbors=3
                    )
                mi_res = np.median(mi_res, axis=0)

            gm_info = [*gm_rel_info, gm_sens_attr, gm_dg]

            if "mi" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["mi"].append([*gm_info, *mi_res])
            if "wd" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["wd"].append([*gm_info, *wd_res])
            if "kl" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["kl"].append([*gm_info, *kl_res])
            # if "dcor" in _gm_analysis_tables_data:
            #     _gm_analysis_tables_data["dcor"].append([*gm_info, *dcor_res])
            #     _gm_analysis_tables_data["dcor_pval"].append([*gm_info, *dcor_pval])
            if "del_dist" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["del_dist"].append([*gm_info, *gm_del_dist])


if __name__ == "__main__":
    """It works only when called from outside of the scripts folder as a script (not as a module)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', '--e', required=True)
    parser.add_argument('--base_plots_path', '--bpp', default=os.path.join('scripts', 'plots'))
    parser.add_argument('--gpu_id', default=1)
    parser.add_argument('--psi_impact', action="store_true")
    args = parser.parse_args()

    consumer_group_map = {
        'gender|age': {'M|F': 'M|O', 'M|M': 'M|Y', 'F|F': 'F|O', 'F|M': 'F|Y'},
        'onehot_feat0|onehot_feat13': {'0|0': '0|0', '0|1': '0|1', '1|0': '1|0', '1|1': '1|1'},
    }

    group_name_map = {
        "M|O": "Older Males",
        "M|Y": "Younger Males",
        "F|O": "Older Females",
        "F|Y": "Younger Females",
    }

    if args.exp_path[-1] != os.sep:
        args.exp_path += os.sep

    path_split_key = 'dp_perturbations' if 'dp_perturbations' in args.exp_path else 'dp_explanations'
    _, dset, mod, _, _, s_attr, eps, cid, _ = args.exp_path.split(path_split_key)[1].split(os.sep)
    eps = eps.replace('epochs_', '')

    model_files = os.scandir(os.path.join(os.path.dirname(sys.path[0]), 'saved'))
    model_file_pattern = re.compile(f"{mod}-{dset.upper()}-" + r"(\w{3}-\d{2}-\d{4})_\d{2}-\d{2}-\d{2}\.pth")
    model_file = [f.path for f in model_files if re.match(model_file_pattern, f.name) is not None][0]
    print("Model file:", model_file)

    config = Config(
        model=mod,
        dataset=dset,
        config_file_list=[os.path.join(current_file, '..', 'config', 'perturbation', 'base_perturbation.yaml')],
        config_dict={"gpu_id": args.gpu_id}
    )
    perturbation_config = config.update_base_perturb_data(os.path.join(current_file, '..', args.exp_path, 'config.pkl'))
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
        model_file,
        perturbation_config
    )

    # Users policies
    zerousers_pol = 'ZN'
    furthestusers_pol = 'FR'
    interrecency_pol = 'IR'

    # Items policies
    itemspref_pol = 'IP'
    timelessitems_pol = 'IT'
    pagerankitems_pol = 'PR'

    policy_order_base = [
        zerousers_pol,
        furthestusers_pol,
        interrecency_pol,
        itemspref_pol,
        timelessitems_pol,
        pagerankitems_pol
    ]

    palette = dict(zip(policy_order_base, sns.color_palette("colorblind")))
    pol_hatches = dict(zip(policy_order_base, ['X', '.', '/', 'O', '*']))

    policy_map = {
        'users_zero_constraint': zerousers_pol,
        'users_furthest_constraint': furthestusers_pol,
        'items_preference_constraint': itemspref_pol,
        'users_interaction_recency_constraint': interrecency_pol,
        'items_timeless_constraint': timelessitems_pol,
        'items_pagerank_constraint': pagerankitems_pol
    }

    pol_key = 'perturbation_policies' if 'perturbation_policies' in config else 'explainer_policies'
    raw_exp_policies = [k for k, v in config[pol_key].items() if v and k in policy_map]
    exp_policies = [policy_map[k] for k in raw_exp_policies]
    curr_policy = '+'.join(exp_policies)

    edge_additions = config['edge_additions']
    eval_metric = config['eval_metric'].upper()
    plots_path = os.path.join(args.base_plots_path, dset, mod, s_attr, f"{cid}_{curr_policy}")
    if args.psi_impact:
        exp_policies_ratios = [config[k + "_ratio"] for k in raw_exp_policies]
        ratios_str = f" ({'+'.join(map(str, exp_policies_ratios))})"
        plots_path += ratios_str
        curr_policy += ratios_str
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    checkpoint = torch.load(model_file)
    orig_test_pref_data = pref_data_from_checkpoint(config, checkpoint, train_data, test_data)
    orig_valid_pref_data = pref_data_from_checkpoint(config, checkpoint, train_data, valid_data)

    demo_group_map = dataset.field2id_token[s_attr]

    evaluator = evaluation.Evaluator(config)
    for _pref_data, _eval_data in zip([orig_test_pref_data, orig_valid_pref_data], [test_data.dataset, valid_data.dataset]):
        _pref_data['Demo Group'] = [
            demo_group_map[dg] for dg in dataset.user_feat[s_attr][_pref_data['user_id']].numpy()
        ]
        _pref_data["Demo Group"] = _pref_data["Demo Group"].map(consumer_group_map[s_attr.lower()]).to_numpy()

        metric_result = evaluation.compute_metric(evaluator, _eval_data, _pref_data, 'cf_topk_pred', 'ndcg')
        _pref_data['Value'] = metric_result[:, -1]
        _pref_data['Quantile'] = _pref_data['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)

    batch_exp = config['user_batch_exp']
    exps, rec_model_preds, test_model_preds = utils.load_dp_perturbations_file(args.exp_path)
    best_exp = utils.get_best_pert_early_stopping(exps[0], config)

    pert_edges = best_exp[utils.pert_col_index('del_edges')]

    _, valid_pert_df, test_pert_df = extract_metrics_from_perturbed_edges(
        {(dset, s_attr): pert_edges},
        models=[mod],
        metrics=["NDCG", "Recall"],
        models_path=os.path.join(current_file, os.pardir, 'saved'),
        on_bad_models='ignore',
        remap=False
    )

    test_pert_df = test_pert_df[test_pert_df['Metric'].str.upper() == eval_metric]
    valid_pert_df = valid_pert_df[valid_pert_df['Metric'].str.upper() == eval_metric]
    for _pert_df in [test_pert_df, valid_pert_df]:
        _pert_df['Quantile'] = _pert_df['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)
        _pert_df["Demo Group"] = _pert_df["Demo Group"].map(consumer_group_map[s_attr.lower()]).to_numpy()

    # print(f'{"*" * 15} Test {"*" * 15}')
    # print(f'{"*" * 15} {s_attr.title()} {"*" * 15}')
    # for dg, sa_dg_df in test_pert_df.groupby('Demo Group'):
    #     print(f'\n{"*" * 15} {dg.title()} {"*" * 15}')
    #     print(sa_dg_df.describe())

    dgs = list(consumer_group_map[s_attr.lower()].values())
    orig_pert_pval_dict = {'Valid': {}, 'Test': {}}
    plot_df_data = []
    for orig_dp_df, pert_dp_df, split in zip(
        [orig_test_pref_data, orig_valid_pref_data],
        [test_pert_df, valid_pert_df],
        ['Test', 'Valid']
    ):
        orig_pert_pval_dict[split][eval_metric] = scipy.stats.kruskal(
            orig_dp_df['Value'], pert_dp_df['Value']
        ).pvalue

        total = orig_dp_df['Value'].mean()
        metr_per_group = [orig_dp_df.loc[orig_dp_df['Demo Group'] == dg, 'Value'].to_numpy() for dg in dgs]
        metr_per_group_means = [gr_metr.mean() for gr_metr in metr_per_group]
        _dp = evaluation.compute_DP(*metr_per_group_means)
        pval = scipy.stats.kruskal(*metr_per_group).pvalue
        plot_df_data.append([_dp, split, 'Orig', *metr_per_group_means, total, pval])

        total = pert_dp_df['Value'].mean()
        metr_per_group = [pert_dp_df.loc[pert_dp_df['Demo Group'] == dg, 'Value'].to_numpy() for dg in dgs]
        metr_per_group_means = [gr_metr.mean() for gr_metr in metr_per_group]
        _dp = evaluation.compute_DP(*metr_per_group_means)
        pval = scipy.stats.kruskal(*metr_per_group).pvalue
        plot_df_data.append([_dp, split, curr_policy, *metr_per_group_means, total, pval])

        try:
            orig_pert_pval_dict[split]['DP'] = scipy.stats.wilcoxon(
                orig_dp_df.sort_values('user_id')['Value'].to_numpy(),
                pert_dp_df.sort_values('user_id')['Value'].to_numpy()
            ).pvalue
        except ValueError:  # zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
            # highest pvalue because the distributions are equal
            orig_pert_pval_dict[split]['DP'] = 1.0

    dp_plot_df = pd.DataFrame(plot_df_data, columns=['$\Delta$' + eval_metric, 'Split', 'Policy', *dgs, eval_metric, 'pvalue'])
    dp_plot_df.to_markdown(os.path.join(plots_path, 'DP_barplot.md'), index=False)
    dp_plot_df.to_latex(os.path.join(plots_path, 'DP_barplot.tex'), index=False)
    dp_plot_df.to_csv(os.path.join(plots_path, 'DP_barplot.csv'), index=False)
    with open(os.path.join(plots_path, 'orig_pert_pval_dict.pkl'), 'wb') as f:
        pickle.dump(orig_pert_pval_dict, f)
    print(dp_plot_df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.barplot(x='Split', y='$\Delta$' + eval_metric, data=dp_plot_df, hue='Policy', ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_path, 'DP_barplot.png'), bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()


    # Graph metrics analysis
    dset_plots_path = os.path.join(args.base_plots_path, dset)
    gmd_path = f'{dset}_graph_metrics_df.parquet'
    if os.path.exists(os.path.join(dset_plots_path, gmd_path)):
        graph_metrics_df = pd.read_parquet(os.path.join(dset_plots_path, gmd_path))
    else:
        graph_metrics_df = None

    gm_metrics_base = ['Degree', 'Density', 'Reachability']
    gm_dep_order = np.array(gm_metrics_base)
    if graph_metrics_df is None:
        graph_metrics_df = evaluation.GraphMetricsExtractor(
            train_data.dataset,
            graph_metrics_df=graph_metrics_df,
            upi_kwargs={'sensitive_attribute': [config['sensitive_attribute']]},
            metrics=gm_metrics_base  # "all"
        ).extract_graph_metrics_per_node()

        if any('UPI ' + sa.title() in graph_metrics_df.columns for sa in config['sensitive_attribute']):
            gm_dep_order = list(gm_dep_order[gm_dep_order != 'UPI'])
            for col in graph_metrics_df.columns:
                if 'UPI' in col:
                    gm_dep_order.append(col)
            gm_dep_order = np.array(gm_dep_order)

        last_user_id = train_data.dataset.user_num - 1
        graph_mdf = graph_metrics_df.set_index('Node')
        graph_mdf.loc[:last_user_id, 'Node Type'] = 'User'
        graph_mdf.loc[(last_user_id + 1):, 'Node Type'] = 'Item'
        graph_metrics_df = graph_mdf.reset_index()

        pg = sns.PairGrid(graph_metrics_df, hue='Node Type')
        pg.map_diag(sns.histplot)
        pg.map_offdiag(sns.scatterplot)
        pg.add_legend()
        pg.figure.savefig(os.path.join(dset_plots_path, f'{dset}_graph_metrics_pair_grid.png'))
        plt.close(pg.figure)
        del pg

    graph_metrics_df.to_parquet(os.path.join(dset_plots_path, gmd_path))

    de = pert_edges.copy()
    de_count = de.shape[1]

    # remove user and item id 0 padding
    de -= 1
    de[1] -= 1

    # each edge is counted once for one node, e.g. user, and once for the other, e.g. item (it is equal to a bincount)
    graph_metrics_df['Del Edges Count'] = np.bincount(de.flatten(), minlength=len(graph_metrics_df))

    user_metadata = orig_test_pref_data.copy(deep=True)
    user_metadata['user_id'] -= 1  # reindexing to zero
    user_metadata['Sens Attr'] = s_attr

    sens_gmdf = graph_metrics_df.join(user_metadata[['user_id', 'Sens Attr', 'Demo Group']].set_index('user_id'), on='Node').fillna('Item')

    gm_analysis_tables_data = {"del_dist": []}

    # dset_stats = os.path.join(base_all_plots_path, 'datasets_stats.csv')
    # if os.path.exists(dset_stats):
    #     dsets_df = pd.read_csv(dset_stats, index_col=0)
    #     if _dataset in dsets_df.columns:
    #         from pandas.api.types import is_numeric_dtype
    #         graph_mean_dg = sens_gmdf.groupby(['Sens Attr', 'Demo Group']).mean().loc[_s_attr, gm_dep_order]
    #         graph_gini_dg = sens_gmdf.groupby(['Sens Attr', 'Demo Group']).agg(
    #             {col: gini for col in sens_gmdf.columns if is_numeric_dtype(sens_gmdf[col])}
    #         ).loc[_s_attr, gm_dep_order]
    #         for gm in gm_dep_order:
    #             for gm_str, graph_stats_dg in zip(
    #                 ["bar" + "{" + f"{gm}" + "}", f"Gini {gm}"], [graph_mean_dg, graph_gini_dg]
    #             ):
    #                 if gm_str not in dsets_df.index:
    #                     gm_stat_dg = ""
    #                 else:
    #                     gm_stat_dg = dsets_df.loc[gm_str, _dataset]
    #                     gm_stat_dg = "" if isinstance(gm_stat_dg, float) else gm_stat_dg  # avoids NaN
    #                     if _s_attr in gm_stat_dg:
    #                         continue

    #                 gm_stat_dg += f" {_s_attr} " + '; '.join(graph_stats_dg[gm].to_frame().reset_index().apply(
    #                     lambda x: f"{x['Demo Group']} : {x[gm]:{'.1f' if gm == 'Degree' and 'bar' in gm_str else '.2f'}}", axis=1
    #                 ).values)
    #                 dsets_df.loc[gm_str, _dataset] = gm_stat_dg

    #                 dsets_df.to_csv(dset_stats)

    compute_graph_metrics_analysis_tables_data(
        sens_gmdf,
        gm_analysis_tables_data,
        [dset, mod, curr_policy],
        gm_metrics_base,
    )

    # def del_dist_applymap(twentiles):
    #     offset = 5
    #     quartiles = np.array([sum(twentiles[offset * i: offset * (i + 1)]) for i in range(20 // offset)])
    #     highest_idx = np.argmax(quartiles)
    #     return f"{del_dist_map[highest_idx]} ({quartiles[highest_idx] * 100:.1f}\%)"

    # def del_dist_hl(del_dist_row, row_sattr):
    #     new_row = del_dist_row.copy(deep=True)
    #     row_model = del_dist_row.name
    #     for row_cols, row_col_val in del_dist_row.items():
    #         if adv_groups_map[(row_cols[0], row_model, row_sattr)] == row_cols[2]:
    #             new_row.loc[row_cols] = ("\hl{" + str(row_col_val) + "}")
    #         else:
    #             new_row.loc[row_cols] = (str(row_col_val))
    #     return new_row

    del_dist_giant_cols = [
        "Dataset", "Model", "Policy", "Sens Attr", "Graph Metric",
        "Demo Group", "Del Edges Distribution", "Quartile"
    ]
    del_dist_giant_path = os.path.join(plots_path, 'del_dist_giant.csv')
    for dep_type, gm_dep_data in gm_analysis_tables_data.items():
        if not gm_dep_data:
            continue

        gm_dep_df = pd.DataFrame(
            gm_dep_data,
            columns=["Dataset", "Model", "Policy", "Sens Attr", "Demo Group", *gm_dep_order]
        ).drop_duplicates(
            subset=["Dataset", "Model", "Policy", "Sens Attr", "Demo Group"]  # removes duplicated gms of item nodes
        )
        gm_dep_df = gm_dep_df.melt(
            ['Dataset', 'Model', 'Policy', 'Sens Attr', 'Demo Group'],
            var_name="Graph Metric", value_name="Value"
        )

        for _pol, gmd_pol_df in gm_dep_df.groupby("Policy"):
            pol_del_dist_giant = []

            gmd_models = gmd_pol_df['Model'].unique()

            # del_fig, del_axs = {}, {}
            # for gm_mod in gmd_models:
            #     del_fig[gm_mod], del_axs[gm_mod] = plt.subplots(
            #         len(gmd_pol_df['Sens Attr'].unique()),
            #         len(gmd_pol_df['Graph Metric'].unique()),
            #         sharex=True, sharey='row', figsize=(25, 10)
            #     )

            for gm_sa_i, (gm_sa, gmd_df) in enumerate(gmd_pol_df.groupby('Sens Attr')):
                # for RQ3
                # gmd_df = gmd_sa_df[(gmd_sa_df["Model"] != 'NGCF') & (gmd_sa_df["Policy"] == mondel_pol)]
                # gmd_df = gmd_sa_df[(gmd_sa_df["Policy"] == mondel_pol)]
                #####
                gm_dep_pivot = gmd_df[["Dataset", "Model", "Demo Group", "Graph Metric", "Value"]].pivot(
                    index=['Model'],
                    columns=['Dataset', 'Graph Metric', 'Demo Group']
                ).droplevel(0, axis=1).reindex(
                    gm_dep_order, axis=1, level=1
                ).reindex(
                    ["Item", *consumer_group_map.get(gm_sa, {}).values()], axis=1, level=2
                )
                gm_dep_pivot.columns = gm_dep_pivot.columns.map(lambda x: (*x[:2], group_name_map.get(x[2], x[2])))

                if dep_type == "del_dist":
                    dset = gm_dep_pivot.columns.get_level_values(0).unique()[0]
                    # if gm_sa.lower() == 'gender' and \
                       # 'ml-1m' in gm_dep_pivot.columns.get_level_values(0).unique() and \
                    for gm_dep_model in gm_dep_pivot.index:
                        del_dist_plot_df = gm_dep_pivot.loc[gm_dep_model, dset].copy(deep=True)
                        len_dist_plot_df = len(del_dist_plot_df)
                        parts = 4
                        del_dist_plot_df = del_dist_plot_df.apply(
                            # lambda x: np.array([0] + [sum(x[2 * i: 2 * (i + 1)]) for i in range(parts)])
                            lambda x: np.array([sum(x[(20 // parts) * i: (20 // parts) * (i + 1)]) for i in range(parts)])
                        )
                        del_dist_plot_df = del_dist_plot_df.to_frame().explode(gm_dep_model)
                        del_dist_plot_df.rename(columns={gm_dep_model: 'Del Edges Distribution'}, inplace=True)
                        col_name = 'Del Edges Distribution'
                        del_dist_plot_df[col_name] = del_dist_plot_df[col_name].astype(float)
                        # del_dist_plot_df[col_name] = del_dist_plot_df.reset_index().groupby(
                        #     ['Graph Metric', 'Demo Group']
                        # )[col_name].cumsum().to_numpy()  # transformation to CDF

                        # percentiles = np.tile(np.arange(0, parts + 1), len_dist_plot_df)
                        percentiles = np.tile(np.arange(0, parts), len_dist_plot_df)
                        bar_width = percentiles[1] / parts / 2
                        del_dist_plot_df['Percentile'] = percentiles / parts + bar_width

                        del_dist_plot_df_vals = del_dist_plot_df.reset_index().values
                        del_dist_giant_data = np.c_[
                            [dset] * del_dist_plot_df_vals.shape[0],
                            [gm_dep_model] * del_dist_plot_df_vals.shape[0],
                            [_pol]  * del_dist_plot_df_vals.shape[0],
                            [gm_sa]  * del_dist_plot_df_vals.shape[0],
                            del_dist_plot_df_vals
                        ]
                        pol_del_dist_giant.extend(del_dist_giant_data.tolist())

                        # plot_cols = del_dist_plot_df.index.get_level_values(0).unique()
                        # plot_cols = [x for x in gm_dep_order if x in plot_cols]  # reordering
                        # for _del_ax, plot_col in zip(del_axs[gm_mod][gm_sa_i], plot_cols):
                        #     sns.barplot(
                        #         data=del_dist_plot_df.loc[plot_col].reset_index(),
                        #         x='Percentile', y=col_name, hue='Demo Group', palette='colorblind', ax=_del_ax
                        #     )
                        #     # sns.scatterplot(
                        #     #     data=del_dist_plot_df.loc[plot_col].reset_index(), legend=False,
                        #     #     x='Percentile', y=col_name, hue='Demo Group', palette='colorblind', ax=_del_ax
                        #     # )
                        #     _del_ax.grid(axis='both', ls=':')
                        #     if gm_sa_i == 0:
                        #         _del_ax.set_title(plot_col)
                        #     if gm_sa_i != len(gmd_pol_df['Sens Attr'].unique()) - 1:
                        #         _del_ax.set_xlabel('')
                        #     if plot_col != plot_cols[0]:
                        #         _del_ax.set_ylabel('')
                        #     # _del_ax.set_xlim((-0.01, 1.01))
                        #     _del_ax.tick_params(
                        #         axis='both',
                        #         which='both',
                        #         bottom=gm_sa_i == len(gmd_pol_df['Sens Attr'].unique()) - 1,
                        #         left=gm_sa_i == 0,
                        #         labelbottom=gm_sa_i == len(gmd_pol_df['Sens Attr'].unique()) - 1,
                        #         labelleft=gm_sa_i == 0,
                        #     )

                        # bottom, top = del_axs[0].get_ylim()
                        # whisk_h = abs(top - bottom) / 30

                        # for _del_ax in del_axs:
                        #     _del_ax.plot(
                        #         np.repeat([0, 0.25, 0.5, 0.75, 1.00], 4),
                        #         [top - whisk_h / 2, top - whisk_h, top, top - whisk_h / 2] * 5,
                        #         'k'
                        #     )
                        #     for x_label, dd_map_label in zip([0.125, 0.375, 0.625, 0.875], del_dist_map):  # del_dist_map2:
                        #         _del_ax.annotate(
                        #             dd_map_label, (x_label, top - whisk_h / 2),
                        #             xytext=(0, (top - whisk_h / 2) / 2),
                        #             textcoords="offset points",
                        #             ha='center',
                        #             va='bottom',
                        #         )

                #     gm_dep_pivot = gm_dep_pivot.applymap(del_dist_applymap)
                #     if gm_sa != "Item":
                #         gm_dep_pivot = gm_dep_pivot.apply(lambda r: del_dist_hl(r, gm_sa), axis=1)
                # else:
                #     gm_dep_pivot = gm_dep_pivot.round(2)

                # gm_dep_pivot.to_latex(
                #     os.path.join(plots_path, f"{dep_type}_{_pol}_table_graph_metrics_{gm_sa}_{exp_data_name}.tex"),
                #     column_format="c" * (gm_dep_pivot.shape[1] + 1),
                #     multicolumn_format="c",
                #     escape=False
                # )

            pol_del_dist_giant_df = pd.DataFrame(pol_del_dist_giant, columns=del_dist_giant_cols)
            pol_del_dist_giant_df.to_csv(del_dist_giant_path, index=False)

            # for gm_mod in gmd_models:
            #     del_fig[gm_mod].tight_layout()
            #     del_fig[gm_mod].savefig(
            #         os.path.join(plots_path, f"{gm_mod}_del_dist_{_pol}_plot_{dset}.png"),
            #         bbox_inches="tight", pad_inches=0, dpi=250
            #     )
