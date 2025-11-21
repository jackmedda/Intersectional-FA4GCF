import os
import pickle
import re
import sys
import argparse

import scipy.stats
import torch
import numpy as np
import pandas as pd

current_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file, os.pardir))

import fa4gcf.utils as utils
from fa4gcf.data import Dataset
from fa4gcf.config import Config
import fa4gcf.evaluation as evaluation
from fa4gcf.utils.case_study import extract_metrics_from_perturbed_edges


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

sensitive_attributes = {
    'lastfm-1m': ['gender|age'],
    'ml-1m': ['gender|age'],
    'kuairec_big': ['onehot_feat0|onehot_feat13'],
    'kuairec_small_watch_ratio_1_inf': ['onehot_feat0|onehot_feat13'],
    'ml-1m_dense': ['gender|age'],
}


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

policy_map = {
    'users_zero_constraint': zerousers_pol,
    'users_furthest_constraint': furthestusers_pol,
    'items_preference_constraint': itemspref_pol,
    'users_interaction_recency_constraint': interrecency_pol,
    'items_timeless_constraint': timelessitems_pol,
    'items_pagerank_constraint': pagerankitems_pol
}


if __name__ == "__main__":
    """It works only when called from outside of the scripts folder as a script (not as a module)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_path', '-sp', required=True)
    parser.add_argument('--no_padding', action='store_true')
    parser.add_argument('--exp_path', '--e', required=True)
    parser.add_argument('--base_plots_path', '--bpp', default=os.path.join('scripts', 'itfr_plots'))
    parser.add_argument('--gpu_id', default=1)
    args, _ = parser.parse_known_args()

    mod = "LightGCN"
    dset = args.scores_path.split(os.sep)[-2]
    s_attr = sensitive_attributes[dset][0]

    config = Config(
        model=mod,
        dataset=dset,
        config_file_list=[
            os.path.join(current_file, '..', 'config', 'base_config.yaml'),
            os.path.join(current_file, '..', 'config', 'dataset', f'{dset}.yaml'),
            os.path.join(current_file, '..', 'config', 'perturbation', 'base_perturbation.yaml'),
            os.path.join(current_file, '..', 'config', 'perturbation', f'{dset}_perturbation.yaml')
        ],
        config_dict={"gpu_id": args.gpu_id, "sensitive_attribute": s_attr}
    )

    dataset = Dataset(config)
    train_data, valid_data, test_data = utils.data_preparation(config, dataset)

    scores_df = np.load(args.scores_path)
    # need to add padding user (row) and item (column) with both ids 0 to the scores tensor
    if not args.no_padding:
        padding_row = np.full((1, scores_df.shape[1]), fill_value=-np.inf)
        scores_df = np.vstack([padding_row, scores_df])
        padding_col = np.full((scores_df.shape[0], 1), fill_value=-np.inf)
        scores_df = np.hstack([padding_col, scores_df])

    scores = torch.tensor(scores_df, dtype=torch.float32).to(config['device'])
    user_data = np.arange(scores_df.shape[0])
    _, model_topk_idx = evaluation.get_top_k(scores, topk=max(config['topk']))
    model_topk_idx = model_topk_idx.detach().cpu().numpy()

    itfr_test_pref_data = pd.DataFrame(zip(user_data, model_topk_idx), columns=['user_id', 'topk_pred'])

    if args.exp_path[-1] != os.sep:
        args.exp_path += os.sep

    path_split_key = 'dp_perturbations' if 'dp_perturbations' in args.exp_path else 'dp_explanations'
    _, dset, mod, _, _, s_attr, eps, cid, _ = args.exp_path.split(path_split_key)[1].split(os.sep)
    eps = eps.replace('epochs_', '')

    model_files = os.scandir(os.path.join(os.path.dirname(sys.path[0]), 'saved'))
    model_file_pattern = re.compile(f"{mod}-{dset.upper()}-" + r"(\w{3}-\d{2}-\d{4})_\d{2}-\d{2}-\d{2}\.pth")
    model_file = [f.path for f in model_files if re.match(model_file_pattern, f.name) is not None][0]
    print("Model file:", model_file)

    perturbation_config = config.update_base_perturb_data(os.path.join(current_file, '..', args.exp_path, 'config.pkl'))
    # perturbation_config, _, _, _, _, _ = utils.load_data_and_model(
    #     model_file,
    #     perturbation_config
    # )

    pol_key = 'perturbation_policies' if 'perturbation_policies' in perturbation_config else 'explainer_policies'
    raw_exp_policies = [k for k, v in perturbation_config[pol_key].items() if v and k in policy_map]
    exp_policies = [policy_map[k] for k in raw_exp_policies]
    curr_policy = '+'.join(exp_policies)

    eval_metric = config['eval_metric'].upper()
    plots_path = os.path.join(args.base_plots_path, dset, mod, s_attr, f"{cid}_{curr_policy}")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    demo_group_map = dataset.field2id_token[s_attr]

    evaluator = evaluation.Evaluator(perturbation_config)
    itfr_test_pref_data['Demo Group'] = [
        demo_group_map[dg] for dg in dataset.user_feat[s_attr][itfr_test_pref_data['user_id']].numpy()
    ]
    itfr_test_pref_data["Demo Group"] = itfr_test_pref_data["Demo Group"].map(consumer_group_map[s_attr.lower()]).to_numpy()

    metric_result = evaluation.compute_metric(evaluator, test_data.dataset, itfr_test_pref_data, 'topk_pred', eval_metric.lower())
    itfr_test_pref_data['Value'] = metric_result[:, -1]
    itfr_test_pref_data['Quantile'] = itfr_test_pref_data['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)

    exps, rec_model_preds, test_model_preds = utils.load_dp_perturbations_file(args.exp_path)
    best_exp = utils.get_best_pert_early_stopping(exps[0], config)

    pert_edges = best_exp[utils.pert_col_index('del_edges')]

    _, _, test_pert_df = extract_metrics_from_perturbed_edges(
        {(dset, s_attr): pert_edges},
        models=[mod],
        metrics=["NDCG", "Recall"],
        models_path=os.path.join(current_file, os.pardir, 'saved'),
        on_bad_models='ignore',
        remap=False
    )

    test_pert_df = test_pert_df[test_pert_df['Metric'].str.upper() == eval_metric]
    test_pert_df['Quantile'] = test_pert_df['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)
    test_pert_df["Demo Group"] = test_pert_df["Demo Group"].map(consumer_group_map[s_attr.lower()]).to_numpy()

    dgs = list(consumer_group_map[s_attr.lower()].values())
    orig_pert_pval_dict = {'Valid': {}, 'Test': {}}
    plot_df_data = []
    orig_pert_pval_dict['Test'][eval_metric] = scipy.stats.kruskal(
        itfr_test_pref_data['Value'], test_pert_df['Value']
    ).pvalue

    total = itfr_test_pref_data['Value'].mean()
    metr_per_group = [itfr_test_pref_data.loc[itfr_test_pref_data['Demo Group'] == dg, 'Value'].to_numpy() for dg in dgs]
    metr_per_group_means = [gr_metr.mean() for gr_metr in metr_per_group]
    _dp = evaluation.compute_DP(*metr_per_group_means)
    pval = scipy.stats.kruskal(*metr_per_group).pvalue
    plot_df_data.append([_dp, 'Test', 'ITFR', *metr_per_group_means, total, pval])

    total = test_pert_df['Value'].mean()
    metr_per_group = [test_pert_df.loc[test_pert_df['Demo Group'] == dg, 'Value'].to_numpy() for dg in dgs]
    metr_per_group_means = [gr_metr.mean() for gr_metr in metr_per_group]
    _dp = evaluation.compute_DP(*metr_per_group_means)
    pval = scipy.stats.kruskal(*metr_per_group).pvalue
    plot_df_data.append([_dp, 'Test', curr_policy, *metr_per_group_means, total, pval])

    try:
        orig_pert_pval_dict['Test']['DP'] = scipy.stats.wilcoxon(
            itfr_test_pref_data.sort_values('user_id')['Value'].to_numpy(),
            test_pert_df.sort_values('user_id')['Value'].to_numpy()
        ).pvalue
    except ValueError:  # zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        # highest pvalue because the distributions are equal
        orig_pert_pval_dict['Test']['DP'] = 1.0

    dp_plot_df = pd.DataFrame(plot_df_data, columns=['$\Delta$' + eval_metric, 'Split', 'Policy', *dgs, eval_metric, 'pvalue'])
    dp_plot_df.to_markdown(os.path.join(plots_path, 'DP_barplot.md'), index=False)
    dp_plot_df.to_latex(os.path.join(plots_path, 'DP_barplot.tex'), index=False)
    dp_plot_df.to_csv(os.path.join(plots_path, 'DP_barplot.csv'), index=False)
    with open(os.path.join(plots_path, 'orig_pert_pval_dict.pkl'), 'wb') as f:
        pickle.dump(orig_pert_pval_dict, f)
    print(dp_plot_df)

