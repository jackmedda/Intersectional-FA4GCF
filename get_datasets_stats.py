import os

import pandas as pd
from recbole.data.utils import create_dataset

from fa4gcf.config import Config


def get_plots_path():
    plots_path = os.path.join(
        script_path,
        'dataset_info'
    )

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


script_path = os.path.dirname(os.path.realpath(__file__))
saved_models_path = os.path.join(script_path, 'saved')

dataset_config_path = os.path.join(script_path, 'config', 'dataset')
perturbation_config_path = os.path.join(script_path, 'config', 'perturbation')

out_data = []
out_columns = [
    '# Users',
    '# Items',
    '# Interactions',
    'Min. Degree per user',
    'Density',
]

consumer_group_map = {
    'gender|age': {'M|F': 'M|O', 'M|M': 'M|Y', 'F|F': 'F|O', 'F|M': 'F|Y'},
    'onehot_feat0|onehot_feat13': {'0|0': '0|0', '0|1': '0|1', '1|0': '1|0', '1|1': '1|1'},
}

sens_attr_map = {
    'gender|age': 'Gender | Age',
    'onehot_feat0|onehot_feat13': 'One-hot feat0 | One-hot feat13',
}

dataset_map = {
    'lastfm-1m': 'LFM1M',
    'ml-1m': 'ML1M',
    'ml-1m_dense': 'ML1MD',
    'kuairec_big': 'KRECB',
    'kuairec_small_watch_ratio_1_inf': 'KRECS',
}

datasets = list(dataset_map.keys())
stats_path = os.path.join(get_plots_path(), 'datasets_stats.csv')
for dset_name in datasets:
    config = Config(
        'LightGCN',
        dset_name,
        config_file_list=[os.path.join(dataset_config_path, f"{dset_name}.yaml")],
        config_dict={'data_path': '/home/recsysdatasets'}
    )
    pert_config_dict = config.update_base_perturb_data(
        os.path.join(perturbation_config_path, f"{dset_name}_perturbation.yaml")
    )
    config.final_config_dict.update(pert_config_dict)
    dataset = create_dataset(config)

    dataset._change_feat_format()
    user_feat = pd.DataFrame(dataset.user_feat.numpy()).iloc[1:]
    sens_info = {}
    for col in consumer_group_map:
        if col in dataset.field2id_token:
            user_feat[col] = user_feat[col].astype(int).map(dataset.field2id_token[col].__getitem__)
            user_feat[col] = user_feat[col].map(consumer_group_map[col])
            col_info = (user_feat[[col]].value_counts() / len(user_feat) * 100).map(lambda x: f"{x:.1f}%")
            sens_info[col] = ' ; '.join(
                col_info.to_frame().reset_index().apply(
                    lambda x: f"{x[col]} : {x['count']}".replace('%', '\%'), axis=1
                ).values
            )

    out_data.append([
        dataset.user_num - 1,
        dataset.item_num - 1,
        dataset.inter_num,
        dataset.history_item_matrix()[2][1:].min().item(),
        f"{1 - dataset.sparsity:.2%}".replace('%', '\%'),
        *[sens_attr_map[s] for s in sens_info.keys()],
        *list(sens_info.values())
    ])
    dset_out_columns = [
        *out_columns,
        'Sensitive Attribute',
        'Representation'
    ]

df = pd.DataFrame(out_data, columns=dset_out_columns, index=[dataset_map[dset_name] for dset_name in datasets])
df.index.name = "Dataset"
df = df.T.fillna('-')

df = df.applymap(lambda x: f'{x:,}' if str(x).isdigit() else x)
print(df)
df.to_csv(stats_path, sep='\t')

with pd.option_context('max_colwidth', None):
    with open(os.path.join(get_plots_path(), 'datasets_stats.tex'), 'w') as tex_file:
        tex_file.write(
            df.to_latex(
                column_format="r"+ "|r" * len(datasets),
                escape=False,
                multirow=True
            ).replace('#', '\#')
        )
