import os
import sys
import argparse

import torch
import polars as pl

current_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file, os.pardir))

import fa4gcf.utils as utils
from fa4gcf.data import Dataset
from fa4gcf.config import Config


if __name__ == "__main__":
    """It works only when called from outside of the scripts folder as a script (not as a module)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--sensitive_attribute', '-sa', required=True)
    parser.add_argument('--scores_dataframe_path', '-sdf', required=True)
    parser.add_argument('--gpu_id', default=1)
    args, _ = parser.parse_known_args()

    consumer_group_map = {
        'gender': {'M': 'M', 'F': 'F'},
        'age': {'M': 'Y', 'F': 'O'},
        'user_wide_zone': {'M': 'America', 'F': 'Other'}
    }

    group_name_map = {
        "M": "Males",
        "F": "Females",
        "Y": "Younger",
        "O": "Older",
        "America": "America",
        "Other": "Other"
    }

    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[
            os.path.join(current_file, '..', 'config', 'base_config.yaml'),
            os.path.join(current_file, '..', 'config', 'dataset', f'{args.dataset}.yaml')
        ],
        config_dict={"gpu_id": args.gpu_id, "sensitive_attribute": args.sensitive_attribute}
    )

    dataset = Dataset(config)
    train_data, valid_data, test_data = utils.data_preparation(config, dataset)

    scores_df = pl.read_csv(args.scores_dataframe_path, separator='\t')  # user_id, item_id, score
    scores_df = scores_df.with_columns(
        pl.col('user_id').cast(pl.String).map_elements(
            dataset.field2token_id[config['USER_ID_FIELD']].__getitem__, return_dtype=pl.Int32
        ),
        pl.col('item_id').cast(pl.String).map_elements(
            dataset.field2token_id[config['ITEM_ID_FIELD']].__getitem__, return_dtype=pl.Int32
        )
    )

    scores = torch.full((dataset.user_num, dataset.item_num), fill_value=-torch.inf, dtype=torch.float32)
    scores_dict = scores_df.to_torch('dict', dtype={'score': pl.Float32})
    scores[(scores_dict['user_id'], scores_dict['item_id'])] = scores_dict['score']

    model = utils.get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = utils.get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    print(
        "Dataset:", config["dataset"],
        "Model:", config["model"],
        "Sensitive attribute:", config["sensitive_attribute"]
    )
    for split, eval_data in zip(['Test', 'Valid'], [test_data, valid_data]):
        result = trainer.evaluate_from_scores(eval_data, scores)
        print(split)
        print(result, end='\n\n')
