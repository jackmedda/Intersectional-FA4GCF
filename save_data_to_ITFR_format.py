import pickle
import os
import sys

import numpy as np
import pandas as pd

import fa4gcf.utils as utils


if __name__ == "__main__":
    reference_model = "XSimGCL"

    model_files = [f.path for f in os.scandir(os.path.join(sys.path[0], 'saved')) if f.name.startswith(reference_model)]
    for model_file in model_files:
        dataset_name = model_file.split('-', 1)[1][::-1].split('-', 5)[-1][::-1].lower()
        config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
            model_file,
            {'data_path': os.path.join(sys.path[0], '../../../recsysdatasets', dataset_name)},
        )

        train_dict = pd.DataFrame(train_data.dataset.inter_feat.numpy()).groupby(dataset.uid_field)[dataset.iid_field].apply(list).to_dict()
        # train_dict[0] = []  # padding for user 0
        valid_dict = pd.DataFrame(valid_data.dataset.inter_feat.numpy()).groupby(dataset.uid_field)[dataset.iid_field].apply(list).to_dict()
        # valid_dict[0] = []  # padding for user 0
        test_dict = pd.DataFrame(test_data.dataset.inter_feat.numpy()).groupby(dataset.uid_field)[dataset.iid_field].apply(list).to_dict()
        # test_dict[0] = []  # padding for user 0
        
        save_path = os.path.join(sys.path[0], 'ITFR', 'fa4gcf_data', dataset_name)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'train.txt'), 'w') as f_train:
            for user_id in sorted(train_dict.keys()):
                items = np.array(train_dict[user_id])
                f_train.write(f"{user_id - 1} " + " ".join(map(str, items - 1)) + "\n")
        with open(os.path.join(save_path, 'valid.txt'), 'w') as f_valid:
            for user_id in sorted(valid_dict.keys()):
                items = np.array(valid_dict[user_id])
                f_valid.write(f"{user_id - 1} " + " ".join(map(str, items - 1)) + "\n")
        with open(os.path.join(save_path, 'test.txt'), 'w') as f_test:
            for user_id in sorted(test_dict.keys()):
                items = np.array(test_dict[user_id])
                f_test.write(f"{user_id - 1} " + " ".join(map(str, items - 1)) + "\n")

        user_group = dataset.user_feat.numpy()
        # removing padding user and reindexing sensitive attribute to start from 0 by removing padding sensitive attribute
        user_group = dict(zip(user_group[dataset.uid_field][1:] - 1, user_group[config['sensitive_attribute']][1:] - 1))
        item_group = dict(zip(np.arange(dataset.item_num)[1:] - 1, [0] * (dataset.item_num - 1)))  # we don't do two-sided fairness here
        with open(os.path.join(save_path, 'user_group.pkl'), 'wb') as f:
            pickle.dump(user_group, f)
        with open(os.path.join(save_path, 'item_group.pkl'), 'wb') as f:
            pickle.dump(item_group, f)
