import os
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default="1", type=str)
parser.add_argument('--exclude_datasets', default=None, nargs='+')
parser.add_argument('--exclude_models', default=None, nargs='+')
parser.add_argument('--do_not_skip', action='store_true', help='Do not skip already evaluated experiments')
parser.add_argument('--do_not_skip_if_no_del_dist_giant', action='store_true', help='Do not skip if del_dist_giant.csv is not present')
parser.add_argument('--only_alpha_ablation', action='store_true', help='Only continue alpha ablation experiments')

args = parser.parse_args()
args.exclude_datasets = args.exclude_datasets or []
args.exclude_models = args.exclude_models or []

current_path = os.path.dirname(os.path.realpath(__file__))
plots_path = os.path.join(current_path, 'plots')
alpha_ablation_plots_path = os.path.join(current_path, 'alpha_ablation_plots')
exp_path = os.path.join(current_path, os.pardir, 'experiments', 'dp_perturbations')

script_str = "python3 scripts/eval_info.py --e \"{}\""
alpha_ablation_str = " --bpp {} --alpha_ablation"
gpu_str = f" --gpu_id {args.gpu_id}"

crashed_runs = []
for folder, subfolders, files in os.walk(exp_path):
    if any(subf.isdigit() for subf in subfolders) and 'OLD_' not in folder:
        folder_sep = folder.split(os.sep)
        dset, mod, _, _, sa, _ = folder_sep[-6:]
        if dset in args.exclude_datasets or mod in args.exclude_models:
            continue
        for subf in subfolders:
            if 1 <= int(subf) <= 60:
                if not os.path.exists(os.path.join(folder, subf, "cf_data.pkl")):
                    print("MISSING CF DATA, SKIPPING", os.path.join(folder, subf))
                    continue

                base_plots_path = plots_path if 1 <= int(subf) <= 15 else alpha_ablation_plots_path
                temp_folder = os.path.join(base_plots_path, dset, mod, sa)
                del_dist_giant_missing = False
                os.makedirs(temp_folder, exist_ok=True)
                plots_subf = [f for f in os.listdir(temp_folder) if f.startswith(f"{subf}_")]
                if args.do_not_skip_if_no_del_dist_giant:
                    if len(plots_subf) > 0:
                        del_dist_giant_missing = not os.path.exists(os.path.join(temp_folder, plots_subf[0], 'del_dist_giant.csv'))

                if os.path.exists(temp_folder) and not args.do_not_skip and not del_dist_giant_missing:
                    if len(plots_subf) > 0 and os.path.exists(os.path.join(temp_folder, plots_subf[0], 'DP_barplot.csv')):
                        print("Skipped exp", os.path.join(temp_folder, plots_subf[0]))
                        continue
                run_subprocess_script_str = script_str.format(os.path.join(folder, subf)) + gpu_str
                if 16 <= int(subf) <= 60:
                    run_subprocess_script_str += alpha_ablation_str.format(alpha_ablation_plots_path)
                elif args.only_alpha_ablation:
                    print(f"EXP NUMBER {subf}, SKIPPING BECAUSE ONLY ABLATION", folder)
                    continue
            else:
                print("EXP NUMBER UNKOWN, SKIPPING", folder)
                continue

            print("RUNNING:", run_subprocess_script_str)
            ret_process = subprocess.run(
                run_subprocess_script_str,
                cwd=os.path.dirname(current_path),
                shell=True
            )

            if ret_process.returncode != 0:
                crashed_runs.append(run_subprocess_script_str)
                print("Crashed runs")

print("Crashed runs")
for run in crashed_runs:
    print(run)
