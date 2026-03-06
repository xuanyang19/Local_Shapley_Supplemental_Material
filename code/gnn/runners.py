#!/usr/bin/env python3
"""Method runners for GNN Shapley computation (parallel with convergence)."""

import numpy as np
import torch
from multiprocessing import Pool

from .config import CHECKPOINT_INTERVAL
from .utils import ConvergenceTracker, save_checkpoint, save_results, maybe_print_progress
from .model import train_and_eval_gnn
from .workers import init_worker, _worker_complementary, _worker_lsmr_a


def run_method_standard(method_name, worker_fn, cache, data_np, config, out_dir):
    tag = method_name
    n_train = cache.n_train
    n_jobs = config['n_jobs']

    conv_tracker = ConvergenceTracker(
        n_train, config['check_interval'], config['conv_threshold'])
    phi_sum = np.zeros(n_train, dtype=float)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} (n_jobs={n_jobs}, convergence-based)...")
    print(f"{'='*60}")

    task_args = []
    v_full_vec = None
    if method_name == "tmc":
        print("Precomputing V(D) for TMC...")
        device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        v_full_vec = train_and_eval_gnn(
            cache.train_list, data_np['x'], data_np['y'],
            data_np['edge_index'], data_np['test_indices'],
            config['n_classes'], config['seed'], config['epochs'], device_0)
        print(f"V(D) mean = {np.mean(v_full_vec):.4f}")

    for m in range(config['max_samples']):
        if method_name == "global_mc":
            task_args.append((m, config['n_classes'], config['seed'], config['epochs']))
        elif method_name == "tmc":
            task_args.append((m, config['n_classes'], config['seed'],
                              config['epochs'], v_full_vec, 5))
        else:  # local_mc
            task_args.append((m, config['n_classes'], config['seed'], config['epochs']))

    n_test = len(data_np['test_indices'])
    checkpoint_interval = config.get('checkpoint_interval', CHECKPOINT_INTERVAL)

    with Pool(processes=n_jobs, initializer=init_worker,
              initargs=(cache, data_np)) as pool:
        num_samples_done = 0

        for m, result in enumerate(pool.imap_unordered(worker_fn, task_args)):
            contrib, n_evals = result

            phi_sum += contrib
            num_samples_done = m + 1

            conv_metric = None
            if num_samples_done % config['check_interval'] == 0:
                phi_current = phi_sum / float(num_samples_done)
                conv_result = conv_tracker.check_convergence(
                    phi_current, num_samples_done)
                conv_metric = conv_result["convergence_metric"]

                if conv_result["converged"]:
                    print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                          f"metric={conv_metric:.6f} < {config['conv_threshold']}")
                    pool.terminate()
                    break

            if num_samples_done % checkpoint_interval == 0:
                save_checkpoint(
                    phi_sum, num_samples_done,
                    conv_tracker.convergence_history,
                    out_dir=out_dir, tag=tag, dataset=config['dataset'],
                    seed=config['seed'], n_test=n_test)

            maybe_print_progress(tag, m, conv_metric, config['print_interval'])

    phi_hat = phi_sum / float(num_samples_done)
    save_results(
        phi_hat, num_samples_done,
        conv_tracker.convergence_history,
        out_dir=out_dir, tag=tag, dataset=config['dataset'],
        seed=config['seed'], n_test=n_test)

    return phi_hat, num_samples_done, conv_tracker.convergence_history


def run_lsmr_a(cache, data_np, config, out_dir):
    tag = "lsmr_a"
    n_train = cache.n_train
    max_L = cache.max_L
    n_jobs = config['n_jobs']
    use_reuse = config.get('use_reuse', False)

    conv_tracker = ConvergenceTracker(
        n_train, config['check_interval'], config['conv_threshold'])

    SV_bins = np.zeros((n_train, max_L), dtype=float)
    M_bins = np.zeros((n_train, max_L), dtype=int)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} (n_jobs={n_jobs}, reuse={use_reuse})...")
    print(f"{'='*60}")

    task_args = []
    for m in range(config['max_samples']):
        task_args.append((m, config['n_classes'], config['seed'],
                          config['epochs'], use_reuse))

    n_test = len(data_np['test_indices'])
    checkpoint_interval = config.get('checkpoint_interval', CHECKPOINT_INTERVAL)

    with Pool(processes=n_jobs, initializer=init_worker,
              initargs=(cache, data_np)) as pool:
        num_samples_done = 0

        for m, result in enumerate(pool.imap_unordered(
                _worker_lsmr_a, task_args)):
            result_list, n_evals = result

            for S_idx, Sc_idx, i_pos, L_len, u in result_list:
                j_plus = i_pos - 1
                j_minus = L_len - i_pos - 1

                if j_plus >= 0 and len(S_idx) > 0:
                    SV_bins[S_idx, j_plus] += u
                    M_bins[S_idx, j_plus] += 1
                if j_minus >= 0 and len(Sc_idx) > 0:
                    SV_bins[Sc_idx, j_minus] -= u
                    M_bins[Sc_idx, j_minus] += 1

            num_samples_done = m + 1

            conv_metric = None
            if num_samples_done % config['check_interval'] == 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    means_per_bin = np.where(
                        M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
                phi_current = means_per_bin.mean(axis=1)
                conv_result = conv_tracker.check_convergence(
                    phi_current, num_samples_done)
                conv_metric = conv_result["convergence_metric"]

                if conv_result["converged"]:
                    print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                          f"metric={conv_metric:.6f} < {config['conv_threshold']}")
                    pool.terminate()
                    break

            if num_samples_done % checkpoint_interval == 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    means_per_bin = np.where(
                        M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
                phi_ckpt = means_per_bin.mean(axis=1)
                save_checkpoint(
                    phi_ckpt, num_samples_done,
                    conv_tracker.convergence_history,
                    out_dir=out_dir, tag=tag, dataset=config['dataset'],
                    seed=config['seed'], n_test=n_test)

            maybe_print_progress(tag, m, conv_metric, config['print_interval'])

    with np.errstate(divide='ignore', invalid='ignore'):
        means_per_bin = np.where(
            M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
    phi_hat = means_per_bin.mean(axis=1)

    save_results(
        phi_hat, num_samples_done,
        conv_tracker.convergence_history,
        out_dir=out_dir, tag=tag, dataset=config['dataset'],
        seed=config['seed'], n_test=n_test)

    return phi_hat, num_samples_done, conv_tracker.convergence_history


def run_complementary(cache, data_np, config, out_dir):
    tag = "complementary"
    n_train = cache.n_train
    n_jobs = config['n_jobs']

    conv_tracker = ConvergenceTracker(
        n_train, config['check_interval'], config['conv_threshold'])

    SV_bins = np.zeros((n_train, n_train), dtype=float)
    M_bins = np.zeros((n_train, n_train), dtype=int)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} with bin-based stratification (n_jobs={n_jobs})...")
    print(f"{'='*60}")

    task_args = []
    for m in range(config['max_samples']):
        i_split = (m % n_train) + 1
        task_args.append((m, i_split, config['n_classes'], config['seed'],
                          config['epochs']))

    with Pool(processes=n_jobs, initializer=init_worker,
              initargs=(cache, data_np)) as pool:
        num_samples_done = 0

        for m, result in enumerate(pool.imap_unordered(
                _worker_complementary, task_args)):
            perm_indices, i_pos, total_u, n_evals = result

            S_indices = perm_indices[:i_pos]
            S_comp_indices = perm_indices[i_pos:]
            j_plus = i_pos - 1
            j_minus = n_train - i_pos - 1

            SV_bins[S_indices, j_plus] += total_u
            M_bins[S_indices, j_plus] += 1
            if len(S_comp_indices) > 0 and j_minus >= 0:
                SV_bins[S_comp_indices, j_minus] -= total_u
                M_bins[S_comp_indices, j_minus] += 1

            num_samples_done = m + 1

            conv_metric = None
            if num_samples_done % config['check_interval'] == 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    means_per_bin = np.where(
                        M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
                phi_current = means_per_bin.mean(axis=1)
                conv_result = conv_tracker.check_convergence(
                    phi_current, num_samples_done)
                conv_metric = conv_result["convergence_metric"]

                if conv_result["converged"]:
                    print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                          f"metric={conv_metric:.6f} < {config['conv_threshold']}")
                    pool.terminate()
                    break

            maybe_print_progress(tag, m, conv_metric, config['print_interval'])

    with np.errstate(divide='ignore', invalid='ignore'):
        means_per_bin = np.where(
            M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
    phi_hat = means_per_bin.mean(axis=1)

    n_test = len(data_np['test_indices'])
    save_results(
        phi_hat, num_samples_done,
        conv_tracker.convergence_history,
        out_dir=out_dir, tag=tag, dataset=config['dataset'],
        seed=config['seed'], n_test=n_test)

    return phi_hat, num_samples_done, conv_tracker.convergence_history
