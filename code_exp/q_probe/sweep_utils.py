from q_probe import data
from q_probe.value_probe import ValueNetwork

import torch
import torch.nn.functional as F
import numpy as np

import random


def sample_list_to_dict(samples):
    sample_dict = {}
    for sample in samples:
        if sample["idx"] not in sample_dict:
            sample_dict[sample["idx"]] = [sample]
        else:
            sample_dict[sample["idx"]].append(sample)
    return sample_dict


def load_samples_to_dict(train_path, test_path, extra_path=None, n_val=200):
    train_samples = data.load_all_samples(train_path)
    sample_dict = sample_list_to_dict(train_samples)

    train_sample_dict = {}
    val_sample_dict = {}
    for idx in sample_dict.keys():
        val_sample_dict[idx] = sample_dict[idx][:n_val]
        train_sample_dict[idx] = sample_dict[idx][n_val:]

    test_samples = data.load_all_samples(test_path)
    test_sample_dict = sample_list_to_dict(test_samples)

    if extra_path is not None:
        extra_samples = data.load_all_samples(extra_path)
        extra_sample_dict = sample_list_to_dict(extra_samples)
    else:
        extra_sample_dict = {}

    return train_sample_dict, val_sample_dict, test_sample_dict, extra_sample_dict


def shortcut_eval_probe(sample_dict, probe, temperature, chunk_size, n_shuffles=1):
    # Compute probe values
    for idx in sample_dict.keys():
        features = np.stack([s["features"] for s in sample_dict[idx]])
        values = probe.get_value(features)
        for i, s in enumerate(sample_dict[idx]):
            s["probe"] = values[i]

    # Compute expected rewards
    reward_list = []
    for idx in sample_dict.keys():
        full_values = torch.tensor(
            np.array([s["probe"] for s in sample_dict[idx]])
        ).flatten()
        full_rewards = torch.tensor(
            np.array([s["reward"] for s in sample_dict[idx]])
        ).flatten()

        for _ in range(n_shuffles):
            # shuffle
            perm = torch.randperm(len(full_values))
            values = full_values[perm]
            rewards = full_rewards[perm]
            # reshape
            num_chunks = len(sample_dict[idx]) // chunk_size
            values = torch.stack(values.split(chunk_size)[:num_chunks])
            rewards = torch.stack(rewards.split(chunk_size)[:num_chunks])
            # define weights
            if temperature > 0:
                weights = F.softmax(values / temperature, dim=1)
            else:
                weights = torch.zeros(values.shape)
                weights[torch.arange(len(values)), torch.argmax(values, dim=1)] = 1.0
            # compute expected reward
            avg_reward = (rewards * weights).sum(dim=1).mean()
            reward_list.append(avg_reward.item())
    print("Expected reward:", sum(reward_list) / len(reward_list))
    return reward_list


def get_pg_train_data(sample_dict, size, n_probs=None):
    trunc_sample_dict = {}
    for idx in sample_dict.keys():
        trunc_sample_dict[idx] = sample_dict[idx][:size]

    feature_list = []  # reshape X to be (n_prompts, n_samples, n_features)
    reward_list = []  # reshape y to be (n_samples, n_samples,)
    means = []

    keys = list(trunc_sample_dict.keys())
    random.shuffle(keys)
    if n_probs is not None:
        keys = keys[:n_probs]

    for idx in keys:
        features = np.stack([s["features"] for s in trunc_sample_dict[idx]])
        rewards = np.stack(
            [s["reward"] for s in trunc_sample_dict[idx]], dtype=np.float64
        )

        feature_list.append(features)
        reward_list.append(rewards)
        means.append(np.mean(rewards))

    normalized_rewards = []
    mean_of_means = np.mean(means)
    for r, mean in zip(reward_list, means):
        normalized_rewards.append(r - mean_of_means)
        # normalized_rewards.append(r - mean)
    print("Mean of means:", mean_of_means)

    return [(f, l) for f, l in zip(feature_list, normalized_rewards)]


def get_mse_train_data(sample_dict, size, n_probs=None):
    train_samples = []

    keys = list(sample_dict.keys())
    random.shuffle(keys)
    if n_probs is not None:
        keys = keys[:n_probs]

    for idx in keys:
        train_samples.extend(sample_dict[idx][:size])
    train_features = [samples["features"] for samples in train_samples]
    train_labels = [samples["reward"] for samples in train_samples]
    return [(f, l) for f, l in zip(train_features, train_labels)]


def eval(
    value_probe,
    val_sample_dict,
    test_sample_dict,
    temperature,
    chunk_size,
    extra_samples=None,
):
    if len(val_sample_dict[0]) == 0:
        val_rewards = 0.0
    else:
        val_rewards = shortcut_eval_probe(
            val_sample_dict, value_probe, temperature, chunk_size
        )
        val_rewards = torch.mean(torch.tensor(val_rewards)).item()

    test_rewards = shortcut_eval_probe(
        test_sample_dict, value_probe, temperature, chunk_size
    )
    test_rewards = torch.mean(torch.tensor(test_rewards)).item()

    if extra_samples is not None:
        extra_rewards = shortcut_eval_probe(
            extra_samples, value_probe, temperature, chunk_size
        )
        extra_rewards = torch.mean(torch.tensor(extra_rewards)).item()
    else:
        extra_rewards = None

    return val_rewards, test_rewards, extra_rewards


def train(probe_config, train_data):
    value_probe = ValueNetwork(warmup_data=train_data, device="cuda", **probe_config)
    return value_probe


def train_and_eval(probe_config, train_data, val_sample_dict, test_sample_dict):
    value_probe = ValueNetwork(warmup_data=train_data, device="cuda", **probe_config)
    val_rewards, test_rewards = eval(
        value_probe,
        val_sample_dict,
        test_sample_dict,
        probe_config["temperature"],
        probe_config["chunk_size"],
    )
    return val_rewards, test_rewards, value_probe
