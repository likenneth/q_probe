from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import random

from q_probe import sweep_utils


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--loss", type=str)
    args.add_argument("--seed", type=int)
    args.add_argument("--type", type=int)
    args = args.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.type == 0:
        # Paths for code llama on MBPP
        train_path = "/n/holyscratch01/kempner_dev/Shared/mcts-llm/samples/14765178"
        test_path = "/n/holyscratch01/kempner_dev/Shared/mcts-llm/samples/13643197"
        human_eval_path = (
            "/n/holyscratch01/kempner_dev/Shared/mcts-llm/samples/17018932"
        )
        run_name = f"probs_mbpp_{args.loss}_sweep_{args.seed}"
        data_size = 200
        n_val = 200
    elif args.type == 1:
        # Paths for openai on MBPP w/ llama embeddings
        train_path = "/n/holyscratch01/kempner_dev/Shared/mcts-llm/llama70B_samples_api/base_train/samples/local_kl"
        test_path = "/n/holyscratch01/kempner_dev/Shared/mcts-llm/llama70B_samples_api/base/samples/local_kl"
        human_eval_path = "/n/holyscratch01/kempner_dev/Shared/mcts-llm/llama70B_samples_api/base_he/samples/local_kl"
        run_name = f"probs_llama70B_samples_api_mbpp_{args.loss}_sweep_{args.seed}"
        data_size = 100
        n_val = 0

    # Default settings
    probe_config = {
        "layer": 1,
        "lr": 5e-5,
        "weight_decay": 0.0,
        "num_epochs": 150,
        "tr_pctg": 1.0,
        "batch_size": 1000,
        "loss": args.loss,
        "chunk_size": 10,  # training
        "temperature": 1.0,  # training
        "seed": args.seed,
    }
    if probe_config["loss"] == "pg":
        probe_config["batch_size"] //= probe_config["chunk_size"]

    # Load data
    (
        train_sample_dict,
        val_sample_dict,
        test_sample_dict,
        he_sample_dict,
    ) = sweep_utils.load_samples_to_dict(
        train_path, test_path, human_eval_path, n_val=n_val
    )
    print("Val size:", len(val_sample_dict), len(val_sample_dict[0]))
    print("Test size:", len(test_sample_dict), len(test_sample_dict[0]))

    # Sweep
    if args.type == 0:
        n_probs = [5, 10, 20, 50, 100, 200, 464]
    elif args.type == 1:
        n_probs = [464]
    layers = [1, 2, 3]
    eval_ks = list(range(2, 50, 2))
    eval_temps = [0, 0.01, 0.1, 1.0]
    results = {
        "n_probs": [],
        "seed": [],
        "layer": [],
        "val_rs": [],
        "test_rs": [],
        "he_rs": [],
        "k": [],
        "temp": [],
    }
    for n in n_probs:
        if probe_config["loss"] == "pg":
            train_data = sweep_utils.get_pg_train_data(train_sample_dict, data_size, n)
            print("Train size:", len(train_data), train_data[0][0].shape)
        else:
            train_data = sweep_utils.get_mse_train_data(train_sample_dict, data_size, n)
            print("Train size:", len(train_data), len(train_data[0]))

        for l in layers:
            s = args.seed
            probe_config["seed"] = s
            probe_config["layer"] = l

            probe = sweep_utils.train(probe_config, train_data)

            for k in eval_ks:
                for temp in eval_temps:
                    val_r, test_r, he_r = sweep_utils.eval(
                        probe,
                        val_sample_dict,
                        test_sample_dict,
                        temperature=temp,
                        chunk_size=k,
                        extra_samples=he_sample_dict,
                    )
                    results["n_probs"].append(n)
                    results["seed"].append(s)
                    results["layer"].append(l)
                    results["val_rs"].append(val_r)
                    results["test_rs"].append(test_r)
                    results["he_rs"].append(he_r)
                    results["k"].append(k)
                    results["temp"].append(temp)

    # Save results
    with open(f"results/{run_name}.pkl", "wb") as f:
        pickle.dump(results, f)
