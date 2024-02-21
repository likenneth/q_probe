from tqdm import tqdm
import os
from glob import glob
import pickle
import numpy as np

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter


def ppprint(code):
    print(highlight(code, PythonLexer(), TerminalFormatter()))


from bigcode_eval import tasks
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from bigcode_eval.tasks.custom_metrics.code_eval import estimate_pass_at_k


def get_reward_fn(task, idx):
    if isinstance(
        task,
        (tasks.mbpp.MBPP, tasks.mbpp_train_val.MBPP, tasks.humaneval.GeneralHumanEval),
    ):

        def reward_fn(generation):
            processed = task.postprocess_generation(generation, idx)
            reference = task.get_reference(task.get_dataset()[idx])
            test_program = processed + "\n" + reference
            result = check_correctness(
                test_program, timeout=3.0, task_id=idx, completion_id=0
            )
            return int(result["passed"]), 0

    else:
        raise NotImplementedError(f"Task {task} not supported")
    return reward_fn


def compute_rewards(task, idx, generations):
    reward_fn = get_reward_fn(task, idx)
    rewards = []
    results = []
    for g in generations:
        rew, res = reward_fn(g)
        rewards += [rew]
        results += [res]
    return rewards, results


def run_policy(policy, task, debug=False, shuffle=False):
    """
    Policy: prompt -> list of completions and (optional) features
    Run policy on dataset and return list of dicts of {idx, generation, reward, features}
    """
    print("Running policy...")
    data = task.get_dataset()
    idxs = np.arange(len(data))
    if shuffle:
        idxs = np.random.permutation(idxs)
    idxs = [int(i) for i in idxs]

    samples = []
    for idx in tqdm(idxs):
        doc = data[idx]
        prompt = task.get_prompt(doc)
        generations, features = policy(prompt)
        rewards, results = compute_rewards(task, idx, generations)

        for g, r, rr, f in zip(generations, rewards, results, features):
            result = dict(
                idx=idx,
                generation=g,
                reward=r,
                result=rr,
                features=f,
            )
            samples += [result]
            ppprint(result["generation"])

        if debug:
            if idx == 4:
                break
    return samples


def save_samples(samples, dir_name, overwrite=True):
    sample_path = os.path.join(dir_name, "samples.pkl")
    if overwrite:
        updated_samples = samples
    else:
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                existing_samples = pickle.load(f)
            updated_samples = existing_samples + samples
        else:
            updated_samples = samples

    with open(sample_path, "wb") as f:
        pickle.dump(updated_samples, f)


def load_sample(file_name):
    with open(file_name, "rb") as f:
        x = pickle.load(f)
    return x


def load_all_samples(dir_name, run="*", processes=20):
    if type(dir_name) == str:
        dir_name = [dir_name]
    files = []
    for dir in dir_name:
        files += list(glob(f"{dir}/{run}/*.pkl"))
    samples = []
    for file in files:
        samples += load_sample(file)
    return samples


def compute_pass_at_k(dir_name, ks=[1, 10, 100]):
    print("Loading data...")
    samples = load_all_samples(dir_name)

    print("Reshaping data...")
    num_tasks = len(np.unique([s["idx"] for s in samples]))
    num_samples = np.zeros(num_tasks)
    num_correct = np.zeros(num_tasks)
    for s in tqdm(samples):
        num_samples[s["idx"]] += 1
        num_correct[s["idx"]] += s["reward"]

    print("Computing pass@k...")
    pass_at_k = {}
    for k in ks:
        if k > num_samples.max():
            print(f"WARNING: k={k} is larger than max num samples {num_samples.max()}")
            continue
        p = estimate_pass_at_k(num_samples, num_correct, k)
        pass_at_k[k] = p
        print(f"pass@{k}: {p.mean()}")
    return pass_at_k
