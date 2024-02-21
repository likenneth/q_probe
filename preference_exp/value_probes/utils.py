# from https://github.com/abacaj/code-eval
import copy
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np


def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    exps = np.exp(
        scaled_logits - np.max(scaled_logits)
    )  # Subtract max for numerical stability
    softmax_output = exps / np.sum(exps)
    return softmax_output


# probing


def train_probes(features, gt_labels, tr_pctg, seed=42, max_iter=1000):
    np.random.seed(seed)
    gt_labels = np.array(gt_labels)
    total_sample = features.shape[0]
    train_idx = np.random.choice(
        range(len(features)), size=int(total_sample * tr_pctg), replace=False
    )

    if tr_pctg == 1.0:
        val_idx = copy.deepcopy(train_idx)
    else:
        val_idx = np.array(list(set(range(len(features))) - set(train_idx)))
    X_train = features[train_idx]
    y_train = gt_labels[train_idx]
    X_val = features[val_idx]
    y_val = gt_labels[val_idx]
    # pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=max_iter, solver="liblinear", class_weight="balanced"))
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=seed, max_iter=max_iter, solver="liblinear"),
    )
    clf = pipe.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    train_acc, train_f1, train_precision, train_recall = (
        accuracy_score(y_train, y_train_pred),
        f1_score(y_train, y_train_pred),
        precision_score(y_train, y_train_pred),
        recall_score(y_train, y_train_pred),
    )
    val_acc, val_f1, val_precision, val_recall = (
        accuracy_score(y_val, y_val_pred),
        f1_score(y_val, y_val_pred),
        precision_score(y_val, y_val_pred),
        recall_score(y_val, y_val_pred),
    )

    return clf, (val_acc, val_f1, val_precision, val_recall)


"""
|            | Full </s> | Partial </s> | Partial |
|------------|:---------:|:------------:|:-------:|
| FP Correct |     1     |       0      |    1    |
| FP Wrong   |     0     |       0      |    0    |
"""


def load_full_program_for_probing(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    programs = []
    gt_labels = []
    for problem_id, sol_l in data.items():
        for sol in sol_l:
            programs.append(sol["completion"] + "\n</s>")
            gt_labels.append(1 if sol["passed"] else 0)
    return programs, gt_labels


def aug_partial_programs_for_probing(programs, gt_labels):
    ps_gts = []
    for p, l in zip(programs, gt_labels):
        lines = [l for l in p.split("\n")]
        for i in range(
            len(lines) - 1
        ):  # from 1-line program to all but one line program
            ps_gts.append(("\n".join(lines[: i + 1]) + "\n", l))

    # all partial programs with EOS, labelled as v=0
    for p, l in zip(programs, gt_labels):
        lines = [l for l in p.split("\n")]
        for i in range(len(lines) - 1):
            ps_gts.append(("\n".join(lines[: i + 1]) + "\n</s>", 0))

    programs.extend([_[0] for _ in ps_gts])
    gt_labels.extend([_[1] for _ in ps_gts])
    print(f"Created {len(programs)} partial programs...")
    return programs, gt_labels
