import os
from omegaconf import OmegaConf
from q_probe import llm
from q_probe.data import run_policy, save_samples
from transformers import set_seed

from bigcode_eval import tasks


def get_policy(config, task):
    print("Loading policy...")
    llm_config = llm.LLMConfig(**OmegaConf.to_container(config.llm), debug=False)
    if "/" in llm_config.model_name:
        llm_obj = llm.LLM(llm_config, task=task)
    else:
        llm_obj = llm.LLM_API(llm_config, task=task)

    def policy(prompt):
        outputs = llm_obj.get_actions(prompt)
        actions = outputs[0]
        hiddens = outputs[-1]
        if config.dataset == "HumanEval" and "gpt" in config.llm.model_name and 0:
            actions = actions
        else:
            actions = ["\n".join([prompt, a]) for a in actions]
        return actions, hiddens

    return policy


def load_task(task_name):
    if task_name == "MBPP":
        task = tasks.mbpp.MBPP()
    elif task_name == "MBPP_train":
        task = tasks.mbpp_train_val.MBPP()
    elif task_name == "HumanEval":
        task = tasks.humaneval.create_task(strip_prompt=True)()
    else:
        assert 0, "dataset must be either MBPP, HumanEval, ..."
    print(f"Evaluating policy on {task_name}...")
    return task


def main(config):
    if config.job_id == "local":
        set_seed(0)
    else:
        set_seed((int(config.job_id) + int(config.task_id) * 17) % (2**32))
    path = os.path.join(
        config.save_dir, "samples", str(config.job_id), str(config.task_id)
    )
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, config.config.replace("/", "_")), "w") as file:
        OmegaConf.save(config, file)

    task = load_task(config.dataset)
    policy = get_policy(config, task)
    for _ in range(config.outer_loop_batches):
        samples = run_policy(
            policy=policy, task=task, debug=config.debug, shuffle=False
        )
        save_samples(samples, path, overwrite=False)


if __name__ == "__main__":
    # load config
    cli_args = OmegaConf.from_cli()
    config = OmegaConf.load(cli_args.get("config", "generation_configs/base.yaml"))
    config = OmegaConf.merge(config, cli_args)
    config.job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    config.task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    print(config)
    main(config)
