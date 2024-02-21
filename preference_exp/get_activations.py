import os
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict

from utils import disable_dropout, init_distributed, get_open_port, rank0_print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ContextualAI/archangel_sft_llama7b")
    parser.add_argument("--json", type=str, default="samples/train_extraction.json")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--location", type=str, choices=["mlp", "mha", "res"], default="res")
    parser.add_argument("--part", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID", -1), help="part")
    parser.add_argument("--layer", type=int, default=-1, help="or all layer")
    parser.add_argument("--total_parts", type=int, default=-1, help="total parts")
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    # load texts
    samples = json.load(open(args.json))["samples"]  # a list of dict with keys ['prompt', 'chosen', 'policy', 'winner']
    programs, gt_labels = [], []
    for sample in samples:
        policy = sample["prompt"] + sample["policy"]
        programs.append(policy.rstrip())
        gt_labels.append(1 if sample["winner"] == "policy" else 0)
        chosen = sample["prompt"] + sample["chosen"]
        programs.append(chosen.rstrip())
        gt_labels.append(1 if sample["winner"] == "chosen" else 0)
    
    if 0:
        programs = programs[:10]
        gt_labels = gt_labels[:10]

    HEADS = [
        f"model.layers.{i}.self_attn.o_proj"
        for i in range(model.config.num_hidden_layers)
    ]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    RES = [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]

    temp_act_dump = []

    def edit_func(output, name):
        if type(output) == tuple:
            output0 = output[0]
        else:
            output0 = output
        # print(output0.shape)
        # output: except for the first token, [BS=1, T=1, H=32*(D=128)=4096]
        single_layer_output = output0[0, -1].detach().cpu()  # [#H*D]
        temp_act_dump.append(single_layer_output)
        return output

    if args.part != -1:
        start_idx = len(programs) // args.total_parts * args.part
        if args.part == args.total_parts - 1:
            end_idx = len(programs)
        else:
            end_idx = len(programs) // args.total_parts * (args.part + 1)
    else:
        assert args.total_parts == -1
        start_idx = 0
        end_idx = len(programs)

    if args.layer == -1:
        args.layer = model.config.num_hidden_layers
    layers = [args.layer]

    if args.location == "mha":
        with TraceDict(
            model, HEADS, retain_output=False, retain_input=True, edit_output=edit_func
        ) as ret:  # this hook takes out the activations before applying o_proj in MHA's
            for program in tqdm(programs[start_idx:end_idx]):
                tokens = tokenizer(program, return_tensors="pt")["input_ids"].cuda()
                _ = model(tokens)
    elif args.location == "mlp":
        with TraceDict(
            model, MLPS, retain_output=True, retain_input=False, edit_output=edit_func
        ) as ret:  # this hook takes out the residual MLP's inject into the highway
            for program in tqdm(programs[start_idx:end_idx]):
                tokens = tokenizer(program, return_tensors="pt")["input_ids"].cuda()
                _ = model(tokens)
    elif args.location == "res":
        for program in tqdm(programs[start_idx:end_idx]):
            tokens = tokenizer(program, return_tensors="pt")["input_ids"]
            hidden_states = model(tokens, output_hidden_states=True).hidden_states
            temp_act_dump.extend(
                [hidden_states[i][0, -1].detach().cpu().numpy() for i in layers]
            )  # list of [#H*D], of length #L
    else:
        assert 0, "location not supported"

    assert len(temp_act_dump) == len(layers) * len(programs[start_idx:end_idx])
    activations = np.array(temp_act_dump)  # [#L*#P, #H*D]
    activations = activations.reshape(
        len(layers),
        len(programs[start_idx:end_idx]),
        model.config.num_attention_heads,
        -1,
    )  # [#L, #P, #H, D]
    activations = np.transpose(activations, (1, 0, 2, 3))  # [#P, #L, #H, D]
    print(activations.shape)

    exp_name = json.load(open(args.json))["config"]["exp_name"]
    save_path = f'features/{exp_name}{f"_part{args.part}of{args.total_parts}" if args.total_parts != -1 else ""}_{args.location}_layer{args.layer}.npy'
    os.makedirs(f'features', exist_ok=True)
    np.save(save_path, activations)  # [#P, #L, #H, D]

    # gt_labels = np.array(gt_labels[start_idx:end_idx])
    # save_path = f'features/{exp_name}{f"_part{args.part}of{args.total_parts}" if args.total_parts != -1 else ""}_{args.location}_layer{args.layer}_gt_labels.npy'
    # np.save(save_path, gt_labels)  # [#P]

if __name__ == "__main__":
    main()