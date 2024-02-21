This repository provides the code for reproducing Q-probe results on feedback expeirment in [Q-Probe: A Lightweight Approach to Reward Maximization for Language Models](https://www.bing.com/)
This repo is built from [HALOs](https://github.com/ContextualAI/HALOs). Appreciate the authors' effort! Code tested to work on 4 A100 GPU's.

### How to install

This part require a separete environment, install by:
```
conda env create -f environment.yaml
conda activate halos
python -m ipykernel install --user --name halos --display-name "halos"
mkdir features
mkdir outputs
mkdir samples
```

### Collecting K generations for each prompt in test set

The `eval.py` will save sampling results into `samples`. While we are changing seed, the seed used for subsetting test set is kept the same, therefore only the completions are different, which is desired.  
The `compare.py` will call GPT-4 to judge each completion and save results in the json files saved as well as in `samples/results.json`. Remember to set your OpenAI API key in your system.  
Below is an example of generating K=48 samples for 512 test prompts from `llama7b_sft`.
```
export M=llama7b_sft
for i in {1..48}
do
echo ${M}_seed${i}
python eval.py --config-path=config --config-name=config loss=sft exp_name=${M}_seed${i} model=$M datasets=[shp,oasst,hh] mode=sample n_samples=512 model.eval_batch_size=16 samples_dir=samples/ seed=$i
python compare.py -f=samples/${M}_seed${i}.json
python get_activations.py --json samples/${M}_seed${i}.json --location res --layer 32
done
```

### Collect activations for generated samples as well as training samples

The training set can be found in `samples/train_extraction.json` (download from [here](https://drive.google.com/file/d/1zw9mZrphoCHDYjKHd13tujP578VCMmvQ/view?usp=sharing)), but we still need to generate its features.  
Below is an exmaple of extracting feature for training set at the last layer of the residual stream, for each of the 16 parts. 
```
for i in {0..15}
do
python get_activations.py --json samples/train_extraction.json --location res --layer 32 --total_parts 16 --part $i
done
```
Swap the `--json` parameter into other json's in `samples` to extract features for them. 

### All compute-heavy work done, now we run Q-prob experiment in a jupyter notebook

Check out `check_probe.ipynb`. If you want to skip the last two steps, you can download all the data needed from [here](https://drive.google.com/drive/folders/1hTMPrqyFDpQpL2VhALB2oCOzYvYtVskk?usp=sharing)
