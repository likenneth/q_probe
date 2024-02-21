# Q-probe

A repo for building and evaluating q-probes on code tasks.

Install environment with 
```
conda create -n q-probe python=3.10
conda activate q-probe
pip install -r requirements.txt
python -m ipykernel install --user --name q-probe --display-name "q-probe"
```


## Usage:

### Data generation:

```
sbatch scripts/run.sh base_train
```

### Q probe training:

```
sbatch scripts/probe_sweep.sh
```

### Figure generation:

You can skip previous steps and download saved data from [here](https://drive.google.com/drive/folders/1T2axGPw9-mJp5HjWS9kh13iEXXqy730n?usp=sharing).

For API experiment, we first call API to get generations and then collect features from LLaMA-70B locally, the script can be found [here](https://drive.google.com/file/d/1Bj8jhUtQAunDnx0O36pOkzKto2MYGJXv/view?usp=sharing) (with some pathes to be adjusted). 


```
scripts/notebooks/plotting.ipynb
```