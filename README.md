# LG_SMILES_competition
This is the source code for LG-HACKATHON hosted by [DACON](https://dacon.io/competitions/official/235640/leaderboard/). The main objective of the competition was to predict the SMILES sequence given a chemical structure image. 
Our prediction achieved 0.99567 in private leaderboard(Tanimoto-Similarity based)
![Image](/figures/img_to_smiles.png)
## Model Weight
You can download the model weights from this [dropbox](https://www.dropbox.com/sh/88zi2kv7vykgsij/AADAZPhlkA6haDNOvXjvsDaFa?dl=0) link


## Requirements
Dependencies (with python =3.6)
### conda environment
```
conda env create -f utils/LG_CONDA_ENV.yml
```

### pip dependency
```
pip install utils/pip_requirements.txt
```

## Data generation

### generate molecule image datas
```
python training_data_generation/dataframe_generation_by_group.py 
python training_data_generation/train_image_generation.py 
```
### generate data information index
```
python training_data_generation/sequence_dataframe_generation.py
```
### data sampling
```
python training_data_generation/data_sampling.py
```


## How to train
### data preprocessing
You will need to modify the src/config.py to accustom your directory setting.
```python
# The data_dir should point the data directory folder. All training and testing files should be placed below
data_dir = Path('/home/jaeho_ubuntu/SMILES/data/')
train_dir = data_dir / 'train' # all train_<number>.png files should be under the train folder
test_dir = data_dir / 'test'  # all test_<number>.png files should be under the test folder
train_csv_dir = data_dir /'train.csv'  #train.csv file provided by LG 
# Sample submission directory
sample_submission_dir = data_dir /'sample_submission.csv'

# train_modified contains a modifed version of train.csv. 
# Information of the train/validation split is stored here. I saved in pickle just for efficiency
train_pickle_dir = data_dir /'train_modified.pkl'

### Data directory containing .hdf5, .json file files.
input_data_dir = data_dir / 'input_data'
base_file_name = 'seed_123_max75smiles'

### seed for train/val split
random_seed = 123

### Reversed_token_file used to map numbers to string. 
reversed_token_map_dir = input_data_dir/ f'REVERSED_TOKENMAP_{base_file_name}.json'
```

Running below scripts will return a .hdf5, .json files needed for training and testing.
```
# Only if you need to make training and validation set 
python --split True --train_file True
# If you need to make a test set provided by LG
python --test_file True
```

### Training

```
python main.py --work_type train
```

if you want to train again form the checkpoint(saved model weight)
```
python main.py --work_type train \
               --model_load_path <path where the model is saved> \
               --mode_load_num <model number>
```

## How to test
When you run prediction with our model, you should choose whether to predict with a single model or an ensamble model. Turn either `--work_type single_test` or `--work_type ensemble_test` flags True.
### simgle model test
single model test requires three flags: `--model load path`, `--model load number`, and `--test file path`
```
python main.py --work_type single_test \
               --model_load_path <path where the model is saved> \
               --model_load_num <model number> \
               --test_file_path <path where the test images are saved>
```

### ensemble test
ensemble model contains weight file of 5 single models. The test requires two flags: `--model load path` and `--test file path`
Running this script also requires `./model/prediction_models.yaml` which contains hyperparameters of the model and the type of model used.
The models in the `--model load path` should match with this `prediction_models.yaml` file configurations. 
```
python main.py --work_type ensemble_test \
               --model_load_path <path where the model is saved> \
               --test_file_path <path where the test images are saved>
```


## Optional Arguments

| optional arguments | types | default | help |
|---|:---:|:---:|:---|
|`--work_type` | str |  `train'` | choose work type 'train' or 'test' |
|`--encoder_type` | str |  `'wide_res'` | choose encoder model type 'wide_res', 'res', and 'resnext'  |
|`--seed` | int |  `1` | set the seed of model |
|`--decode_length` | int |  `140` | length of decoded SMILES sequence |
|`--emb_dim` | int |  `512` | dimension of word embeddings |
|`--attention_dim` | int |  `512` | dimension of attention linear layers |
|`--decoder_dim` | int |  `512` | dimension of decoder RNN |
|`--dropout` | float |  `0.5` | droup out rate |
|`--device` | str |  `'cuda'` | sets device for model and PyTorch tensors |
|`--gpu_non_block` | bool |  `True` | GPU non blocking flag |
|`--cudnn_benchmark` | bool |  `True` | set to true only if inputs to model are fixed size; otherwise lot of computational overhead |
|`--epochs` | int |  `50` | number of epochs to train for |
|`--batch_size` | int |  `384` | batch size |
|`--workers` | int |  `8` | for data-loading; right now, only 1 works with h5py |
|`--encoder_lr` | float |  `1e-4` | learning rate for encoder if fine-tuning |
|`--decoder_lr` | float |  `4e-4` | learning rate for decoer |
|`--grad_clip` | float |  `5.` | clip gradients at an absolute value of |
|`--fine_tune_encoder` | bool |  `True` | fine-tune encoder |
|`--model_save_path` | str |  `'graph_save'` | model save path |
|`--model_load_path` | str |  `None` | model load path |
|`--model_load_num` | int |  `None` | epoch number of saved model |
|`--test_file_path` | str |  `None` | test file path |


## Top 5-Model Results in Public Score

| Pytorch Model Name | Dimension | Pubilic Score | DataSet | 
|---|:---:|:---:|:---|
|`Wide ResNet101-2` | 512 |  0.9729 | OurDataSet2 |
|`Wide ResNet101-2` | 512 |  0.9625 | OurDataSet1 |
|`ResNet152` | 512 |  0.9622 | LG DataSet |
|`ResNet152` | 256 |  0.9512 | LG DataSet  |
|`ResNeXt-101-32x8d` | 256 |  0.9677 | OurDataSet1 |