import os
import argparse
import torch
import torchvision.transforms as transforms
import pandas as pd
import time
import ray

from model.Model import MSTS
from src.datasets import SmilesDataset
from src.config import input_data_dir, base_file_name, sample_submission_dir, reversed_token_map_dir
from utils import logger, make_directory, load_reversed_token_map, smiles_name_print, str2bool


def main():
    start_time = time.time()

    smiles_name_print()
    parser = argparse.ArgumentParser()

    parser.add_argument('--work_type', type=str, default='train', help="choose work type 'train' or 'test'")
    parser.add_argument('--encoder_type', type=str, default='wide_res', help="choose encoder model type 'wide_res', 'res', and 'resnext' ")
    parser.add_argument('--seed', type=int, default=1, help="choose seed number")

    parser.add_argument('--decode_length', type=int, default=140, help='length of decoded SMILES sequence')
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='droup out rate')
    parser.add_argument('--device', type=str, default='cuda', help='sets device for model and PyTorch tensors')
    parser.add_argument('--gpu_non_block', type=str2bool, default=True, help='GPU non blocking flag')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True, help='set to true only if inputs to model are fixed size; otherwise lot of computational overhead')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=384, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning')
    parser.add_argument('--decoder_lr', type=float, default=4e-4, help='learning rate for decoer')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of')
    parser.add_argument('--fine_tune_encoder', type=str2bool, default=True, help='fine-tune encoder')

    parser.add_argument('--model_save_path', type=str, default='graph_save', help='model save path')
    parser.add_argument('--model_load_path', type=str, default=None, help='model load path')
    parser.add_argument('--model_load_num', type=int, default=None, help='epoch number of saved model')
    parser.add_argument('--test_file_path', type=str, default=None, help='test file path')

    config = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = 'cpu' if device == 'cpu' else config.device
    print('torch work_type:', config.device)
    model = MSTS(config)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if config.work_type == 'train':
        make_directory(config.model_save_path)
        if not (config.model_load_path == None) and not (config.model_load_num == None):
            model.model_load()
            print('model loaded')
        else:
            print('start from initial')

        train_loader = torch.utils.data.DataLoader(
            SmilesDataset(input_data_dir, base_file_name, 'TRAIN',
                          transform=transforms.Compose([normalize])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            SmilesDataset(input_data_dir, base_file_name, 'VAL',
                          transform=transforms.Compose([normalize])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        log_index = ['t_loss', 't_accr', 'v_loss', 'v_accr']
        logger(log_index)
        for itr in range(config.epochs):
            print('epoch:', itr)
            t_l, t_a = model.train(train_loader)
            v_l, v_a = model.validation(val_loader)
            model.model_save(save_num=itr)
            logger([t_l, t_a, v_l, v_a])


    elif config.work_type == 'single_test':
        if not config.test_file_path == None:

            submission = pd.read_csv('utils/sample_submission.csv')
            reversed_token_map = load_reversed_token_map(reversed_token_map_dir)
            data_list = os.listdir(config.test_file_path)

            transform = transforms.Compose([normalize])
            model.model_load()
            print('model loaded')
            submission = model.model_test(submission, data_list, reversed_token_map, transform)
            submission.to_csv('sample_submission.csv', index=False)

        else:
            print('the test file path is none')

    elif config.work_type == 'ensemble_test':
        ray.init()
        if not config.test_file_path == None:

            submission = pd.read_csv('utils/sample_submission.csv')
            reversed_token_map = load_reversed_token_map(reversed_token_map_dir)
            data_list = os.listdir(config.test_file_path)

            transform = transforms.Compose([normalize])
            submission = model.ensemble_test(submission, data_list, reversed_token_map, transform)
            submission.to_csv('sample_submission2.csv', index=False)

        else:
            print('the test file path is none')

    else:
        print('incorrect work type received.')

    print('process time:', time.time() - start_time)

if __name__ == '__main__':
    main()