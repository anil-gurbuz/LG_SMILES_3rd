import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class SmilesDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, base_file_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param base_file_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        if self.split in {'TRAIN','VAL'}:
            self.h = h5py.File(data_folder/ f"{self.split}_IMAGES_{base_file_name}.hdf5", 'r')
            # Load encoded sequences (completely into memory)
            with open(data_folder/ f"{self.split}_SMILES_SEQUENCES_{base_file_name}.json", 'r') as j:
                self.sequences = json.load(j)

            # Load sequence lengths (completely into memory)
            with open(data_folder/ f"{self.split}_SMILES_SEQUENCE_LENS_{base_file_name}.json", 'r') as j:
                self.sequence_lens = json.load(j)



        else: #self.split in {'TEST'}
            self.h = h5py.File(data_folder/ "TEST_LG_IMAGES.hdf5", 'r')

        self.imgs = self.h['images']


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.dataset_size = self.imgs.shape[0]


    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        if self.split in {'TRAIN', 'VAL'}:
            sequence = torch.LongTensor(self.sequences[i])
            sequence_len = torch.LongTensor([self.sequence_lens[i]])
            return img, sequence, sequence_len
        else:
            return img

    def __len__(self):
        return self.dataset_size
