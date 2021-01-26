import os
import random
import click

DATA_PATH = '/train_dataset/' # This is a path for original data, and it matches with data_generated directory.
SAMPLED_PATH = '/sampled_data_path/' # This is a path for sampled data

@click.command()
@click.option('--random_seed', default=1, help='random seed for sampling training data')
@click.option('--max_seq', default=100, help='max_seq for splitting data')
@click.option('--num_sample',default = 35000, help='number of samples for each sequence')

def main(random_seed,max_seq,num_sample):
    """
    This function is for sampling data from original dataset.
    :param random_seed: random seed for sampling training data
    :param max_seq: max_seq for splitting data
    :param num_sample: number of samples for each sequence.
    """
    for idx in range(0,max_seq):
        val = str(idx+1).zfill(4)
        # If the number of data in a sequence data folder is smaller than expected number of samples,
        # just move whole data without sampling.
        if len(os.listdir(DATA_PATH + val)) <= num_sample:
            print('start to sample '+val+'!')
            os.mkdir(SAMPLED_PATH)
            os.system('mv '+DATA_PATH+val+' '+SAMPLED_PATH)

        # If the number of data in a sequence data folder is bigger than expected number of samples,
        # sample the data with a specific random seed.
        elif len(os.listdir(DATA_PATH + val)) > num_sample:
            print('start to sample ' + val + '!')
            inner_files = os.listdir(DATA_PATH+val)
            inner_files.sort()
            random.seed(random_seed)
            sampled_list = random.sample(inner_files,num_sample)
            for i in sampled_list:
                try:
                    os.system('mv ' + DATA_PATH + val+'/'+i+' '+SAMPLED_PATH+val)
                except:
                    os.mkdir( SAMPLED_PATH)
                    os.system('mv ' + DATA_PATH + val+'/'+i+' '+SAMPLED_PATH+val)

if __name__ == '__main__':
    main()
