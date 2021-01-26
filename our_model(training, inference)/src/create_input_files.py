import argparse

from .utils import train_validation_split_df, create_input_files, create_test_files, str2bool
from .config import data_dir, train_dir, test_dir, train_csv_dir, train_pickle_dir, sample_submission_dir, input_data_dir, random_seed

def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--split', default=False, type=str2bool,
                        help='Should we make a split to dataframe?')
    # test files
    parser.add_argument('--test_file', default=False, type=str2bool,
                        help='Should we make a test set?')
    # train files
    parser.add_argument('--train_file', default=False, type=str2bool,
                        help='Should we make a train validation set?')
    config = parser.parse_args()

    return config
if __name__ == '__main__':
    # Get configuration
    config = vars(parse_args())
    if config['split']==True:
        print('Carrying out split')
        train_validation_split_df(data_dir=data_dir,
                            train_csv_dir=train_csv_dir,
                            random_seed = random_seed,
                            train_size=0.8)

    if config['test_file']==True:
        print('Creating test files')
        create_test_files(submission_csv_dir=sample_submission_dir,
                            test_dir = test_dir,
                            output_folder=input_data_dir)

    if config['train_file']==True:
        print('Creating train, validation file')
        # Create input files (along with word map)
        create_input_files(train_dir=train_dir,
                        train_pickle_dir=train_pickle_dir,
                        output_folder=input_data_dir,
                        min_token_freq=5,
                        max_len=75,
                        random_seed=random_seed)


