import os
import csv
import json

def _csv_writer(file_name, write_data):
    f = open(file_name, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(write_data)
    f.close()


def logger(log_data):
    _csv_writer('log.csv', log_data)


def make_directory(path):
    try:
        os.mkdir(path)
        print(path + ' is generated!')
    except OSError:
        pass

def load_reversed_token_map(path):
    """Gets the path of the reversed token map json"""
    with open(path, 'r') as j:
        reversed_token_map = json.load(j)
    return reversed_token_map

def decode_predicted_sequences(predicted_sequence_list,reversed_token_map):
    """
    :param predicted_sequence_list: List of sequences in predicted form ex) [27,1,2,5]
    :param reveresed_token_map: Dictionary mapping of reversed token map
    :return: predicted_sequence_str:
    """
    predicted_sequence_str = ""
    for e in predicted_sequence_list:
        e = str(e)
        if reversed_token_map[e]=='<unk>':
            continue
        elif reversed_token_map[e] in {'<end>','<pad>'}:
            break
        else:
            predicted_sequence_str+=reversed_token_map[e]
    
    return predicted_sequence_str


async def async_decode_predicted_sequences(predicted_sequence_list, reversed_token_map):
    """
    :param predicted_sequence_list: List of sequences in predicted form ex) [27,1,2,5]
    :param reveresed_token_map: Dictionary mapping of reversed token map
    :return: predicted_sequence_str:
    """
    predicted_sequence_str = ""
    for e in predicted_sequence_list:
        e = str(e)
        if reversed_token_map[e] == '<unk>':
            continue
        elif reversed_token_map[e] in {'<end>', '<pad>'}:
            break
        else:
            predicted_sequence_str += reversed_token_map[e]

    return predicted_sequence_str

def smiles_name_print():
    print('  ______   __    __   __   __       ______   ______    ')
    print(' /\  ___\ /\ "-./  \ /\ \ /\ \     /\  ___\ /\  ___\   ')
    print(' \ \___  \\\\ \ \-./\ \\\\ \ \\\\ \ \____\ \  __\ \ \___  \  ')
    print('  \/\_____\\\\ \_\ \ \_\\\\ \_\\\\ \_____\\\\ \_____\\\\/\_____\ ')
    print('   \/_____/ \/_/  \/_/ \/_/ \/_____/ \/_____/ \/_____/ ')


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
