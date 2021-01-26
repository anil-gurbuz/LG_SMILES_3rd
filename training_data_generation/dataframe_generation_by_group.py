'''
아래의 코드는 병렬처리를 위해 약 1억 1천만개의 전체 데이터프레임을 코어수에 따라 그룹으로 나누고 
각각의 그룹의 데이터프레임을 생성하는 코드입니다. 생성된 그룹별 데이터 프레임은 다음 파일인 
train_image_generation.py의 input으로 활용됩니다. 
''' 

"""
The code below is the code that divides about 110 million 
total data frames into groups according to the number of cores 
for parallel processing and generates data frames for each group.
The generated group-by-group data frames are used as inputs for 
the following files: train_image_generation.py.
"""


import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm

# CID_SMILES is downloaded from pubchem
# Uploading the data 
f = open("CID-SMILES")

# The number of total data
len(list(f))

# Removing unneccesary charaters & Extracting SMILES sequences
a = []
usage = []
length = []
f = open("CID-SMILES") 
for i in tqdm(range(111307682)) : 
    line = f.readline()
    a.append(line)
    a[i] = a[i][:-1].split("\t")[1]
    usage.append(a[i])
    length.append(len(a[i]))
    
# Making dataframe
df = pd.DataFrame(usage)
df.columns = ["SMILES"]
df['length'] = length
df['group'] = 0

# Oragainizing the group by the number of core
# The number of data sameple for one group calculated as 
# the number of total data sample / the number of core
# ex) 111307682 / 31 = 3700000
# The number of core can be different by each environment 

for i in range(1, 32) : 
    filtered_df = df
    filtered_df['group'][(i-1)* 3700000 : i * 3700000] = i

new_path = '/train_dataset/'
for i in range(1, 32) :
    g_filtered = filtered_df[filtered_df['group'] == i]
    g_filtered.to_csv(new_path + "filtered_df_group{}.csv".format(i)) 

