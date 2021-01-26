
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm, trange

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
    
# Making filename & Resetting the index accordingly the groups 
def index_reset_by_group (data, group) :
    group_df = data[data['group'] == group]
    group_df = group_df.reset_index()
    
    # making filename as "the length of smiels"_train_"index"
    # ex) 0020_train_4
    
    sm_length= []
    for i in range(len(group_df)) :
        sm_length.append(len(group_df['SMILES'][i]))
    
    g_length= []
    for i in range(len(group_df)) : 
        g_length.append(str(sm_length[i]).zfill(4))
        
    g_file_name = [] 
    for i in range(len(group_df)) :
        g_file_name.append('{0}_train_{1}'.format(g_length[i], i))
    
    group_df['length'] = sm_length
    group_df['file_name'] = g_file_name
    group_df= group_df[['file_name','SMILES','length', 'group']]
    
    return group_df

# Generating dataframe 
for j in tqdm(range(1, 32)) : 
    globals()['g{}'.format(j)] = index_reset_by_group(filtered_df, j)

# Concatenating the all datafrane 
ending_df = pd.concat([g1,g2,g3,g4,g5,g6,g7,g8,
                       g9,g10,g11,g12,g13,g14,g15,g16,
                       g17,g18,g19,g20,g21,g22,g23,g24,
                       g25,g26,g27,g28,g29,g30,g31])
# Saving as pickle file
new_path = '/sequence_dataframe/'
ending_df.to_pickle(new_path +'training_dataset_real.pkl')




