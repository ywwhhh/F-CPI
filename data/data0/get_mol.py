
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem



data_train = [token.strip().split('_') for token in open('/home/ywh/flourine_smile_68test/data/data0/train_set')]
out_train = open('/home/ywh/flourine_smile_68test/data/data0/mol_train','w')

for line in data_train:
    f = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[0]), 4, 512)).tolist()
    nf = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[2]), 4, 512)).tolist()
    out_train.write(' '.join(list(map(str,f)))+'_'+' '.join(list(map(str,nf)))+'\n')




data_valid = [token.strip().split('_') for token in open('/home/ywh/flourine_smile_68test/data/data0/valid_set')]
out_valid = open('/home/ywh/flourine_smile_68test/data/data0/mol_valid','w')

for line in data_valid:
    f = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[0]), 4, 512)).tolist()
    nf = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[2]), 4, 512)).tolist()
    out_valid.write(' '.join(list(map(str,f)))+'_'+' '.join(list(map(str,nf)))+'\n')

data_test = [token.strip().split('_') for token in open('/home/ywh/flourine_smile_68test/data/data0/test_set')]
out_valid = open('/home/ywh/flourine_smile_68test/data/data0/mol_test','w')

for line in data_test:
    f = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[0]), 4, 512)).tolist()
    nf = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[2]), 4, 512)).tolist()
    out_valid.write(' '.join(list(map(str,f)))+'_'+' '.join(list(map(str,nf)))+'\n')