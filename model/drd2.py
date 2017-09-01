import pickle
#import csv
import numpy as np
from sklearn import svm
from rdkit import Chem
from rdkit import DataStructs

from rdkit.Chem import AllChem




def read_smi(filename):
    with open(filename) as file:
        smiles = file.readlines()
    smiles = [i.strip() for i in smiles]
    return smiles


#def read_smiles_csv(filename):
    # Assumes smiles is in column 0
 #   with open(filename) as file:
  #      reader = csv.reader(file)
   #     smiles_idx = next(reader).index("smiles")
    #    data = [row[smiles_idx] for row in reader]
    #return data


#smiles=read_smi('drugs_sub.smi')
#smiles=read_smiles('DRD2_test.smi')

#smiles=smiles[:10]


#smiles_test= ['COc1ccccc1N1CCN(CCCOC(=O)C23CC4CC(CC(C4)C2)C3)CC1','COc1ccccc1N1CCN(CCOC(=O)C23CC4CC(CC(C4)C2)C3)CC1','COc1ccccc1N1CCN(CCCCN2CCn3c(cc4ccccc43)C2=O)CC1','COc1ccccc1N1CCN(CCC(=O)NC23CC4CC(CC(C4)C2)C3)CC1','COc1ccccc1N1CCN(CCCNC(=O)NC23CC4CC(CC(C4)C2)C3)CC1']
#smiles=smiles_test

def fingerprints_from_mols(mols):
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 3) for mol in mols] 
            np_fps = []
            for fp in fps:
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                np_fps.append(arr)
            return np_fps


with open("clf_drd2.pkl", "r") as f:
            clf = pickle.load(f)

def activity_scores(smiles):            
	mols = [Chem.MolFromSmiles(smile) for smile in smiles]
	valid = [1 if mol!=None else 0 for mol in mols]
	valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean==1]
	valid_mols = [mols[idx] for idx in valid_idxs]
	if len(valid_mols)>0:
		fps = fingerprints_from_mols(valid_mols)
		activity_score = clf.predict_proba(fps)[:, 1]    
		return activity_score, valid_idxs
	else:
		return [], []
