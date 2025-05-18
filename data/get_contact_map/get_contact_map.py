from Bio.PDB.PDBParser import PDBParser
import os
import numpy as np

# The source file comes from the pdbs suffix file of the protein, such as uniprot or alphafold database

path = '/home/ywh/pdbs'
parser = PDBParser()
dict = {}
for file in os.listdir(path):


    structure_id = file.split('.')[0]
    filename = path+'/'+file
    structure = parser.get_structure(structure_id, filename)
    ls = []
    for residue in structure.get_residues():
        if len(ls)>=1024:
            break
        a =residue['CA']

        ls.append(a)
    cont = []
    for i,a in enumerate(ls):
        for j,b in enumerate(ls):
            if(a-b)<8:
                if not i == j:
                    cont.append([i,j])
    cont = np.array(cont)
    dict[structure_id] = cont

np.savez('/home/ywh/contact_map.npz',**dict)

