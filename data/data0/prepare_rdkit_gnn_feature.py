from rdkit import Chem
import rdkit
import networkx as nx
#############进一步处理,n编号9

# mol_f = Chem.SDMolSupplier('data/pos/ff')
# mol_nf = Chem.SDMolSupplier('data/pos/fnf')
er = 0
#
# count = []

vocab = [token.strip() for token in open('/home/ywh/flourine_smile_68test/data/data0/valid_set')]

output = open('/home/ywh/flourine_smile_68test/data/data0/valid_graph', 'w')
trans_num = {3: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 11: 7, 14: 8, 15: 9, 16: 10, 17: 11, 19: 12, 34: 13, 35: 14, 53: 15}
trans_hy = {rdkit.Chem.rdchem.HybridizationType.S: 1, rdkit.Chem.rdchem.HybridizationType.SP: 2, rdkit.Chem.rdchem.HybridizationType.SP2: 3, rdkit.Chem.rdchem.HybridizationType.SP3: 4, rdkit.Chem.rdchem.HybridizationType.SP3D: 5}
trans_chi = {rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2}
for i in range(len(vocab)):
    f = vocab[i].split('_')[0]
    nf = vocab[i].split('_')[2]
    f = Chem.MolFromSmiles(f)
    nf = Chem.MolFromSmiles(nf)

    atoms_f = f.GetAtoms()
    atoms_nf = nf.GetAtoms()
    # f_list = []
    # for i in range(len(atoms_f)):
    #     if atoms_f[i].GetAtomicNum() == 9:
    #         f_list.append(atoms_f[i])
    #
    # nf_list = []
    # for i in range(len(atoms_nf)):
    #     if atoms_nf[i].GetAtomicNum() == 9:
    #         nf_list.append(atoms_nf[i])

    all = []

    for j in range(len(atoms_f)):

        num = trans_num[atoms_f[j].GetAtomicNum()] if atoms_f[j].GetAtomicNum() in trans_num else 17 #(18)
        hybrid = trans_hy[atoms_f[j].GetHybridization()] if atoms_f[j].GetHybridization() in trans_hy else 6  # [SP3, SP2, ](7)
        degree = atoms_f[j].GetDegree() + 1 if atoms_f[j].GetDegree() < 5 else 6  # (7)
        aro = int(atoms_f[j].GetIsAromatic()) + 1  # 2为芳香原子(3)
        charge = 1 if atoms_f[j].GetFormalCharge() == 0 else 2  # 2带电荷(3)
        chiral = trans_chi[atoms_f[j].GetChiralTag()] + 1  # 手性(4)
        val = atoms_f[j].GetImplicitValence() + 1 if atoms_f[j].GetImplicitValence() < 4 else 5  # (6)
        all.append([num, hybrid, degree, aro, charge, chiral, val])

    all_nf = []
    for j in range(len(atoms_nf)):

        num = trans_num[atoms_nf[j].GetAtomicNum()] if atoms_nf[j].GetAtomicNum() in trans_num else 17 #(18)
        hybrid = trans_hy[atoms_nf[j].GetHybridization()] if atoms_nf[j].GetHybridization() in trans_hy else 6  # [SP3, SP2, ](7)
        degree = atoms_nf[j].GetDegree() + 1 if atoms_nf[j].GetDegree() < 5 else 6  # (7)
        aro = int(atoms_nf[j].GetIsAromatic()) + 1  # 2为芳香原子(3)
        charge = 1 if atoms_nf[j].GetFormalCharge() == 0 else 2  # 2带电荷(3)
        chiral = trans_chi[atoms_nf[j].GetChiralTag()] + 1  # 手性(4)
        val = atoms_nf[j].GetImplicitValence() + 1 if atoms_nf[j].GetImplicitValence() < 4 else 5  # (6)
        all_nf.append([num, hybrid, degree, aro, charge, chiral, val])
    #####邻接矩阵


    edges_f = []
    for bond in f.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edges_f.append([a, b])

    g_f = nx.Graph(edges_f).to_directed()
    edge_index_f = []
    for e1, e2 in g_f.edges:
        edge_index_f.append([e1, e2])
####################
    edges_nf = []
    for bond in nf.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edges_nf.append([a, b])

    g_nf = nx.Graph(edges_nf).to_directed()
    edge_index_nf = []
    for e1, e2 in g_nf.edges:
        edge_index_nf.append([e1, e2])
    #####

######################
    for j in range(len(all)):
        output.write(str(all[j][0]) + ' ' + str(all[j][1]) + ' ' + str(all[j][2]) + ' ' + str(all[j][3]) + ' ' + str(all[j][4]) + ' ' + str(all[j][5]) + ' ' + str(all[j][6]) + '$')
    output.write('_')

    for j in range(len(edge_index_f)):
        output.write(str(edge_index_f[j][0]) + ' ' + str(edge_index_f[j][1]) + '$')
    output.write('_')

    ##################



    for j in range(len(all_nf)):
        output.write(str(all_nf[j][0]) + ' ' + str(all_nf[j][1]) + ' ' + str(all_nf[j][2]) + ' ' + str(all_nf[j][3]) + ' ' + str(all_nf[j][4]) + ' ' + str(all_nf[j][5]) + ' ' + str(all_nf[j][6]) + '$')
    output.write('_')

    for j in range(len(edge_index_nf)):
        output.write(str(edge_index_nf[j][0]) + ' ' + str(edge_index_nf[j][1]) + '$')

    output.write('\n')









