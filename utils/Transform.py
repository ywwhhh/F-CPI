class Transform():
    def __init__(self):
        ##38
        self.ligand = ['#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N',
         'O', 'P', 'S', '[', '\\', ']', 'a', 'c', 'e', 'i', 'l', 'n', 'o', 'r', 's']
        ##34
        self.protein = ['M', 'L', 'A', 'R', 'N', 'P', 'Q', 'V', 'E', 'G', 'D', 'T', 'K', 'C', 'W', 'I', 'F', 'S', 'Y', 'H', 'U', 'X']
        ##68
        self.reaction = ['IC50', 'Kd', 'Potency', 'EC50', 'Ki', 'K2', 'AC50', 'Activity', 'IP', 'Kb', 'ID50', 'Inhibition', '2PT',
         'AbsAC1000', 'ED50', 'Kinact', 'IC90', 'EC20', 'Km', 'IC5', 'EC10', 'Ratio', 'IC95', 'INH', 'Ke', 'MEC',
         'Ke(app)', 'IC30', 'PT', 'EC1.5', 'Kii', 'Kis', 'EC2', 'Synergy', 'Metabolism', 'Kieq', 'AbsAC40', 'pC3',
         'EC90', 'Vm', 'CD', 'Emax', 'EC30', "Ki''", 'CD50', 'Kbapp', 'IC20', 'CC50', 'Efficacy', 'EC', 'EC15',
         'AbsAC1', 'MMC', 'C50', 'Ka', 'pA2', 'pC2A', 'GI50', 'EC60', 'fIC50', 'Kr', 'IC80', 'K0.5', 'K1', 'Max', 'K',
         'Concentration', "Ki'"]



    def ligand_enc_dict_1(self, bias):
        ldict = {}
        for i in range(len(self.ligand)):
            ldict[self.ligand[i]] = i + bias
        return ldict

    def protein_enc_dict_1(self, bias):
        pdict = {}
        for i in range(len(self.protein)):
            pdict[self.protein[i]] = i + bias
        return pdict

    def reaction_enc_dict(self, bias):
        rdict = {}
        for i in range(len(self.reaction)):
            rdict[self.reaction[i]] = i + bias
        return rdict

    def get_enc(self, str1, mode, n_gram, bias):
        dict1 = eval("self.{}_enc_dict_{}".format(mode, n_gram))(bias)
        a = []
        for i in range(len(str1)-n_gram+1):
            a.append(dict1[str1[i:i+n_gram]])
        return a

    def get_reac(self, str1, bias):
        dict1 = self.reaction_enc_dict(bias)
        return dict1[str1]





