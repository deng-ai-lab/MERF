from Bio.PDB.Polypeptide import three_to_one, three_to_index, is_aa

aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}

BACKBONE_ATOMs = ['N', 'CA', 'C', 'O']

NON_STANDARD_SUBSTITUTIONS = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

# given residue type, return unique sidechain postfixes as specific ordering
RESIDUE_SIDECHAIN_POSTFIXES = {
    'A': ['B'],
    'R': ['B', 'G', 'D', 'E', 'Z', 'H1', 'H2'],
    'N': ['B', 'G', 'D1', 'D2'],
    'D': ['B', 'G', 'D1', 'D2'],
    'C': ['B', 'G'],
    'E': ['B', 'G', 'D', 'E1', 'E2'],
    'Q': ['B', 'G', 'D', 'E1', 'E2'],
    'G': [],
    'H': ['B', 'G', 'D1', 'D2', 'E1', 'E2'],
    'I': ['B', 'G1', 'G2', 'D1'],
    'L': ['B', 'G', 'D1', 'D2'],
    'K': ['B', 'G', 'D', 'E', 'Z'],
    'M': ['B', 'G', 'D', 'E'],
    'F': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'Z'],
    'P': ['B', 'G', 'D'],
    'S': ['B', 'G'],
    'T': ['B', 'G1', 'G2'],
    'W': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'E3', 'Z2', 'Z3', 'H2'],
    'Y': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'Z', 'H'],
    'V': ['B', 'G1', 'G2'],
}

GLY_INDEX = 5
ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4



def augmented_three_to_one(three):
    """
    input three code for an aa, first change the non_standard(potential) into uniform code, then change to one code
    :param three: input three code (name) for an aa
    :return: one code (name) for an aa
    """
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return three_to_one(three)


def augmented_three_to_index(three):
    """
    input three code for an aa, first change the non_standard(potential) into uniform code, then change to index (0~19)
    :param three: input three code (name) for an aa
    :return: index 0~19
    """
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return three_to_index(three)


def augmented_is_aa(three):
    """
    input three code for an aa, first change the non_standard(potential) into uniform code, then using is_aa to check
    :param three: input three code (name) for an aa
    :return:is_aa: True, not an aa: False
    """
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return is_aa(three, standard=True)


def is_hetero_residue(res):
    return len(res.id[0].strip()) > 0