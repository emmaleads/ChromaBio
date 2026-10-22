"""RDKit helpers: validation, key properties, dye-class detection, structure image."""
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, rdMolDescriptors

# (name, SMARTS). Heuristic classification, not exhaustive.
DYE_CLASSES = [
    ("Azo", "[#6][NX2]=[NX2][#6]"),
    ("Anthraquinone", "O=C1c2ccccc2C(=O)c2ccccc12"),
    ("Xanthene (e.g. rhodamine, fluorescein)",
     "[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1~[#8]~2"),
    ("Triarylmethane", "[CX3;!R](c)(c)=[C;R]"),
    ("Phenothiazine / thiazine", "c1ccc2[#7]~[#6]3~[#6]~[#6]~[#6]~[#6]~[#6]~3~[#16]~c2c1"),
    ("Indigoid", "O=C1c2ccccc2NC1=C1Nc2ccccc2C1=O"),
    ("Nitro", "[N+](=O)[O-]"),
]


def parse(smiles):
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles else None
    return mol


def canonical(mol):
    return Chem.MolToSmiles(mol)


def key_properties(mol):
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 2),
        "logP (Crippen)": round(Crippen.MolLogP(mol), 2),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "H-bond donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-bond acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Formal charge": Chem.GetFormalCharge(mol),
    }


def dye_classes(mol):
    found = []
    for name, sma in DYE_CLASSES:
        patt = Chem.MolFromSmarts(sma)
        if patt is not None and mol.HasSubstructMatch(patt):
            found.append(name)
    return found


def structure_image(mol, size=(420, 320)):
    return Draw.MolToImage(mol, size=size)
