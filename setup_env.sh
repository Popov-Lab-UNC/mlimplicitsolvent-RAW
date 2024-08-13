# check if mamba exists and if not, just use conda
if ! command -v mamba &> /dev/null
then
    alias mamba=conda
fi

mamba create -n solv python=3.10 -y && conda activate solv &&
mamba install -c conda-forge cudatoolkit=11.8 rdkit pytorch pytorch_geometric pytorch_cluster pytorch_sparse torch-scatter openmm pdbfixer openmoltools openff-toolkit openmmforcefields -y &&
pip install -r requirements.txt