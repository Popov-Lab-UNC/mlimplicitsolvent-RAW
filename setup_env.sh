# check if mamba exists and if not, just use conda
if ! command -v mamba &> /dev/null
    alias mamba=conda
fi

mamba create -n solv python=3.10 -y && conda activate solv &&
mamba install -c conda-forge cudatoolkit=11.8 -y &&
mamba install -c conda-forge -c omnia openmm openff-toolkit openff-forcefields openmmforcefields openmmtools torchmd-net -y &&
pip install -r requirements.txt