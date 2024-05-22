# ML Implicit Solvent

## Getting started

First, set up your conda enironment. Just run `source setup_env.sh`. This will create a new `solv` conda environment (this will go a lot faster if you have `mamba` installed).

To use the `BigBindSolv` dataset, you'll need to tell the configuration where that folder is. Just create a file `configs/local.yaml`, and put it in there:
```yaml
bigbind_solv_dir: /path/to/bigbind/solv/dataset
```
Just put the actual path to the dataset (on highgarden, it's in `/trophsa/BigBindSolv`).

Currently, `train.py` just loops through the train dataset and does nothing.