# DS-Unit-4-Sprint-1-NLP

## Setting Up Conda Environment

From Command Line inside folder where requirements.txt file is located:
```sh
conda create -n conda-env-name python==3.8
```
- Activate environment:
```sh
conda activate conda-env-name
```
- Add packages for this sprint:
```sh
pip install -r requirements.txt
```
- Add Ipython Kernel reference for use from JupyterLab:
```sh
python -m ipykernel install --user --name conda-env-name --display-name "U4-S1-NLP (Python3)"
```
- Install spacy models:
```sh
python -m spacy download en_core_web_md
```
```sh
python -m spacy download en_core_web_lg
```
