# nlp-generalization
Improving generalization of language models

Project inspired by Asa's work https://aclanthology.org/2021.mrl-1.20.pdf 

Setup a conda env and install the below dependencies:
```
conda create --name nlp_gen
conda activate nlp_gen
conda install transformers
conda install -c pytorch pytorch
conda install -c anaconda scikit-learn
conda install -c conda-forge datasets
pip install pyhessian
```

To experiment with this repo:
1. Clone this repository using:
`git clone https://github.com/sakshambassi/nlp-generalization.git`
2. Go inside the newly created dir:
`cd nlp-generalization`
3. Clone HFSharpness repo: 
`git clone https://github.com/sakshambassi/hfsharpness.git`
4. Now run any experiment like:
`python text_classification.py`
5. or run it on Greene server:
`sbatch greene_run.slurm`
