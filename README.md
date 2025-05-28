# InComeS
This is the repository of the paper **InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing**


### Environments & Tools

**Pip installation**
```
conda create -y -n gist_env python=3.10
conda activate gist_env

pip install numpy==1.26.4 pandas nltk scipy scikit-learn
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install Liger-Kernel==0.5.4 accelerate==1.5.2
pip install transformers==4.45.2 tokenizers datasets evaluate wandb deepspeed==0.15.3 flash-attn==2.7.3
```
