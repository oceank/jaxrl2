# jaxrl2

If you use JAXRL2 in your work, please cite this repository in publications:
```
@misc{jaxrl,
  author = {Kostrikov, Ilya},
  doi = {10.5281/zenodo.5535154},
  month = {10},
  title = {{JAXRL: Implementations of Reinforcement Learning algorithms in JAX}},
  url = {https://github.com/ikostrikov/jaxrl2},
  year = {2022},
  note = {v2}
}
```

## Installation
Create an anaconda environment with Python 3.9
```bash
    conda create -n path python=3.9
'''

[Install mujoco-py without the root privileges](https://github.com/openai/mujoco-py/issues/627)
```bash
    conda install -c conda-forge glew
    conda install -c conda-forge mesalib
    conda install -c menpo glfw3
```
Then add the conda environment include to CPATH
```bash
    export CPATH=$CONDA_PREFIX/include:${CPATH}
```
Finally, install patchelf with
```bash
    pip install patchelf
```

Install prerequite packages in requirements.txt, jaxrl2 and JAX with CUDA. Run
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # [Note: wheels only available on linux and it installs CUDA 11.](https://jax.readthedocs.io/en/latest/installation.html)
```


## Examples

[Here.](examples/)

## Tests

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= pytest tests
```

# Acknowledgements 

Thanks to [@evgenii-nikishin](https://github.com/evgenii-nikishin) for helping with JAX. And [@dibyaghosh](https://github.com/dibyaghosh) for helping with vmapped ensembles.
