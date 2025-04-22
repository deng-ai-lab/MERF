# MERF 

Multi-site reinforcing framework (MERF) is a deep learning model for antibody evolution that enables simultaneous mutations across multiple residues.

## Overview

Antibodies hold a vast potential for biomedical applications and therapeutic development primarily driven by their in vivo evolutionary mechanisms. Recent advances in artificial intelligence pave the path for computational antibody evolution by nominating single-residue mutations towards an optimized affinity. However, in nature, biological evolution is instead rooted in the simultaneous mutations across multiple residues. Modeling such multi-residue evolution remains challenging due to the vast evolutionary landscape, computational complexity, and local maxima traps. 

To address these challenges, we introduce MERF, a multi-agent reinforcement learning framework designed to efficiently explore multi-residue evolutionary space for antibody affinity enhancement. MERF redefines antibody evolution as a policy-making process rather than following traditional prediction-driven frameworks. MERF alternates between generating mutation policy and learning value function based on the binding energy change, providing an efficient reinforcing framework for discovering affinity-enhancing multi-site mutations while minimizing computational costs.


## MERF software package

MERF requires the following packages for installation:

- Python >= 3.8
- PyTroch = 1.12.1
- Numpy >= 1.23.5
- easydict = 1.10
- biopython = 1.79
- scipy = 1.10.1
- scikit-learn = 1.2.2

All required python packages can be installed through `pip/conda` command. 

To install MERF package, use

```terminal
git clone https://github.com/deng-ai-lab/MERF
```

To enable antibody evolution with Rosetta Docking protocol, users can install Rosetta and compile it in Massage Passing Interface (MPI) format, following [Rosetta Documents](https://docs.rosettacommons.org/demos/latest/tutorials/install_build/install_build)

## Dataset

The SKEMPI v2.0 dataset is downloaded from [SKEMPI v2](https://life.bsc.es/pid/skempi2/). The AB-Bind dataset is downloaded from [AB-Bind](https://github.com/sarahsirin/AB-Bind-Database).

For reproductivity, we provide structures after atom repairation in  `\data`, both wild-type and mutant Ab-Ag complexes.

## Usage

### Pretrain MERF

By running `pretrain_and_val.py` scripts, users can get pretrained MERF parameters with cross-validation results in AB-Bind dataset. 

### Antibody evolution

To enable antibody evolution, users should install and compile Rosetta in MPI format first. After installation, users can run the following code to make ensure the environment variables.

```
export PATH="{PATH_TO_MPI}/mpi_instll/bin/:$PATH"
export PATH="{PATH_TO_ROSETTA}/rosetta.source.release-340/main/source/bin/:$PATH"
export LD_LIBRARY_PATH="{PATH_TO_MPI}/mpi_instll/lib/":$LD_LIBRARY_PATH
```

The Rosetta software paths in `evo_abbind.py` and `evo_sars.py` are expected to changed as the software installation path. For evolution tasks, user can select run `evo_abbind.py` to evolve antibodies in AB-Bind and `evo_sars.py` to evolve antibodies to bind three major SARS-CoV-2 variants.

## Copyright
Software provided as is under **MIT License**.

Fengji Li @ 2025 BUAA and Deng ai Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

