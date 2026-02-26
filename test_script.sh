#!/bin/bash
conda create -n sccytotrek_env python=3.10 -y
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sccytotrek_env
pip install -e .
python demo_analysis.py
