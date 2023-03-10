#!/bin/bash
mkdir -p modules
cd modules

python -m pip download numpy==1.22.0 gymnasium==0.26.3 pettingzoo==1.22.3 # tianshou==0.4.1 torch==1.13.1

unzip -o '*.whl'
rm *.whl