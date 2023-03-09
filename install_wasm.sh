#!/bin/bash
mkdir -p modules
cd modules

python -m pip download pettingzoo==1.22.3 # tianshou==0.4.1 torch==1.13.1

unzip -o '*.whl'
rm *.whl