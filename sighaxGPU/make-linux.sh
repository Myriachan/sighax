#!/bin/bash
COMMAND="nvcc -std=c++11 -m64 -O3 -D_FORCE_INLINES kernel.cu Main.cpp -lgmp -o sighaxGPU"
echo $COMMAND
$COMMAND
