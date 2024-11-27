#!/bin/bash

nvcc -std=c++14  Main.cu -o res
./res ../../dataset/cnr-2000
