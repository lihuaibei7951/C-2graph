#!/bin/bash

nvcc -std=c++14 SSSPMain.cu -o result
./result	../../dataset/cnr-2000/origin
