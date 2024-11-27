#!/bin/bash

nvcc -std=c++14 SSSPMain.cu -o result
./result	../../dataset/uk2002/origin
