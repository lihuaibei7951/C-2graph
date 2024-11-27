#!/bin/bash

nvcc -std=c++14 SSSPMain.cu -o result
./result	../../dataset/cnr-2000/origin 1

#./result	/home/lhb/cucode/CGgraphV1/graph/arabic 0
#./result	/home/lhb/cucode/CGgraphV1/graph/it2004 0
#./result	/home/lhb/cucode/CGgraphV1/graph/uk2002 0
#./result	/home/lhb/cucode/CGgraphV1/graph/gsh2015 0