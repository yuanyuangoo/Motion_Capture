#########################################################################
# File Name: run.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Wed Feb  8 17:32:54 2017
#########################################################################
#!/bin/bash
cd build
cmake ..
make
./Fast_RCNN ../Skating.mp4
cd ..
