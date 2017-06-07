#!/bin/bash

source /opt/intel/bin/compilervars.sh intel64 -platform linux
args=$@
echo  "/home/scidb/pilot/pcc/apps/pcc/PCC $args"
nohup bash -c "/home/scidb/pilot/pcc/apps/pcc/PCC $args" &

