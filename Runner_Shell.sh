#!/bin/bash -l


for l2 in $(seq 0.1 0.1 1);
do

   l=1;
   l1=1;
   l3=1;
   
   export l l1 l2 l3
   python NonLinear_GD.py "$l" "$l1" "$l2" "$l3" &

done
