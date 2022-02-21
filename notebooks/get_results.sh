#!/bin/bash

for i in {0..11}
do 
	cat results_ro_en.txt | grep "LAYER: ${i} | HEAD: \d" -A 1 | grep AUC | cut -d ' ' -f 3 | tr '\n' '\t'
	echo ""
done
cat results_ro_en.txt | grep "LAYER: \d*$" -A 1 | grep AUC | cut -d ' ' -f 3
cat results_ro_en.txt | grep AUC | head -n 1


echo ""
echo "======================="
echo ""


for i in {0..11}
do 
	cat results_ro_en.txt | grep "LAYER: ${i} | HEAD: \d" -A 4 | grep "Recall" -A 1 | grep AUC | cut -d ' ' -f 3 | tr '\n' '\t'
	echo ""
done
cat results_ro_en.txt | grep "LAYER: \d*$" -A 4 | grep "Recall" -A 1 | grep AUC | cut -d ' ' -f 3
cat results_ro_en.txt | grep AUC | head -n 2 | tail -n 1