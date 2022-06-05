#!/bin/bash -e 

rm ourOuts/*


for i in $(seq -w 1 13)
do
	python3 takuzu.py < testes-takuzu/input_T$i > ourOuts/out$i
	DIFF = $(diff testes-takuzu/output_T$i ourOuts/out$i) 

	if [ "$DIFF" != "" ] 
	then
		echo "Damn, $i not Good"
	else
		echo "All Good $i"
	fi

done
