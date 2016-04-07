#!/bin/bash
# run_acq.sh
# by: Fred Mailhot

# check the number of command line arguments
if [[ $# < 12 ]]
then
    echo "USAGE: $0 <# of runs> <# of generations> <popsize> <flow> <lexsize> <instances> <vh> <case> <pl> <neut> <knn> <coart>"
else
    for i in `seq $1`
        do
            echo "$2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}" > options.log
            python acq_model.py -g $2 -s $3 -f $4 -l $5 -i $6 -t train -v $7 -c $8 -p $9 -n ${10} -k ${11} -a ${12} > convo.log
            # move lexicons, training data to timestamp-named subfolder
            DIRNAME=$(date +%s)_$5_$4
            mkdir -p ${10}/$DIRNAME
            mv trainfile* ${10}/$DIRNAME
            mv lexicon* ${10}/$DIRNAME
            mv *.log ${10}/$DIRNAME
            echo "Not testing"
            python acq_model.py -g $2 -s $3 -f $4 -l $5 -i $6 -t test_prod -v $7 -c $8 -p $9 -n ${10} -k ${11} -a ${12}
            mv *.log ${10}/$DIRNAME
            #rm ${10}/$DIRNAME/lexicon*
        done
fi
