#/bin/bash

echo "No morphology, disharmonic"
time ./lex_maker.py -v false -c false -p false
echo "plural morphology, disharmonic"
time ./lex_maker.py -v false -c false -p true
echo "case morphology, disharmonic"
time ./lex_maker.py -v false -c true -p false
echo "all morphology, disharmonic"
time ./lex_maker.py -v false -c true -p true
