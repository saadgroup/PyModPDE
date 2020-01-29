#!/bin/bash
dir=doc
if [ -d $dir ]
then
    pdoc --html src/pymodpde.py --output-dir doc -f
else
    pdoc --html src/pymodpde.py --output-dir doc
fi
