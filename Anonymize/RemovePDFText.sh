#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Input File> <Output File>"
    exit 1
fi

# Creamos el nuevo pdf eliminando el texto del pdf
gs -o "$2" -sDEVICE=pdfwrite -dFILTERTEXT -sstdout=%stderr "$1" 2> /dev/null
exit $?
