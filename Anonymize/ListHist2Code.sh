#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <Input File (List of Hists)> <Output File (List of Codes)> <Password>"
    exit 1
fi

rm -f "$2"

# Lee el fichero Input File que debería tener una lista con los nombres de los ficheros sin cifrar
# Los cifra y los guarda en el archivo OutputFile (original + cifrado)
while read line; do
    echo -n -e "$line\t" >> $2
    bash Hist2Code.sh "$line" "$3" >> "$2"
done < "$1"
