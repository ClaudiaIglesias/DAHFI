#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <Input Folder> <Output Folder> <Password>"
    exit 1
fi

FOLDER="$1"
FOLDERO="$2"
PASSWR="$3"

mkdir -p "$FOLDERO"

# número de pdf en el directorio Input Folder
total=$(find $FOLDER -name "*.pdf" | wc -l)
n=0
for i in "$FOLDER"/*.pdf
do  
    n=$((n+1))
    # Extraemos el nombre del pdf
    name=$(basename -s .pdf "$i")
    # Creamos el nombre del nuevo archivo
    cname="$(bash Hist2Code.sh "$name" "$PASSWR").pdf"

    echo -n "$n/$total. Converting \"$name.pdf\" to \"$cname\"..."
    # Creamos el nuevo archivo, anonimizado
    bash RemovePDFText.sh "$i" "$FOLDERO/$cname"
    
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31m Error\033[0m"
        rm -f "$FOLDERO/$cname"
        exit 1
    else
        echo -e "\033[1;32m Done\033[0m"
    fi
done
