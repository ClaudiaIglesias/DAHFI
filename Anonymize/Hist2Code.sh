#!/bin/bash

# Creamos un diccionario
DICT=Hist2Code.dic

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <History Number> <Password>"
    exit 1
fi


if [ -f "$DICT" ]; then
# Si ya existe esa entrada en el diccionario
    if grep -q "$1" "$DICT"; then
        grep "$1" "$DICT" | head -n 1 | cut -f 2
        exit 0
    fi
fi

# Si no existe, crea un código nuevo (openssl cifra el History Number con la contraseña como clave para el cifrado)
code=$(echo "$1" | openssl aes-256-cbc -pbkdf2 -a -salt -k "$2" | xxd -p | tr -d '\n' | tr [a-z] [A-Z])

# Guardamos el nuevo código en el diccionario
echo $code
echo -e "$1\t$code" >> $DICT

exit 0
