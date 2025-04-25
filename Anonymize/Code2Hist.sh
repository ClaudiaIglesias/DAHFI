#!/bin/bash

# Recibe el nombre cifrado del archivo y la contraseña
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Code> <Password>"
    exit 1
fi

# Si se consigue descifrar correctamente, se imprime el nombre original
Hist=$(echo "$1" | xxd -p -r | openssl aes-256-cbc -pbkdf2 -a -d -salt -k "$2" 2> /dev/null)
if [ $? -ne 0 ]; then
    echo "Error in the decryption"
    exit 1
else
    echo $Hist
    exit 0
fi
