#!/bin/bash

file_name=$1
pdf_folder=$2
svg_folder=$3

# Path to the PDF
pdf_file="$pdf_folder$file_name.pdf"

# Check if the PDF exists
if [ ! -f "$pdf_file" ]; then
    echo "Error: PDF file '$pdf_file' not found."
    exit 1
fi

# Check if the folder already exists, and exit if it does
if [ -d "${svg_folder}${file_name}" ]; then
    echo "SVG folder '${svg_folder}${file_name}' already exists. Skipping conversion"
    exit 1
fi

# Create the svg folder if it doesn't exist
mkdir -p "${svg_folder}${file_name}"

# Get the number of pages of the pdf
num_pages=$(pdfinfo "$pdf_file" | grep Pages | awk '{print $2}')

# Convert to svg
for ((i=1; i<=num_pages; i++))
do
    svg_file="${svg_folder}${file_name}/${file_name}-part$i.svg"

    # Check if the SVG file already exists
    if [ -f "$svg_file" ]; then
        continue
    fi


    #inkscape "$pdf_file" --pdf-page=$i  --export-plain-svg="${svg_folder}${file_name}/${file_name}-part$i.svg"
    # Versión 1.2.2 de inkscape:
    #inkscape "$pdf_file" --pdf-page=$i --export-filename="${svg_folder}${file_name}/${file_name}-part$i.svg" 
    # Versión 1.3.1 de inkscape:
    inkscape "$pdf_file" --pages=$i --export-type=svg --export-filename="$svg_file" >/dev/null 2>&1
    #2>> error_log.txt para que lo guarde en un archivo de errores

done