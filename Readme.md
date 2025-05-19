# Detección de Hipoxia Fetal con Machine Learning

Este proyecto forma parte de un Trabajo de Fin de Grado y tiene como objetivo aplicar técnicas de aprendizaje automático para la detección de hipoxia fetal utilizando señales de frecuencia cardíaca fetal (FHR) y otros datos clínicos. Se utilizan dos conjuntos de datos distintos: la base de datos pública CTU-CHB y datos reales del Hospital 12 de Octubre (H12O).

## Estructura del Proyecto

- `process_ctu-chb.ipynb`: Notebook donde se muestra todo el proceso aplicado sobre la base de datos **CTU-CHB**, incluyendo:
  - Preprocesado de las señales FHR.
  - Extracción de características (lineales, no lineales y morfológicas).
  - Entrenamiento y evaluación de varios modelos de Machine Learning con optimización de hiperparámetros.
  - Análisis de resultados y visualización de métricas (precisión, F1-score, etc.).

- `process_h12O.ipynb`: Notebook donde se procesa la base de datos del **Hospital 12 de Octubre**, incluyendo:
  - Preprocesado de las señales FHR.
  - Extracción de características (lineales, no lineales y morfológicas).
  - Entrenamiento y evaluación de varios modelos de Machine Learning con optimización de hiperparámetros.
  - Análisis de resultados y visualización de métricas (precisión, F1-score, etc.).


  Nota: La base de datos del Hospital 12 de Octubre aún no puede ser publicada, ya que el hospital está recopilando más datos antes de su publicación oficial. Por esto, este notebook no va a poder ser probado hasta que no se publiquen dichos datos.

- `Anonymize`: Contiene los scripts necesarios para anonimizar los PDFs con las señales clínicas.

## Requisitos

Para ejecutar los notebooks, se debe tener instalado Python en una versión compatible:

- **Python >= 3.9 y < 3.13**

Hay disponible un archivo _requirements.txt_ para poder instalar las dependencias con:

```bash
pip install -r requirements.txt
```

Además, en caso de querer ejecutar el paso de PDF a CSV, es necesario instalar **Inskacape** con el comando:

```bash
sudo apt install inkscape -ys
```

**El paso de PDF a CSV solo se puede hacer en linux ya que necesita ejecutar un script de bash.**


## Uso de la base de datos CTU-CHB

La base de datos pública **CTU-CHB** puede descargarse desde [PhysioNet](https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/).

Para poder utilizarla con este proyecto, se deben organizar los archivos de la siguiente manera:

```
ctu-chb/
├── data/
│ └── *.csv     
└── extra/
└── *.hea       
```

- Los archivos `.csv` deben ir en la carpeta `ctu-chb/data/`.
- Los archivos `.hea`, en `ctu-chb/extra/`.

Si se utiliza una estructura de carpetas diferente, es posible modificar las variables `main_folder`, `data_folder` y `hea_folder` en la primera celda del notebook para que se adapten a la organización elegida.





