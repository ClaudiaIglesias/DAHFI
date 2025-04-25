import pandas as pd
import numpy as np
import math
import scipy
import os, sys
import matplotlib.pyplot as plt


def get_targets_by_pH(pH_threshold=None, path_csv='data/'):
    """ Function that return targets depending on the pH value
    
    Parameters:
        - pH_threshold: threshold value for classification or None for regression
        - path_csv: path to the folder with the clinical data to extract pHs
        
    Output:
        - class_dict: dictionary with the nhc as key and the class as value (0 if pH <= pH_threshold, 1 if pH > pH_threshold, NaN if pH is NaN)"""
    
    data_df = pd.read_csv(path_csv+'clinical_data.csv', index_col=0, compression='gzip')

    phs = data_df['ph'].values.astype(float)

    # Create a dictionary with the nhc and the class (0 if pH <= pH_threshold, 1 if pH > pH_threshold, NaN if pH is NaN)
    class_dict = {str(nhc): (np.nan if np.isnan(ph) else int(ph <= pH_threshold)) for nhc, ph in zip(data_df.index, phs)}

    return class_dict


def create_clinical_data(path_hea='ctu-chb/', path_clinical='data/'):
    """ Function that creates a CSV file with clinical data from the CTU-CHB database.
    
    Parameters:
        - path_hea: path to the folder with the hea files
        - path_clinical: path to the folder where save the csv file"""

    all_files = sorted(os.listdir(path_hea))
    data_matrix = []
    numbers = []
    cols = ['ph', 'bdecf', 'pco2', 'be', 'apgar1', 'apgar5', 'gest. weeks', 'weight', 'sex', 'age', 'gravidity', 
               'parity', 'diabetes', 'hypertension', 'preeclampsia', 'liq. praecox', 'pyrexia', 'meconium',
               'presentation', 'induced', 'I. stage', 'noProgress', 'CK/KP', 'II. stage', 'deliv. type']

    for file in all_files:
        if ".hea" in file:
            numbers.append(file.split('.')[0])
            with open(path_hea+file, "r") as file:
                for line in file:
                    if "#pH" in line:
                        ph = str(line.split()[1])
                    elif "#BDecf" in line:
                        bdecf = str(line.split()[1])
                    elif "#pCO2" in line:
                        pco2 = str(line.split()[1])
                    elif "#BE" in line:
                        be = str(line.split()[1])
                    elif "#Apgar1" in line:
                        apgar1 = str(line.split()[1])
                    elif "#Apgar5" in line:
                        apgar5 = str(line.split()[1])
                    
                    elif "#Gest. weeks" in line:
                        weeks = str(line.split()[2])
                    elif "#Weight(g)" in line:
                        weight = str(line.split()[1])
                    elif "#Sex" in line:
                        sex = str(line.split()[1])
                    
                    elif "#Age" in line:
                        age = str(line.split()[1])
                    elif "#Gravidity" in line:
                        gravidity = str(line.split()[1])
                    elif "#Parity" in line:
                        parity = str(line.split()[1])
                    elif "#Diabetes" in line:
                        diabetes = str(line.split()[1])
                    elif "#Hypertension" in line:
                        hypertension = str(line.split()[1])
                    elif "#Preeclampsia" in line:
                        preeclampsia = str(line.split()[1])
                    elif "#Liq. praecox" in line:
                        liq_praecox = str(line.split()[2])
                    elif "#Pyrexia" in line:
                        pyrexia = str(line.split()[1])
                    elif "#Meconium" in line:
                        meconium = str(line.split()[1])
                    
                    elif "#Presentation" in line:
                        presentation = str(line.split()[1])
                    elif "#Induced" in line:
                        induced = str(line.split()[1])
                    elif "#I.stage" in line:
                        I_stage = str(line.split()[1])
                    elif "#NoProgress" in line:
                        no_progress = str(line.split()[1])
                    elif "#CK/KP" in line:
                        ck_kp = str(line.split()[1])
                    elif "#II.stage" in line:
                        II_stage = str(line.split()[1])
                    elif "#Deliv. type" in line:
                        deliv_type = str(line.split()[2])

            data = np.array([ph, bdecf, pco2, be, apgar1, apgar5, weeks, weight, sex, age, gravidity, parity, diabetes,
                           hypertension, preeclampsia, liq_praecox, pyrexia, meconium,
                           presentation, induced, I_stage, no_progress, ck_kp, II_stage, deliv_type])

            data_matrix.append(data)

    data_df = pd.DataFrame(data=data_matrix, index=numbers, columns=cols)
    data_df.to_csv(path_clinical+'clinical_data.csv', compression='gzip')
    
    return



def create_clinical_data_excel(excel_file, path_output='csv/'):
    """ Function that creates a CSV file with the NHC and pH values from the Excel file.
    
    Parameters:
        - excel_file: path of the Excel file to extract the data from.
        - path_output: path where to save the CSV file"""

    # Load the Excel file
    data_df = pd.read_excel(excel_file, dtype=str)

    # Select only the columns we need (NHC and pH)
    selected_cols = ["NHC", "PH_INTRAPARTO", "PH_CORDON_ARTERIA", "PH_CORDON_VENA"]
    data_df = data_df[selected_cols].copy()

    # Remove rows with NaN in NHC
    data_df = data_df.dropna(subset=["NHC"])


    # Convert columns to numeric, ignoring errors
    for col in ["PH_INTRAPARTO", "PH_CORDON_ARTERIA", "PH_CORDON_VENA"]:
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce")


    # For each value of data_df, ph_cordon_arteria if it exists, if not ph_cordon_vena, if not ph_intraparto, if not NaN
    data_df["ph"] = data_df["PH_CORDON_ARTERIA"].combine_first(data_df["PH_CORDON_VENA"]).combine_first(data_df["PH_INTRAPARTO"])

    # Remove the individual pH columns
    data_df.drop(columns=["PH_INTRAPARTO", "PH_CORDON_ARTERIA", "PH_CORDON_VENA"], inplace=True)

    # Set the NHC as index (file name)
    data_df.set_index("NHC", inplace=True)

    # Save as compressed CSV
    data_df.to_csv(f"{path_output}clinical_data.csv", compression="gzip")

    return