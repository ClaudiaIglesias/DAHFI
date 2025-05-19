import os
import skfda
import pandas as pd
import numpy as np
from src.preprocessing import preprocessing_signals
from src.preprocessing import preprocessing_signals 
from src.linear_features import get_linear_features
from src.nonlinear_features import get_nonlinear_features
from src.morphological_features import get_morphological_features
from src.database import create_clinical_data, get_targets_by_pH, create_clinical_data_excel



def combine_files(data_folder = 'bbdd/data/', output_folder = 'bbdd/', compressed=False, extra_data=False):
    """ Function that combines all the files in the data folder into a single CSV file
    
    Parameters:
        - data_folder: path of the folder containing the data files
        - output_folder: path of the folder to store the combined CSV file
        - compressed: whether the files are compressed or not
        - extra_data: whether to include the data from antoher csv file or not
    """

    # To store FHR and UC values for each file
    fhr_data_dict = {}
    uc_data_dict = {}

    ################ Check if file already exists and has all data to avoid reprocessing

    # Count the number of CSV files in the data folder
    num_files = sum(1 for filename in os.listdir(data_folder) if filename.endswith('.csv'))

    # Paths for the output CSV files
    fhr_csv_path = output_folder + 'fhr_data.csv'
    uc_csv_path = output_folder + 'uc_data.csv'

    # Check if the result files already exist
    if os.path.exists(fhr_csv_path) and os.path.exists(uc_csv_path):
        # Read the existing files to check their row counts
        existing_fhr_df = pd.read_csv(fhr_csv_path)
        existing_uc_df = pd.read_csv(uc_csv_path)

        # If the existing FHR and UC DataFrames have the same number of rows as the number of files, exit function
        if existing_fhr_df.shape[0] == num_files and existing_uc_df.shape[0] == num_files:
            print("The files fhr_data.csv and uc_data.csv already contain the expected number of rows. No need to create them.")
            return

    ################ Process the files

    # Loop through each file in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            base_filename = filename.split('.')[0]

            # Read the file
            if compressed:
                df = pd.read_csv(file_path, skiprows=[1], compression='gzip')
            else:
                df = pd.read_csv(file_path, skiprows=[1]) 

            # Correct the column names, deleting the extra ''
            df.columns = df.columns.str.replace("'", "").str.strip()

            fhr_data = df['FHR'].values
            uc_data = df['UC'].values
            
            elapsed_time = np.arange(0, len(fhr_data) * 0.25, 0.25)  

            # Store the FHR and UC values in a DataFrame
            fhr_df = pd.DataFrame(data=fhr_data.reshape(1, -1), columns=elapsed_time)
            uc_df = pd.DataFrame(data=uc_data.reshape(1, -1), columns=elapsed_time)
            
            # Store the first row as the file name
            fhr_df.insert(0, 'FileName', base_filename)  
            uc_df.insert(0, 'FileName', base_filename)  

            # Append to the data dictionary
            fhr_data_dict[base_filename] = fhr_df
            uc_data_dict[base_filename] = uc_df

    # Combine all DataFrames
    final_fhr_df = pd.concat(fhr_data_dict.values(), ignore_index=True)
    final_uc_df = pd.concat(uc_data_dict.values(), ignore_index=True)


    ################### If extra fhr data is provided, load it and concatenate

    if extra_data:
        # Load the data
        extra_fhr_path = output_folder + 'fhr_12O.csv'
        extra_fhr_df = pd.read_csv(extra_fhr_path, compression='gzip')

        # Rename the first column to 'FileName' to match the format
        if extra_fhr_df.columns[0] != 'FileName':
            extra_fhr_df = extra_fhr_df.rename(columns={extra_fhr_df.columns[0]: 'FileName'})

        # Concatenate
        final_fhr_df = pd.concat([final_fhr_df, extra_fhr_df], ignore_index=True)


    # Save the combined DataFrame to a CSV file
    final_fhr_df.to_csv(fhr_csv_path, index=False)
    final_uc_df.to_csv(uc_csv_path, index=False)




def create_processed_file(output_folder = 'bbdd/'):
    """ Function that preprocesses the FHR and UC signals and saves the processed data to a new CSV file
    
    Parameters:
        - output_folder: path of the folder to store the processed CSV file"""

    fhr_df = pd.read_csv(output_folder + 'fhr_data.csv', index_col=0)
    uc_df = pd.read_csv(output_folder + 'uc_data.csv', index_col=0)

    # Check if the result files already exist and have the same number of rows as the original files
    if os.path.exists(output_folder + 'fhr_data_processed.csv') and os.path.exists(output_folder + 'uc_data_processed.csv'):
        # Read the existing files to check their row counts
        fhr_proc_df = pd.read_csv(output_folder + 'fhr_data_processed.csv')
        uc_proc_df = pd.read_csv(output_folder + 'uc_data_processed.csv')

        # If the existing FHR and UC DataFrames have the same number of rows as the original files, exit function
        if fhr_proc_df.shape[0] == fhr_df.shape[0] and uc_proc_df.shape[0] == uc_df.shape[0]:
            print("The processed files already contain the expected number of rows. No need to reprocess.")
            return
    

    # Get the filenames from the original file
    filenames_fhr = fhr_df.index
    filenames_uc = uc_df.index

    # The data without the filename column
    fhr_np = fhr_df.iloc[:, 1:].to_numpy()
    uc_np = uc_df.iloc[:, 1:].to_numpy()

    # Preprocessing
    fhr_proc = preprocessing_signals(fhr_np)
    uc_proc = preprocessing_signals(uc_np) 

    fhr_proc_np = np.array(fhr_proc.data_matrix)[:,:,0]
    uc_proc_np = np.array(uc_proc.data_matrix)[:,:,0]

    # Create the dataframe and add the filenames as the first column
    fhr_proc_df = pd.DataFrame(fhr_proc_np)
    fhr_proc_df.insert(0, 'FileName', filenames_fhr)
    uc_proc_df = pd.DataFrame(uc_proc_np)
    uc_proc_df.insert(0, 'FileName', filenames_uc)

    # Save the processed data to a new CSV file
    fhr_proc_df.to_csv(output_folder + 'fhr_data_processed.csv', index=False)
    uc_proc_df.to_csv(output_folder + 'uc_data_processed.csv', index=False)



def create_matrix(hea_folder='bbdd/extra/', output_folder='bbdd/', feature_set=['linear'], excel=False):
    """ Function that creates a matrix of selected features from the FHR data and adds the class column with the pH values

    Parameters:
        - hea_folder: path to the folder containing the .hea files
        - output_folder: path to the folder where to store the features file
        - feature_set: list of features to extract ('linear', 'nonlinear', 'morphological')
        - excel: whether to use the Excel file with pH (excel = true) values or the .hea files (excel = false) """

    # Load the FHR data
    fhr_df = pd.read_csv(output_folder + 'fhr_data_processed.csv', index_col=0)
    fhr_df.index = fhr_df.index.astype(str)
    fhr_np = fhr_df.to_numpy()

    # Set up the DataFrame to store the selected features
    selected_features = pd.DataFrame(index=fhr_df.index)

    # Obtain linear features if selected
    if 'linear' in feature_set:
        linear_columns = ['mean', 'std', 'LTV', 'delta', 'STV', 'II']
        # If the linear features have already been calculated, load them
        if os.path.exists(output_folder + 'linear_features.csv'):
            linear_features = pd.read_csv(output_folder + 'linear_features.csv', index_col=0)
            linear_features.index = linear_features.index.astype(str)
            linear_features = linear_features.iloc[:, :-1]  # Remove the last column (class)
        else:
            linear_data = get_linear_features(fhr_np).T
            linear_features = pd.DataFrame(data=linear_data, index=fhr_df.index, columns=linear_columns)
        selected_features = pd.concat([selected_features, linear_features], axis=1)


    # Obtain nonlinear features if selected
    if 'nonlinear' in feature_set:
        nonlinear_columns = ['ApEn_0.15', 'ApEn_0.20', 'SampEn_0.15', 'SampEn_0.20', 'LZC']
        # If the nonlinear features have already been calculated, load them
        if os.path.exists(output_folder + 'nonlinear_features.csv'):
            nonlinear_features = pd.read_csv(output_folder + 'nonlinear_features.csv', index_col=0)
            nonlinear_features.index = nonlinear_features.index.astype(str)
            nonlinear_features = nonlinear_features.iloc[:, :-1]  # Remove the last column (class)
        else:
            nonlinear_data = get_nonlinear_features(fhr_np).T
            nonlinear_features = pd.DataFrame(data=nonlinear_data, index=fhr_df.index, columns=nonlinear_columns)
        selected_features = pd.concat([selected_features, nonlinear_features], axis=1)


    # Obtain morphological features if selected
    if 'morphological' in feature_set:
        morphological_columns = ['baseline', 'dcc', 'acc']
        # If the morphological features have already been calculated, load them
        if os.path.exists(output_folder + 'morphological_features.csv'):
            morphological_features = pd.read_csv(output_folder + 'morphological_features.csv', index_col=0)
            morphological_features.index = morphological_features.index.astype(str)
            morphological_features = morphological_features.iloc[:, :-1]  # Remove the last column (class)
        else:
            morphological_data = get_morphological_features(fhr_np).T
            morphological_features = pd.DataFrame(data=morphological_data, index=fhr_df.index, columns=morphological_columns)
        selected_features = pd.concat([selected_features, morphological_features], axis=1)


    # Generate clinical data and get the pH labels
    if excel:
        create_clinical_data_excel(output_folder + "/extra/clases.xls", output_folder + "extra/clinical_12O.csv", output_folder)
    else:
        create_clinical_data(hea_folder, output_folder)

    pH_threshold = 7.2
    ph_classes  = get_targets_by_pH(pH_threshold, output_folder)

    
    selected_features['pH'] = selected_features.index.map(ph_classes)
    
    # Delete rows with NaN values in the 'pH' column
    selected_features = selected_features.dropna(subset=['pH'])

    # Convert the 'pH' column to integer type
    selected_features['pH'] = selected_features['pH'].astype(int)


    # Store the selected features in a CSV file
    if len(feature_set) == 1:
        feature_names = feature_set[0]
    else:
        feature_names = '_'.join([feature[:6] for feature in feature_set])

    output_file = output_folder + f'{feature_names}_features.csv'
    selected_features.to_csv(output_file, index=True)





def plot_processed_data(file, data_folder = 'bbdd/data/', main_folder = 'bbdd/', compression = False):
    """ Function that plots the original and processed FHR and UC signals for a given file
    
    Parameters:
        - file: name of the file to plot, without the folder name or extension
        - data_folder: path of the folder containing the data files
        - main_folder: path of the folder containing the processed data files
        - compression: whether the files are compressed or not"""

    import matplotlib.pyplot as plt

    # Load the data
    filename = data_folder + file + ".csv"

    if compression:
        original_data = pd.read_csv(filename, skiprows=[1], compression='gzip')
    else:
        original_data = pd.read_csv(filename, skiprows=[1]) 

    # Load the processed FHR and UC data from the new files
    processed_fhr_df = pd.read_csv(main_folder + 'fhr_data_processed.csv', index_col=0)
    processed_fhr_df.index = processed_fhr_df.index.astype(str)
    processed_fhr_row = processed_fhr_df.loc[file]

    processed_uc_df = pd.read_csv(main_folder + 'uc_data_processed.csv', index_col=0)
    processed_uc_df.index = processed_uc_df.index.astype(str)
    processed_uc_row = processed_uc_df.loc[file]

    #Correct the column names, deleting the extra '' and whitespaces
    original_data.columns = original_data.columns.str.replace("'", "").str.strip()

    # Extract FHR and UC signals
    original_fhr = original_data['FHR'].to_numpy()
    original_uc = original_data['UC'].to_numpy()
    processed_fhr = processed_fhr_row.to_numpy()
    processed_uc = processed_uc_row.to_numpy()

    # Create time arrays based on the number of entries
    time_original = np.arange(0, len(original_fhr) * 0.25, 0.25)  
    time_processed = np.arange(0, len(processed_fhr) * 0.25, 0.25)  

    # Show only the last seconds of the original one to match the processed one
    num_samples = 20.5 * 60 / 0.25
    last_original_fhr = original_fhr[-int(num_samples):]
    last_original_uc = original_uc[-int(num_samples):]
    time_last_original = np.linspace(0, 1100, len(last_original_fhr))

    # Create subplots
    plt.figure(figsize=(10, 15))

    # Original Data - FHR
    plt.subplot(6, 1, 1)
    plt.plot(time_original, original_fhr, label='Original FHR', color='b')
    plt.title('Original FHR')
    plt.xlabel('Time (seconds)')
    plt.ylabel('FHR (bpm)')
    plt.grid()

    # Original Data - UC
    plt.subplot(6, 1, 4)
    plt.plot(time_original, original_uc, label='Original UC', color='r')
    plt.title('Original UC')
    plt.xlabel('Time (seconds)')
    plt.ylabel('UC (nd)')
    plt.grid()

    # Processed Data - FHR
    plt.subplot(6, 1, 3)
    plt.plot(time_processed, processed_fhr, label='Processed FHR', color='b')
    plt.title('Processed FHR')
    plt.xlabel('Time (seconds)')
    plt.ylabel('FHR (bpm)')
    plt.grid()

    # Processed Data - UC
    plt.subplot(6, 1, 6)
    plt.plot(time_processed, processed_uc, label='Processed UC', color='r')
    plt.title('Processed UC')
    plt.xlabel('Time (seconds)')
    plt.ylabel('UC (nd)')
    plt.grid()

    # Last x seconds of Original Data
    plt.subplot(6, 1, 2)
    plt.plot(time_last_original, last_original_fhr, label='Last Original FHR', color='b')
    plt.title(f'Last 15 minutes of Original FHR')
    plt.xlabel('Time (seconds)')
    plt.ylabel('FHR (bpm)')
    plt.grid()

    # Last x seconds of Original Data - UC
    plt.subplot(6, 1, 5)
    plt.plot(time_last_original, last_original_uc, label='Last Original UC', color='r')
    plt.title(f'Last 15 minutes of Original UC')
    plt.xlabel('Time (seconds)')
    plt.ylabel('UC (nd)')
    plt.grid()

    plt.tight_layout()
    plt.show()



