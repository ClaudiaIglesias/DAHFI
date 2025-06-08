import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class ModelTesting:
    def __init__(self, model, file_name = 'linear_features.csv', main_folder="bbdd/", standardize=False, replace_inf_with=1e10):
        """ Class to test a model finding the best parameters and using cross-validation.

        Parameters:
            - model: the model to test.
            - file_name: the name of the CSV file with the features.
            - main_folder: the folder where the CSV file is located.
            - standardize: whether to standardize the features.
            - replace_inf_with: value to replace inf values in the dataset.
        """
        self.model = model
        self.standardize = standardize
        self.load_and_prepare_data(main_folder, file_name, replace_inf_with)


    @staticmethod
    def handle_infinity_values(X, replace_with=1e10):
        """ Replace inf values with NaN.
        
        Parameters:
            - X: DataFrame or array-like, the input data.
            - replace_with: value to replace inf values with.
            
        Output:
            - X: DataFrame or array-like, the input data with inf values replaced."""
        
        return X.replace([np.inf, -np.inf], replace_with)
    

    def load_and_prepare_data(self, main_folder, file_name, replace_inf_with=1e10):
        """ Loads the data from a CSV file and prepares it for training.
        
        Parameters:
            - main_folder: the folder where the CSV file is located.
            - file_name: the name of the CSV file with the features.
            - replace_inf_with: value to replace inf values in the dataset."""
        
        # Load the CSV file
        data_file = main_folder + file_name
        data = pd.read_csv(data_file)

        if 'FileName' in data.columns:
            data = data.drop(columns=['FileName'])

        # Split the data into features (X) and target variable (y)
        self.X = data.drop(columns=['pH'])
        self.y = data['pH']

        # Handle inf values by replacing them with 1e10
        self.X = self.handle_infinity_values(self.X, replace_inf_with)

        # Impute missing values with the mean of each column
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)

        # Standardize the features
        if self.standardize:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
    

    def train_model(self, param_grid=None, scoring='balanced_accuracy', n_splits=10, report = False):
        """Performs cross-validation on the model.
        
        Parameters:
            - param_grid: dictionary with the parameters to test.
            - scoring: scoring method to use for cross-validation.
            - n_splits: number of folds for cross-validation.
            - report: whether to print the classification report.
            
        Output:
            - balanced_acc: balanced accuracy score.
            - f1: F1 score."""

        # Cross validation
        # El random_state 12, 10 folds, es el mejor para ctu-chb
        cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=12)
        cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=11)

        # Grid Search to find the best parameters
        if param_grid:
            grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring=scoring, cv=cv_inner, n_jobs=-1)
            grid_search.fit(self.X, self.y)

            # Update the model with the best parameters
            self.model.set_params(**grid_search.best_params_)

        # Perform cross-validation using the best parameters

        # Make predictions
        y_pred_cv = cross_val_predict(self.model, self.X, self.y, cv=cv_outer)

        # Get accuracy using the calculated predictions
        accuracy = accuracy_score(self.y, y_pred_cv)
        balanced_acc = balanced_accuracy_score(self.y, y_pred_cv)
        f1 = f1_score(self.y, y_pred_cv, average='binary')

        if report:
            print("\tBest Parameters:", grid_search.best_params_)
            print(f"\tAccuracy: {accuracy:.4f}")
            print(f"\t\033[35mBalanced Accuracy: {balanced_acc:.4f}\033[0m")
            print("\nClassification report:")
            print(classification_report(self.y, y_pred_cv), "\n")

        return balanced_acc, f1


