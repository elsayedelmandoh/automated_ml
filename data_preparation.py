import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st


def handle_missing_values(data, missing_threshold=0.75, info=False):

    """
    This function identifies columns with missing values, 
    determines their data types (numeric or categorical), 
    and fills in the missing values with 
    mode if categorical 
    median if numeric and contian outliers
    mean if numeric and not contian outliers

    Args:
        data (DataFrame): The input dataset maybe containing with missing values.
        missing_values (float): The missing values percentage 
        info (bool): If True, the information about the missing values is available

    Returns:
        data (DataFrame): The input dataset without missing values.
    """
    st.write(data.head())
    # get columns missing values
    columns_with_missing = data.columns[data.isna().any()]
    # check number of missing values
    for column in columns_with_missing:
        num_missing = data[column].isna().sum()
        if info:
            st.write(f"Number of missing values in '{column}': {num_missing}")
        missing_percentage = num_missing / len(data)
        # drop column with high missing values
        if missing_percentage >= missing_threshold:
            if info:
                st.write(f"Dropping '{column}' due to missing values exceeding {missing_threshold * 100}%")
            data.drop(column, axis=1, inplace=True)
        else:                
            if data[column].dtype == "object":
                data[column].fillna(data[column].mode()[0], inplace=True)
                if info:
                    st.write(f"Handled Missing values to '{column}' using Mode")
            else:
                outliers= detect_outliers_iqr(data, column)
                if outliers.any():
                    data[column].fillna(data[column].median(), inplace=True)
                    if info:
                        st.write(f"Handled Missing values to '{column}' using Median")
                else:
                    data[column].fillna(data[column].mean(), inplace=True)
                    if info:
                        st.write(f"Handled Missing values to '{column}' using Mean")

    st.write(data.isna().sum())
    st.write(data.head())
    return data

def apply_dropping(data, kp='first', drop_cols=None , info=False):
    """
    This function drop duplicates rows from the dataframe

    Args:
        data (DataFrame): The input dataset maybe containing with duplicates rows.
        kp (str, optional): 'first': keep first duplicate and drop last duplicate ,
                            'last': keep last duplicates and drop first duplicates. 
                            Defaults to 'first'.
        drop_cols (list or str, optional): drop selected columns from dataframe.
        info (bool): If True, the information about the missing values is available

    Returns:
        data (DataFrame): The input dataset after dropping.
    """
    data.drop_duplicates(inplace=True, keep=kp)
    if info:
        st.write("Number of duplicates rows now:")
        st.write(data.duplicated().sum())
    
    if drop_cols:
        if info:
            st.write(f"Columns {drop_cols} dropped.")
        data.drop(drop_cols, axis=1, inplace=True)
        st.write(data.head())
    return data

def select_encoding_method(data, column):
    """
     This function is responsible for selecting the appropriate encoding method 
     (Label Encoding or One-Hot Encoding) 
     for a given categorical column based on its cardinality.

    Args:
        data (DataFrame): The DataFrame containing the data.
        column (Series): The name of the categorical column in which you want to select encoding method

    Returns:
        str: A string for selected encoding method for the column 
            ("Label Encoding" or "One-Hot Encoding").
    """
    nunique_threshold = 10  # threshold
    if data[column].nunique() > nunique_threshold:
        # If high cardinality
        return "One-Hot Encoding"
    else:
        # If low cardinality
        return "Label Encoding"

def apply_encoding(data, columns=None, info=False):
    """ This function automates the encoding process for selected categorical columns in the dataset. 
    It iterates through each categorical column, 
    select encoding method using the 'select_encoding_method' function, 
    and apply selected encoding to the column.

    Args:
        data (DataFrame): The dataset without encoding 
        columns (list or None): The columns to be encoded. If None, encode all categorical columns.
        info (bool): If True, the information about the categorical columns is available

    Returns: 
        data (DataFrame): The dataset with encoding.
        columns_encoded (list): List of columns that were encoded.
    """
    columns_encoded = []
    
    if columns is None:
        categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
        # if np.issubdtype(data[target_column].dtype, object):
        #     categorical_columns.remove(target_column)
        if not len(categorical_columns):
            st.write("Not found categorical columns for encoding.")
            st.write(data.head())
            return data, columns_encoded
    else:
        categorical_columns = columns
        
    for column in categorical_columns:
        encoding_method = select_encoding_method(data, column)

        if encoding_method == "Label Encoding": # if data is ordinal 
            if info:
                st.write(f"Apply Label Encoding to '{column}':")
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            columns_encoded.append(column)
            
        elif encoding_method == "One-Hot Encoding": # if data is nominal
            if info:
                st.write(f"Apply One-Hot Encoding to '{column}': ") 
            data = pd.get_dummies(data, columns=[column], drop_first=True, dtype=int)
            columns_encoded.extend(data.columns.tolist())
            
    st.write(data.head())
    return data, columns_encoded

def detect_outliers_iqr(data, column):
    """
    This function is used to detect outliers in a specific numerical column of a DataFrame 
    using the IQR (Interquartile Range) method.
    
    Args:
        data (DataFrame): The DataFrame containing the data.
        column (Series): The name of the numerical column in which you want to detect outliers

    Returns:
        outliers (Series): A Boolean Series indicating whether each data point in the specified column is an outlier (True) or not (False).
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers

def apply_scaling(data, target_column, columns_encoded,  columns=None, info=False):
    """
    This function automates the process of applying data scaling to selected numerical columns in a DataFrame 
    based on the presence of outliers.
    
    >> Robust Scaling
    IQR = Q3 - Q1
    Scaled_x = (x - Median) / IQR
    because Robust Scaling based on 'median'
    then it's not affected by 'outlier'
    
    >> Standard Scaling or Z-Score Standardization
    Z = (x - mean) / standard deviation
    because Standard Scaling based on 'mean' and 'Standard deviation' 
    then it's affected by 'outlier'
    good for K-Means clustering and Principal Component Analysis (PCA)
    
    >> Min-Max Scaling or Normalization
    Scaled_x = (x - min) / (max - min)
    because Min-Max Scaling based on 'min' and 'max'
    then it's affected by 'outlier'
    good for neural networks and support vector machines
    
    >> Standard scaling is less sensitive to outliers compared to min-max scaling
    
    Args:
        data: The DataFrame containing the data to be scaled.
        target_column (Series): Target column.
        columns (list, or None): List of numerical column names to scale. If None, all numerical columns will be scaled.
        info (bool): If True, the information about the numerical columns is available
        columns_encoded (list): List of columns to exclude from scaling.

    Returns:
        data: The DataFrame containing the scaled data.
    """ 
    if columns is None:
        numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()
        if np.issubdtype(data[target_column].dtype, np.number):
            numerical_columns.remove(target_column)
        if not len(numerical_columns):
            st.write("Not found numerical columns for scaling.")
            st.write(data.head())
            return data
    else:
        numerical_columns= columns
        
    # Exclude columns that were encoded from scaling
    if columns_encoded:
        numerical_columns = [col for col in numerical_columns if col not in columns_encoded]

    for column in numerical_columns:
        outliers = detect_outliers_iqr(data, column)
        if outliers.any():
            if info:
                st.write(f"Apply Robust Scaling to '{column}'")
            scaler = RobustScaler()
            data[column] = scaler.fit_transform(data[[column]])
        else:
            if info:
                st.write(f"Apply Min-Max Scaling to '{column}'")
            scaler = MinMaxScaler()  # or StandardScaler()
            data[column] = scaler.fit_transform(data[[column]])
        
    st.write(data.head())
    return data

def prepare_data(data, target_column, missing_values=True, missing_threshold= 0.75, info_missing_values=False, duplicating=True, keep_duplicates='first', drop_cols=None, info_duplicates=False, encoding=True, columns_to_encode=None, info_encoding=False, scaling=True, columns_to_scale=None, info_scaling=False):
    """
    Preprocesses the input data by 
    handling missing values, 
    duplicate values,
    encoding categorical features, 
    scaling numerical features.
    
    Parameters:
        data (DataFrame): input data
        target_column (Series): target column
        missing_values (bool): apply missing values or not
        missing_threshold (float): threshold for missing values
        duplicating (bool): apply duplicate values or not
        keep_duplicates (str): keep duplicate values
        encoding (bool): apply encoding categorical features or not
        columns_to_encode (list): columns to encode
        scaling (bool): apply scaling numeric features or not
        columns_to_scale (list): columns to scale 
        
    Returns:
        data (DataFrame): preprocessed data.
    """
    if missing_values:
        data= handle_missing_values(data, missing_threshold=missing_threshold, info=info_missing_values)
    
    if duplicating:
        data= apply_dropping(data, kp=keep_duplicates, drop_cols=drop_cols, info=info_duplicates)
    
    if encoding:
        data, columns_encoded = apply_encoding(data, columns_to_encode, info=info_encoding)
    
    if scaling:
        data= apply_scaling(data, target_column, columns_encoded, columns_to_scale, info=info_scaling)
    return data


def create_polynomial_features(data, numeric_columns, polynomial_degree=2):
    """
    Create polynomial features for selected numeric columns in a DataFrame and replace the original columns.

    Args:
        data (DataFrame): The dataset containing the data to create polynomial features.
        numeric_columns (list): List of numeric column names to create polynomial features for.
        polynomial_degree (int, optional): Degree of polynomial features. Defaults to 2.

    Returns:
        data_poly (DataFrame): The dataset containing the polynomial features, with original columns replaced.
    """
    poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    
    data_numeric = data[numeric_columns]
    data_poly_array = poly.fit_transform(data_numeric)
    poly_column_names = poly.get_feature_names_out(input_features=numeric_columns)
    data_poly = pd.DataFrame(data_poly_array, columns=poly_column_names)
    
    data = data.drop(columns=numeric_columns)
    data_poly = pd.concat([data, data_poly], axis=1)
    
    return data_poly