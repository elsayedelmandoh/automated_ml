import streamlit as st

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score , r2_score

def get_data():
    """
    Acquires data from user-defined sources, 
    either by uploading a CSV/Excel/Txt file 
    or connecting to an SQL database.
    
    Args:
        None
    
    Returns:
        pd.DataFrame or None: The loaded data as a Pandas DataFrame, otherwise None.
    """
    data_source = st.selectbox("Select data source", ['', "Upload CSV/ Txt/ Excel", "SQL Database"], index=0)
    data = None

    if data_source == "Upload CSV/ Txt/ Excel":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx","txt"])
        if uploaded_file:
            data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            st.write("Data loaded successfully!")


    elif data_source == "SQL Database":
        db_url = st.text_input("Enter the database URL (e.g., sqlite:///database.db)")
        if db_url:
            engine = create_engine(db_url)
            available_tables = engine.table_names()  # Get available table names
            table_name = st.selectbox("Select a table", available_tables)
            if table_name:
                query = f"SELECT * FROM {table_name}"
                data = pd.read_sql(query, engine)
                st.write("Data loaded successfully!")
                
    # FEATURE WORK: add some error handling in case there are issues with file uploads or database connections.
    return data
            

def visualize_target_numeric(data, target_column):
    """
    This function creates visualizations for a numeric target column, 
    including a histogram and a box plot.

    Args:
        data (DataFrame): : Input data for exploration.
        target_column (series): The numeric target column.
    
    Returns:
        None (displays visualizations in the Streamlit app)
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 20))
    
    sns.histplot(data=data, x=target_column, bins=20, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of distribution '{target_column}'")
    
    sns.boxplot(y=target_column, data=data, ax=axes[1])
    axes[1].set_title(f"Box Plot of '{target_column}'")
    
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)


def visualize_other_numeric(data, numeric_column, target_column):
    """This function creates visualizations for the relationship between a numeric column and a target column, 
    including a scatter plot and a line plot.

    Args:
        data (DataFrame): Input data for exploration.
        numeric_column (Series):  Numeric column.
        target_column (Series): Target column.
    
    Returns:
        None (displays visualizations in the Streamlit app)
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 20))
    
    sns.scatterplot(x=numeric_column, y=target_column, data=data, ax=axes[0])
    axes[0].set_xlabel(numeric_column)
    axes[0].set_ylabel(target_column)
    axes[0].set_title(f"Scatter Plot of {numeric_column} vs {target_column}")
    
    sns.lineplot(x=numeric_column, y=target_column, data=data, ax=axes[1])
    axes[1].set_xlabel(numeric_column)
    axes[1].set_ylabel(target_column)
    axes[1].set_title(f"Line Plot of {numeric_column} vs {target_column}")
    
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    

def visualize_target_categorical(data, target_column):
    """This function creates visualizations for a categorical target column, 
    including a bar plot, a pie chart, and a heatmap.

    Args:
        data (DataFrame): Input data for exploration.
        target_column (Series): Target column.
    
    Returns:
        None (displays visualizations in the Streamlit app)
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 20))
    
    ax= sns.countplot(y=target_column, data=data, order=data[target_column].value_counts().index, ax=axes[0])
    axes[0].set_title(f"Bar Plot of '{target_column}'")
    axes[0].set_ylabel('')
    for p in ax.patches:
        ax.annotate(
            f'{p.get_width()}', 
            (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.), 
            ha='center', 
            va='center', 
            rotation=270,
            xytext=(5, 0),
            textcoords='offset points'
                    )
    
    target_counts = data[target_column].value_counts()
    axes[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140, rotatelabels=True)
    axes[1].axis('equal')
    axes[1].set_title(f"Pie Chart of '{target_column}'")

    target_cross_tab = pd.crosstab(index= data[target_column], columns="count")
    sns.heatmap(target_cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=axes[2])
    axes[2].set_title(f"Heatmap of '{target_column}'")
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    

def visualize_other_categorical(data, categorical_column, target_column):
    """This function creates visualizations for the relationship between a categorical column and a target column, 
    including a count plot and a histogram

    Args:
        data (DataFrame): Input data for exploration.
        categorical_column (Series):  Categorical column.
        target_column (Series): Target column.
    
    Returns:
        None (displays visualizations in the Streamlit app)
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 15))
    
    sns.countplot(x=categorical_column, hue=target_column, data=data, ax=axes[0])
    axes[0].set_title(f"Count Plot of '{categorical_column}' vs '{target_column}'")
    
    sns.histplot(data=data, x=categorical_column, hue=target_column, multiple='stack', ax=axes[1])
    axes[1].set_title(f"Histogram of distribution '{categorical_column}' vs '{target_column}'")
    
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    
def visualize_categorical_numeric(data, categorical_column, numeric_column):
    """This function creates visualizations for the relationship between a categorical column and a numeric column, 
    including a bar plot, a box plot, and a violin plot.

    Args:
        data (DataFrame): Input data for exploration.
        categorical_column (Series):  Categorical column.
        numeric_column (Series):  Numeric column.
        
    Returns:
        None (displays visualizations in the Streamlit app)
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

    sns.barplot(x=categorical_column, y=numeric_column, data=data, ax=axes[0])
    axes[0].set_xlabel(categorical_column)
    axes[0].set_ylabel(numeric_column)
    axes[0].set_title(f"Bar Plot of {numeric_column} vs {categorical_column}")
    
    sns.boxplot(x=categorical_column, y=numeric_column, data=data, ax=axes[1])
    axes[1].set_xlabel(categorical_column)
    axes[1].set_ylabel(numeric_column)
    axes[1].set_title(f"Box Plot of {numeric_column} vs {categorical_column}")
    
    sns.violinplot(x=categorical_column, y=numeric_column, data=data, ax=axes[2])
    axes[2].set_xlabel(categorical_column)
    axes[2].set_ylabel(numeric_column)
    axes[2].set_title(f"Violin Plot of {numeric_column} vs {categorical_column}")
    
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

def visualize_time_series(data, time_series_column, target_column):
    """This function creates a time series plot to visualize the relationship 
    between a time series column and a target column.

    Args:
        data (DataFrame): The dataset containing the time series and target columns.
        time_series_column (Series): Series column
        target_column (Series): Target column

    Returns:
        None (displays visualizations in the Streamlit app)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(x=time_series_column, y=target_column, data=data, ax=ax)
    ax.set_xlabel(time_series_column)
    ax.set_ylabel(target_column)
    ax.set_title(f"Line Plot of {target_column} over {time_series_column}")
    
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)


def explore_data(data, target_column, plots=True, info=True):
    """
    This function generates data exploration and analysis visualizations for a given dataset. 
    It displays the head and tail of the data, 
    the number of rows and columns, 
    the data types of features, 
    allows to select a target column for analysis. 
    options to select the best number of numerical and categorical features to visualize in relation to the target column.

    Args:
        data (pd.DataFrame): input data for exploration.
    
    Returns:
        target_column (Series): target column
    """
    if info:
        num_head_rows = st.number_input("Number of rows to display (head)", min_value=1, max_value=len(data), value=5)
        st.subheader("Head of the data")
        st.write(data.head(num_head_rows))

        num_tail_rows = st.number_input("Number of rows to display (tail)", min_value=1, max_value=len(data), value=5)
        st.subheader("Tail of the data")
        
        st.write(data.tail(num_tail_rows))
        
        st.subheader("Number of rows and columns")
        st.write(f"Your data has '{data.shape[0]}' rows and '{data.shape[1]}' columns.")
        
        st.subheader("Datatypes of features")
        st.write(data.dtypes) 
        target_column_type = data[target_column].dtype
    
    if plots:
        if np.issubdtype(target_column_type, np.number):
            st.subheader("Numeric Target Column Analysis")
            st.write(f"Summary Statistics for {target_column}:")
            st.write(data[target_column].describe())
            
            st.subheader(f"Visualizations for '{target_column}'")
            visualize_target_numeric(data, target_column)

            if len(data.select_dtypes(include=[np.number]).columns) > 1:
                st.subheader("Select of numerical features")
                num_features = st.number_input(f"Number of features to visulize with {target_column}: ", min_value=1, max_value=len(data.select_dtypes(include=[np.number]).columns) -1, value=1)
                best_num_features = best_numeric_features(data, target_column, num_features)
            else: 
                st.write("No other numeric columns to visualize relationships with.")
                
            if len(best_num_features) > 0:
                st.subheader("Visualize Relationships with Other Numeric Columns")
                for numeric_col in best_num_features:
                    visualize_other_numeric(data, numeric_col, target_column)
            else:
                st.write("No other numeric columns to visualize relationships with.")

            if len(best_cat_features) > 0:
                st.subheader("Visualize Relationships with Other Categorical Columns")
                for categorical_col in best_cat_features:
                    visualize_categorical_numeric(data, categorical_col, target_column)
            else:
                st.write("No other categorical columns to visualize relationships with.")

            time_series_columns = data.select_dtypes(include=[np.datetime64]).columns.tolist()
            if len(time_series_columns) > 0:
                st.subheader("Visualize Relationships with Other time series Columns")
                for time_series_col in time_series_columns:
                    visualize_time_series(data, time_series_col, target_column)
            else:
                st.write("No other time series columns to visualize relationships with.")

            
        elif np.issubdtype(target_column_type, object):
            st.subheader("Categorical Target Column Analysis")
            st.write(f"Summary Statistics for {target_column}:")
            st.write(data[target_column].describe())
            st.write(f"Value Counts for {target_column}:")
            st.write(data[target_column].value_counts())
            
            st.subheader(f"Visualizations for '{target_column}'")
            visualize_target_categorical(data, target_column)

            if len(data.select_dtypes(include=[object]).columns) > 1:
                st.subheader("Select the best number of categorical features")
                num_features = st.number_input(f"Number of features to visulize with {target_column}: ", min_value=1, max_value=len(data.select_dtypes(include=[object]).columns) -1, value=1)
                best_cat_features = best_categorical_features(data, target_column, num_features)
            else:
                st.write("No other numeric columns to visualize relationships with.")

            st.subheader("Visualize Relationship with Other Categorical Columns")
            if len(best_cat_features) > 0:
                st.subheader("Visualize Relationships with Other Categorical Columns")
                for categorical_col in best_cat_features:
                    visualize_other_categorical(data, categorical_col, target_column)
            else:
                st.write("No other categorical columns to visualize relationships with.")

            if len(best_num_features) > 0:
                st.subheader("Visualize Relationships with Other Numeric Columns")
                for numeric_col in best_num_features:
                    visualize_categorical_numeric(data, numeric_col, target_column)
            else:
                st.write("No other numeric columns to visualize relationships with.")

            time_series_columns = data.select_dtypes(include=[np.datetime64]).columns.tolist()
            if len(time_series_columns) > 0:
                st.subheader("Visualize Relationships with Other time series Columns")
                for time_series_col in time_series_columns:
                    visualize_time_series(data, time_series_col, target_column)
            else:
                st.write("No other time series columns to visualize relationships with.")

        else:
            st.write("Visualization not supported for this column type.")
        

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
    columns_with_missing = data.columns[data.isna().any()]
    for column in columns_with_missing:
        num_missing = data[column].isna().sum()
        if info:
            st.write(f"Number of missing values in '{column}': {num_missing}")
        missing_percentage = num_missing / len(data)

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
        st.write("Numer of duplicates rows now:")
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

def apply_encoding(data, target_column, columns=None, info=False):
    """ This function automates the encoding process for selected categorical columns in the dataset. 
    It iterates through each categorical column, 
    select encoding method using the 'select_encoding_method' function, 
    and apply selected encoding to the column.

    Args:
        data (DataFrame): The dataset without encoding 
        target_column (Series): Target column 
        columns (list or None): The columns to be encoded. If None, encode all categorical columns.
        info (bool): If True, the information about the categorical columns is available

    Returns: 
        data (DataFrame): The dataset with encoding 
    """
    if columns is None:
        categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
        if np.issubdtype(data[target_column].dtype, object):
            categorical_columns.remove(target_column)
        if not len(categorical_columns):
            st.write("Not found categorical columns for encoding.")
            st.write(data.head())
            return data
    else:
        categorical_columns = columns
        
    for column in categorical_columns:
        encoding_method = select_encoding_method(data, column)

        if encoding_method == "Label Encoding": # if data is ordinal 
            if info:
                st.write(f"Apply Label Encoding to '{column}':")
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])

        elif encoding_method == "One-Hot Encoding": # if data is nominal
            if info:
                st.write(f"Apply One-Hot Encoding to '{column}': ") 
            data = pd.get_dummies(data, columns=[column], drop_first=True, dtype=int)

    st.write(data.head())
    return data

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

def apply_scaling(data, target_column, columns=None, info=False):
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
        st.subheader("Handling missing values:")
        data= handle_missing_values(data, missing_threshold=missing_threshold, info=info_missing_values)
    
    if duplicating:
        st.subheader("Drop duplicates:")
        data= apply_dropping(data, kp=keep_duplicates, drop_cols=drop_cols, info=info_duplicates)
    
    if encoding:
        st.subheader("Encoding Categorical Features:")
        data= apply_encoding(data, target_column, columns_to_encode, info=info_encoding)
    
    if scaling:
        st.subheader("Scaling Numerical Features")
        data= apply_scaling(data, target_column, columns_to_scale, info=info_scaling)
    return data


def best_numeric_features(data, target_column, num_features):
    """This function selects the best numeric features for analysis 
    based on their F-scores in relation to a target column.

    Args:
        data (DataFrame): Input data for exploration.
        target_column (Series): Target column.
        num_features (int): The number of best numeric features to select.

    Returns:
        selected_numeric_features (list): List of the selected best numeric features.
    """
    st.write("Handling missing values")
    data= handle_missing_values(data)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    numeric_features = X.select_dtypes(include=[np.number])
    f_selector = SelectKBest(score_func=f_classif, 
                             k='all')
    f_selector.fit(numeric_features, y)
    numeric_scores = pd.DataFrame({'Feature': numeric_features.columns, 
                                   'F-Score': f_selector.scores_}).sort_values(by='F-Score', 
                                                                               ascending=False)
    
    selected_numeric_features = numeric_scores.nlargest(num_features, 
                                                        'F-Score')['Feature'].tolist()
    
    return selected_numeric_features


def best_categorical_features(data, target_column, num_features):
    """This function selects the best categorical features for analysis 
    based on their chi-squared scores in relation to a target column.

    Args:
        data (DataFrame): Input data for exploration.
        target_column (Series): Target column.
        num_features (int): The number of best categorical features to select.

    Returns:
        selected_categorical_features (list): List of the selected best categorical features.
    """
    st.write("Handling missing values")
    data= handle_missing_values(data)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    categorical_features = X.select_dtypes(include=[object])
    for column in categorical_features:
        encoder= LabelEncoder() # i don't use "get_dummies" because it create more 'new columns' can't make visualize with them  
        categorical_features[column] = encoder.fit_transform(categorical_features[column])

    f_selector = SelectKBest(score_func=chi2, 
                             k='all')
    f_selector.fit(categorical_features, y)
    categorical_scores = pd.DataFrame({'Feature': categorical_features.columns, 
                                'Chi-Squared Score': f_selector.scores_}).sort_values(by='Chi-Squared Score', 
                                                                            ascending=False)
    
    selected_categorical_features = categorical_scores.nlargest(num_features, 
                                                         'Chi-Squared Score')['Feature'].tolist()

    return selected_categorical_features


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


def train_val_test_split(data, target_column):
    """
     This function is responsible for splitting the input dataset into three subsets: 
     training with 75%, 
     validation with 15%,
     testing with 10%. 
     It takes the dataset and the name of the target column as input

    Args:
        data (DataFrame): The DataFrame containing the dataset.
        target_column (Series): Target column to predict.

    Returns:
        x_train: The features data for the training set.
        y_train: The target data for the training set.
        x_val: The features data for the validation set.
        y_val: The target data for the validation set.
        x_test: The features data for the testing set.
        y_test: The target data for the testing set.
    """
    x= data.drop(target_column, axis=1)
    y= data[target_column]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test


def select_models(data, target_column):
    """
    Automatically select suitable models for regression or classification tasks based on the datatype of the target variable
    and create instances of the selected models.

    Args:
        data (DataFrame): Input data containing both features and the target variable.
        target_column (str): Name of the target variable column.

    Returns:
        models_selected (dict): A dictionary containing instances of the selected models.
        models_type (str): The type of the selected models (Regression or classification)
    """
    target_dtype = data[target_column].dtype

    models_regression = {
        # 'LinearRegression': LinearRegression(), # make error on large datasets that make it difficult to fit 
        'RidgeRegression': Ridge(alpha=0.5, # default alpha = 1.0
                                 solver='auto'
                                 ), 
        'SupportVectorRegressor': SVR(kernel='rbf',
                                      degree=3
                                      ),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=7
                                                   ),
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=4
                                                       ),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=10,
                                                       max_depth=5
                                                       ),
        'XBoostingRegressor': XGBRegressor(eta= 0.1, # default= 0.3
                                             gamma= 5, # default= 0
                                             max_depth= 5
                                             )
    }

    models_classification = {
        'LogisticRegression': LogisticRegression(C= 1, # default = 1.0
                                                max_iter= 10, # default= 100
                                                random_state=42
                                                 ),
        'SupportVectorClassifier': SVC(kernel='rbf',
                                       degree=3
                                       ),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=7
                                                     ),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=4,
                                                         ),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=10,
                                                         max_depth=5
                                                         ),
        'XBoostingClassifier': XGBClassifier(eta= 0.1, # default= 0.3
                                             gamma= 5, # default= 0
                                             max_depth= 5
                                             )
    }
    
    return (models_classification, 'classification') if pd.api.types.is_object_dtype(target_dtype) or len(data[target_column].unique()) < 10 or pd.api.types.is_bool_dtype(target_dtype) else (models_regression, 'regression')


def create_models(selected_models, selected_model_names):
    """ Create instances of selected machine learning models.

    Args:
        selected_models (dict): A dictionary of selected machine learning models, typically generated by the 'select_models' function.
        selected_model_names (list): A list of model names to create instances for.


    Returns:
        models (dict): A dictionary containing instances of the selected machine learning models.
    """
    models = {}
    for model_name in selected_model_names:
        if model_name in selected_models:
            model = selected_models[model_name]
            models[model_name] = model
    return models


def compare_models(models, models_type, x_train, y_train, x_val, y_val):
    """
    Compare the performance of different machine learning models for either classification or regression tasks and
    return the best model along with its evaluation scores on training and validation data,
    In the case of regression tasks, Evaluation scores are  MAE, MSE, R-square.
    In the case of classification tasks, Evaluation scores are accuracy, recall, precision.
    
    Args:
        models (dict): A dictionary containing instances of selected machine learning models.
        models_type (str): The type of task, either 'classification' or 'regression'.
        x_train (DataFrame): Features data for the training set.
        y_train (Series): Target data for the training set.
        x_val (DataFrame): Features data for the validation set.
        y_val (Series): Target data for the validation set.
        
    Returns:
        tuple: A tuple containing the best model, 
                the name of the best model,
                evaluation scores on training data, 
                evaluation scores on validation data.
    """
    best_model_name = None
    best_model = None
    report_train = {}
    report_val = {}

    if models_type == 'classification':
        best_accuracy_val = 0
        
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_train_preds = model.predict(x_train)
            y_val_preds = model.predict(x_val)

            accuracy_train = accuracy_score(y_train, y_train_preds)
            accuracy_val = accuracy_score(y_val, y_val_preds)

            recall_train = recall_score(y_train, y_train_preds)
            recall_val = recall_score(y_val, y_val_preds)

            precision_train = precision_score(y_train, y_train_preds)
            precision_val = precision_score(y_val, y_val_preds)

            if accuracy_val > best_accuracy_val:
                best_accuracy_val = accuracy_val
                best_model_name = model_name
                best_model = model
                

            report_train[model_name] = {
                'Accuracy': accuracy_train,
                'Recall': recall_train,
                'Precision': precision_train,
            }

            report_val[model_name] = {
                'Accuracy': accuracy_val,
                'Recall': recall_val,
                'Precision': precision_val,
            }
        return best_model, best_model_name, report_train, report_val
    
    elif models_type == 'regression':
        best_mae_val = float('inf')
        
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_train_preds = model.predict(x_train)
            y_val_preds = model.predict(x_val)

            mae_train = mean_absolute_error(y_train, y_train_preds)
            mae_val = mean_absolute_error(y_val, y_val_preds)

            mse_train = mean_squared_error(y_train, y_train_preds)
            mse_val = mean_squared_error(y_val, y_val_preds)

            r2_train = r2_score(y_train, y_train_preds)
            r2_val = r2_score(y_val, y_val_preds)

            if mae_val < best_mae_val:
                best_mae_val = mae_val
                best_model_name = model_name
                best_model = model

            report_train[model_name] = {
                'MAE': mae_train,
                'MSE': mse_train,
                'R-squared': r2_train,
            }

            report_val[model_name] = {
                'MAE': mae_val,
                'MSE': mse_val,
                'R-squared': r2_val,
            }
        return best_model, best_model_name, report_train, report_val
    
    else:
        raise ValueError(f"Models type '{model_name}' not found")

def create_report(*dicts, prefix=None):
    """
    Combine multiple dictionaries into a single DataFrame and add a prefix to columns.

    Args:
        *dicts: Any number of dictionaries to be combined.
        prefix (str or list): Prefix or list of prefixes to be added to column names (optional).

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    final_df = pd.DataFrame()

    if isinstance(prefix, str):
        prefix = [prefix] * len(dicts)

    for i, (dic, pre) in enumerate(zip(dicts, prefix)):
        df = pd.DataFrame.from_dict(dic, orient='index') 
        df = df.add_prefix(f'{pre}_')
        final_df = pd.concat([final_df, df], axis=1)
        
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Model'}, inplace=True)

    return final_df

def evaluate_model(model, models_type, x_test, y_test):
    """ Evaluate the performance of a machine learning model on Testing data and return the evaluation scores.
        In the case of regression tasks, Evaluation scores are  MAE, MSE, R-square.
        In the case of classification tasks, Evaluation scores are accuracy, recall, precision.
        
    Args:
        model: The trained machine learning model to evaluate.
        models_type (str): The type of task, either 'classification' or 'regression'.
        x_val (DataFrame): Features data for the validation set.
        y_val (Series): Target data for the validation set.

    Returns:
        dict: A dictionary containing evaluation scores.
    """
    y_val_preds = model.predict(x_test)
    
    if models_type == 'classification':
        accuracy_val = accuracy_score(y_test, y_val_preds)
        recall_val = recall_score(y_test, y_val_preds)
        precision_val = precision_score(y_test, y_val_preds)
        
        report_test = {
            'Accuracy': accuracy_val,
            'Recall': recall_val,
            'Precision': precision_val,
        }
        return report_test
    else:
        mae_val = mean_absolute_error(y_test, y_val_preds)
        mse_val = mean_squared_error(y_test, y_val_preds)
        r2_val = r2_score(y_test, y_val_preds)

        report_test= {
        'MAE': mae_val,
        'MSE': mse_val,
        'R-squared': r2_val,
        }
        return report_test

def plot_model_scores(report_train, report_val, report_test, best_model):
    """
    Plot the scores (accuracy, recall, precision, mean_absolute_error, mean_squared_error, r2_score) of the best model on training, validation and testing data.

    Args:
        report_train (dict): A dictionary containing scores on the training data for different models.
        report_val (dict): A dictionary containing scores on the validation data for different models.
        report_test (dict): A dictionary containing scores on the test data for different models.
        best_model (str): The name of the best model.

    Returns:
        None (displays visualizations in the Streamlit app)
    """
    train_scores = report_train[best_model]
    val_scores = report_val[best_model]
    test_scores = report_test

    metrics = list(train_scores.keys())

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 6 * len(metrics)))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(["Training", "Validation", "Testing"], [train_scores[metric], val_scores[metric], test_scores[metric]], color=['blue', 'green', 'red'])
        ax.set_title(f"{metric} for {best_model}")
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    
def predict_model(model, x_test):
    '''
    This function generates the label using a trained model.  
    When unseen data is not passed, it predicts the label and score on the holdout set.
    '''
    pass


def save_model(model, filename):
    '''
    This function saves the ML pipeline as a pickle file for later use.
    '''
    # feature work 
    pass

def load_model(model_path):
    '''
    This function loads a previously saved pipeline.
    '''
    # Feature work ..
    pass

def deploy_model():
    """
    This function deploys the entire ML pipeline on the cloud. 
    """
    # Feature work ..
    pass 

def main():
    """    
    Main function for the Automated Machine Learning model.
    This function manages the entire workflow of the automated machine learning model. It steps through
    data acquisition,
    data exploration, 
    data preparation, 
    model creation, 
    model comparison, 
    model testing, 
    model visualization.
    
    Returns:
        None
    """
    st.title("Automated Machine Learning model")
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'data_acquisition'
        st.session_state.data = None
        
        st.session_state.target_column = None
        st.session_state.eda = None
        st.session_state.info = None
        
        st.session_state.option = None
        st.session_state.mv = None
        st.session_state.mt = None
        st.session_state.info_mv = None
        st.session_state.dp = None
        st.session_state.kp = None
        st.session_state.drp_cls = None
        st.session_state.info_dp = None
        st.session_state.en = None
        st.session_state.categorical_columns = None
        st.session_state.en_cols = None
        st.session_state.info_en = None
        st.session_state.sc = None
        st.session_state.numerical_columns = None
        st.session_state.sc_cols = None
        st.session_state.info_sc = None
        
        st.session_state.selected_models = None
        st.session_state.models_type = None
        
        st.session_state.selected_model_names = None
        st.session_state.models = None
        
        st.session_state.x_train = None
        st.session_state.y_train = None
        
        st.session_state.x_val = None
        st.session_state.y_val = None
        
        st.session_state.x_test = None
        st.session_state.y_test = None
        
        st.session_state.best_model = None
        st.session_state.best_model_name = None
        st.session_state.report_train = None
        st.session_state.report_val = None
        st.session_state.report_train_Val_df = None
        
        st.session_state.report_test = None
        st.session_state.report_test_df = None
        st.session_state.tvt = None

    if st.session_state.current_step == "data_acquisition":
        st.header("Data Acquisition")
        data= get_data()

        if data is not None:
            st.session_state.data = data
            
            if st.button("Proceed to Data Exploration"):
                st.session_state.current_step = "data_exploration"
                
    elif st.session_state.current_step == "data_exploration":
        st.header("Data Exploration and Analysis")
        data = st.session_state.data

        st.subheader("Select a Target Column")
        target_column = st.selectbox("Select the target column", data.columns.tolist())
        st.session_state.target_column = target_column
        
        st.subheader("Display EDA visualizations?")
        st.write("Note: STILL FOUND SOME PROBLEMS IN STEP visualizations I WILL HANDLE THEM")
        eda = st.checkbox("Do you want to display EDA visualizations?")
        st.session_state.eda = eda
        
        st.subheader("Display EDA Analysis?")
        info = st.checkbox("Do you want to display EDA Analysis?")
        st.session_state.info = info
        explore_data(data, target_column, plots=eda, info=info)

        if st.button("Proceed to Data Preparation"):
            st.session_state.current_step = "data_preparation"
        
    elif st.session_state.current_step == "data_preparation":
        st.header("Data Preparation")
        data = st.session_state.data  
        target_column = st.session_state.target_column 

        st.write("NOTE: Apply all preprocessing functions 'not' show optional parameters for information processing")
        st.write("NOTE: Apply specific preprocessing function show optional parameters for information processing")
        option = st.selectbox("Select Preprocessing Option", ['', "Apply all preprocessing functions", "Apply specific preprocessing function"], index=0)
        st.session_state.option = option
        
        if st.session_state.option == "Apply all preprocessing functions":
            data = prepare_data(data, target_column)
            st.session_state.data = data

        elif st.session_state.option == "Apply specific preprocessing function":
            st.subheader("Handling missing values:")
            mv =  st.checkbox("Do you want to handling missing values")
            st.session_state.mv = mv
            if st.session_state.mv:
                mt = st.number_input("If missing values > this percentage, drop column", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
                st.session_state.mt = mt
                info_mv = st.checkbox("Do you want to display info about how to handled missing values?")
                st.session_state.info_mv = info_mv
                if st.button("Apply") and mt != 0.0:
                    data = handle_missing_values(data, missing_threshold=mt, info=info_mv)
                    st.session_state.data = data
                
            st.subheader("Apply dropping:")
            dp = st.checkbox("Do you want to apply dropping")
            st.session_state.dp = dp
            if st.session_state.dp:
                kp = st.radio("Do you want to keep duplicates rows 'first' or 'last'?", [False, 'first', 'last'], index=0)
                st.session_state.kp = kp
                drp_cls = st.multiselect("Select columns for dropping", list(data.drop(target_column, axis=1).columns))
                st.session_state.drp_cls = drp_cls
                info_dp = st.checkbox("Do you want to display numer of duplicates rows?")
                st.session_state.info_dp = info_dp
                if st.button("Apply dropping") and kp:
                    data = apply_dropping(data, kp=kp, drop_cols=drp_cls, info=info_dp)
                    st.session_state.data = data
            
            st.subheader("Encoding Categorical Features:")
            en = st.checkbox("Do you want to encoding categorical features")
            st.session_state.en = en 
            if st.session_state.en:
                categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
                st.session_state.categorical_columns = categorical_columns
                
                if np.issubdtype(data[target_column].dtype, object):
                    categorical_columns.remove(target_column)
                    st.session_state.categorical_columns = categorical_columns
                   
                if len(categorical_columns):
                    if st.checkbox("Select All categorical Columns"):
                        en_cols = categorical_columns
                    else:
                        en_cols = st.multiselect("Select Categorical Columns", categorical_columns)
                    st.session_state.en_cols = en_cols
                    
                    info_en = st.checkbox("Do you want to display info about how to handled categorical features?")
                    st.session_state.info_en = info_en
                    if st.button("Apply Encoding") and en_cols:
                        data = apply_encoding(data, target_column, columns=en_cols, info=info_en)
                        st.session_state.data = data
                else:
                    st.write("Not found categorical columns for encoding.")

            st.subheader("Scaling Numerical Features")
            sc = st.checkbox("Do you want to scaling numerical features")
            st.session_state.sc = sc
            if st.session_state.sc:
                numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()
                st.session_state.numerical_columns = numerical_columns
                
                if np.issubdtype(data[target_column].dtype, np.number):
                    numerical_columns.remove(target_column)
                    st.session_state.numerical_columns = numerical_columns

                if len(numerical_columns):
                    if st.checkbox("Select All Numerical Columns"):
                        sc_cols = numerical_columns
                    else:
                        sc_cols = st.multiselect("Select Numerical Columns", numerical_columns)
                    st.session_state.sc_cols = sc_cols
                    
                    info_sc = st.checkbox("Do you want to display info about how to handled numerical features?")
                    st.session_state.info_sc = info_sc
                    if st.button("Apply Scaling") and sc_cols:
                        data = apply_scaling(data, target_column, columns=sc_cols, info=info_sc)
                        st.session_state.data = data
                else:    
                    st.write("Not found numerical columns for scaling.")

        if st.button("create models and comparison"):
            st.session_state.current_step = "create_models_comparison"
    elif st.session_state.current_step == "create_models_comparison":
        st.header("Create Models")
        data = st.session_state.data  
        target_column = st.session_state.target_column 

        selected_models, models_type = select_models(data, target_column)
        st.session_state.selected_models = selected_models
        st.session_state.models_type = models_type
        
        if len(selected_models):
            if st.checkbox("Select All models"):
                selected_model_names = list(selected_models.keys())
            else:
                selected_model_names = st.multiselect("Select Models to Use", list(selected_models.keys()))
            st.session_state.selected_model_names = selected_model_names

            if st.button("Create the models") and selected_model_names:
                models = create_models(selected_models, selected_model_names)
                st.session_state.models = models
                
                st.header("Compare Models")
                x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, target_column)
                st.session_state.x_train = x_train
                st.session_state.y_train = y_train
                
                st.session_state.x_val = x_val
                st.session_state.y_val = y_val
                
                st.session_state.x_test = x_test
                st.session_state.y_test = y_test
                
                best_model, best_model_name, report_train, report_val = compare_models(models, models_type, x_train, y_train, x_val, y_val)
                st.session_state.best_model = best_model
                st.session_state.best_model_name = best_model_name
                st.write(f"Best model: {best_model_name}")
                
                st.session_state.report_train = report_train
                st.session_state.report_val = report_val
                st.write("Training and Validation Report:")
                report_train_Val_df = create_report(report_train, report_val, prefix=['Train', 'Val'])
                st.session_state.report_train_Val_df = report_train_Val_df
                st.write(report_train_Val_df)
        else:
            st.write("Not models selected for creation.")
        
        if st.button("Proceed to Test Model"):
            st.session_state.current_step = "model_testing"
        
    elif st.session_state.current_step == "model_testing":
        st.header("Model Testing")
        best_model = st.session_state.best_model
        best_model_name = st.session_state.best_model_name
        models_type = st.session_state.models_type
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test
        
        report_test = evaluate_model(best_model, models_type, x_test, y_test)
        st.session_state.report_test = report_test
        st.write(f"Model: {best_model_name}")
        st.write("Testing Report:")
        report_test_df = create_report(report_test, prefix='Test')
        st.session_state.report_test_df = report_test_df
        st.write(report_test_df)

        tvt = st.checkbox("Do you want to display Report Training & Validation again to make comparsion?")
        st.session_state.tvt = tvt
        if tvt:
            report_train_Val_df = st.session_state.report_train_Val_df
            st.write(report_train_Val_df)
        
        if st.button("Proceed to Plot Model"):
            st.session_state.current_step = "plot_model"
            
    elif st.session_state.current_step == "plot_model":
        st.header("Plot Model")
        best_model_name = st.session_state.best_model_name
        st.write(f"The model: {best_model_name}")
        
        report_train = st.session_state.report_train
        report_val = st.session_state.report_val
        report_train_Val_df = st.session_state.report_train_Val_df
        st.write("Training and Validation Report:")
        st.write(report_train_Val_df)
        
        report_test = st.session_state.report_test
        report_test_df = st.session_state.report_test_df
        st.write("Testing Report:")
        st.write(report_test_df)
        plot_model_scores(report_train, report_val, report_test, best_model_name)
        
        st.write("Project still need more features which will be as feature work: ")
        st.write("Author: Elsayed Elmandoh")
        st.write("Machine Learning Engineer")
        
# I hope get more feedback (in mistaks) to improve performance more
# I still need to handel some plots
# I need search more to know what's the best way to display plots automated
# Give me steps make them or something for searching


if __name__ == "__main__":
    main()