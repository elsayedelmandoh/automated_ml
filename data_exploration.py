import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import streamlit as st
from data_preparation import handle_missing_values


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