import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score


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