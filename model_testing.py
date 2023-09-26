from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score

def test_model(model, models_type, x_test, y_test):
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
    