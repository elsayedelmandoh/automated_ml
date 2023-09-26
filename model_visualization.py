import streamlit as st
import matplotlib.pyplot as plt


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