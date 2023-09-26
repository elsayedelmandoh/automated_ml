import joblib
import streamlit as st

def save_model(best_model):
    '''
    This function saves the ML pipeline as a pickle file for later use.
    '''
    # Serialize the model to a binary file using joblib
    model_filename = "best_model.pkl"
    joblib.dump(best_model, model_filename)

    # Provide a download link for the serialized model file
    st.download_button(label="Download Model", data=open(model_filename, 'rb').read(), file_name=model_filename)
