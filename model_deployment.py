import streamlit as st

def load_model():
    '''
    This function loads a previously saved pipeline.
    '''
    return st.file_uploader("Choose a model", type= ['pkl', 'h5', 'joblib'])


def deploy_model():
    """
    This function deploys the entire ML pipeline on the cloud. 
    """
    # Feature work ..
    pass 