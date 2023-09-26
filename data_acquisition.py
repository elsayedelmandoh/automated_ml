import pandas as pd
from sqlalchemy import create_engine
import streamlit as st


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
    data_source = st.selectbox("Select data source", ['', "Upload CSV/ Txt/ Excel File", "Upload SQL Database", "Upload Pretrained Model"], index=0)
    data = None

    if data_source == "Upload CSV/ Txt/ Excel File":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx","txt"]) # png . jpg
        if uploaded_file:
            data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)


    elif data_source == "Upload SQL Database":
        db_url = st.text_input("Enter the database URL (e.g., sqlite:///database.db)") # type='password'
        if db_url:
            engine = create_engine(db_url)
            available_tables = engine.table_names()  # Get available table names
            table_name = st.selectbox("Select a table", available_tables)
            if table_name:
                query = f"SELECT * FROM {table_name}"
                data = pd.read_sql(query, engine)
                
    elif data_source == "Upload Pretrained Model":
        if st.button("Proceed to Load Model"):
            st.session_state.current_step = "load_model"
                
    # FEATURE WORK: add some error handling in case there are issues with file uploads or database connections.
    return data