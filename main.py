import streamlit as st
import numpy as np

from data_acquisition import get_data
from data_exploration import explore_data
from data_preparation import handle_missing_values, apply_dropping, apply_encoding, apply_scaling, prepare_data
from model_creation import select_models, create_models
from model_comparison import train_val_test_split, compare_models, create_report 
from model_testing import test_model
from model_visualization import plot_model_scores
from model_saving import save_model 
from model_deployment import load_model, predict_model, deploy_model

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
    model visualization,
    model saving.
    
    Returns:
        None
    """
    st.title("Automated Machine Learning model")
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'data_acquisition'
        st.session_state.data = None
        # data exploration
        st.session_state.target_column = None
        st.session_state.eda = None
        st.session_state.info = None
        # data preparation
        st.session_state.option = None
        # missing values
        st.session_state.mv = None
        st.session_state.mt = None
        st.session_state.info_mv = None
        # dropping
        st.session_state.dp = None
        st.session_state.kp = None
        st.session_state.drp_cls = None
        st.session_state.info_dp = None
        # encoding
        st.session_state.en = None
        st.session_state.categorical_columns = None
        st.session_state.en_cols = None
        st.session_state.info_en = None
        st.session_state.columns_encoded = None
        # scaling
        st.session_state.sc = None
        st.session_state.numerical_columns = None
        st.session_state.sc_cols = None
        st.session_state.info_sc = None
        # model creation
        st.session_state.selected_models = None
        st.session_state.models_type = None
        
        st.session_state.selected_model_names = None
        st.session_state.models = None
        # model comparison
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
        # model testing
        st.session_state.report_test = None
        st.session_state.report_test_df = None
        st.session_state.tvt = None
        # model loading
        st.session_state.loaded_model = None
        

    if st.session_state.current_step == "data_acquisition":
        st.header("Data Acquisition")
        data= get_data()

        if data is not None:
            st.session_state.data = data
            st.write("Data loaded successfully!")
            
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

        st.write("NOTE: Apply 'all' preprocessing functions 'not show' optional parameters for information processing")
        st.write("NOTE: Apply 'specific' preprocessing function 'show' optional parameters for information processing")
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
                    st.subheader("Handling missing values:")
                    data = handle_missing_values(data, missing_threshold=mt, info=info_mv)
                    st.session_state.data = data
                
            st.subheader("Apply dropping:")
            dp = st.checkbox("Do you want to apply dropping")
            st.session_state.dp = dp
            if st.session_state.dp:
                kp = st.radio("Do you want to keep duplicates rows 'first' or 'last'?", [False, 'first', 'last'], index=0)
                st.session_state.kp = kp
                st.write(data.head())
                drp_cls = st.multiselect("Select columns for dropping", list(data.drop(target_column, axis=1).columns))
                st.session_state.drp_cls = drp_cls
                info_dp = st.checkbox("Do you want to display number of duplicates rows?")
                st.session_state.info_dp = info_dp
                if st.button("Apply dropping") and kp:
                    st.subheader("Apply Dropping:")
                    data = apply_dropping(data, kp=kp, drop_cols=drp_cls, info=info_dp)
                    st.session_state.data = data
            
            st.subheader("Encoding Categorical Features:")
            en = st.checkbox("Do you want to encoding categorical features")
            st.session_state.en = en 
            if st.session_state.en:
                categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
                st.session_state.categorical_columns = categorical_columns
                
                # if np.issubdtype(data[target_column].dtype, object):
                #     categorical_columns.remove(target_column)
                #     st.session_state.categorical_columns = categorical_columns
                   
                if len(categorical_columns):
                    if st.checkbox("Select All categorical Columns"):
                        en_cols = categorical_columns
                    else:
                        en_cols = st.multiselect("Select Categorical Columns", categorical_columns)
                    st.session_state.en_cols = en_cols
                    
                    info_en = st.checkbox("Do you want to display info about how to handled categorical features?")
                    st.session_state.info_en = info_en
                    if st.button("Apply Encoding") and en_cols:
                        st.subheader("Encoding Categorical Features:")
                        data, columns_encoded = apply_encoding(data, columns=en_cols, info=info_en)
                        st.session_state.data = data
                        st.session_state.columns_encoded = columns_encoded
                else:
                    st.write("Not found categorical columns for encoding.")

            st.subheader("Scaling Numerical Features")
            sc = st.checkbox("Do you want to scaling numerical features")
            st.session_state.sc = sc
            if st.session_state.sc:
                numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()
                columns_encoded = st.session_state.columns_encoded
                if columns_encoded:
                    numerical_columns = [col for col in numerical_columns if col not in columns_encoded]

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
                        st.subheader("Scaling Numerical Features")
                        columns_encoded = st.session_state.columns_encoded
                        data = apply_scaling(data, target_column, columns_encoded, columns=sc_cols, info=info_sc)
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
        
        report_test = test_model(best_model, models_type, x_test, y_test)
        st.session_state.report_test = report_test
        st.write(f"Model: {best_model_name}")
        st.write("Testing Report:")
        report_test_df = create_report(report_test, prefix='Test')
        st.session_state.report_test_df = report_test_df
        st.write(report_test_df)

        tvt = st.checkbox("Do you want to display Report Training & Validation again to make comparsion with Testing?")
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
        
        if st.button("Proceed to Save Model"):
            st.session_state.current_step = "save_model"
            
    
    elif st.session_state.current_step == "save_model":
        st.header("Model Saving")
        best_model = st.session_state.best_model
        save_model(best_model)

        st.write("Project still need more features which will be as feature work: ")
        st.write("Author: Elsayed Elmandoh")
        st.write("Machine Learning Engineer")
        
    elif st.session_state.current_step == "load_model":
        st.header("Load Model")
        loaded_model = load_model()
        st.session_state.loaded_model = loaded_model
        
        if loaded_model:
            st.write("Model Loaded.")
            st.write("This step is as feature work.")
            # predict = predict_model(loaded_model,)

if __name__ == "__main__":
    main()