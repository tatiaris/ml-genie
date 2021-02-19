import streamlit as st
import pandas as pd
import altair as alt
import streamlit.components.v1 as components
import urllib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso

def remove_footer_hamburger():
    hide_streamlit_style = """
        <style>
        p {margin: 0;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .copyright {text-align: right; color: #666; font-size: 0.875rem; padding: 1rem 0; border-top: 1px solid #ccc;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def copyright_html():
    ''''''
    st.markdown(f'<div class="copyright">Copyright © 2020 - 2021 <a href="https://tatiaris.com">Rishabh Tatia</a></div>', unsafe_allow_html=True)

def load_html_code(fname):
    homepage_html = open(f'frontend/{fname}', 'r').read()
    components.html(homepage_html, height=120)

def display_page_info(header, subheader):
    st.header(header)
    subheader
    ''''''
    ''''''

try:
    remove_footer_hamburger()
    display_page_info('Machine Learning Genie', 'Quickly analyze your dataset and generate predictions')

    data_file = st.file_uploader('Upload your .csv data file here')

    if data_file is not None:
        dataframe = pd.read_csv(data_file)
        '''Original Dataset:'''
        dataframe

        col_names = dataframe.columns.values.tolist()
        input_vars = st.multiselect('Input Variables:', col_names)
        final_inp_vars = list(input_vars)

        for v in input_vars:
            var_type = st.selectbox(f'Variable type for {v}', ['Numerical', 'Categorical', 'Ranked'])
            if (var_type == 'Ranked'):
                unique_vals = dataframe[v].unique()
                ranked_order = st.multiselect(f'Select the rank order for {v} from BEST to WORST', unique_vals)
                if (len(ranked_order) < len(unique_vals)):
                    st.text('Please choose the rank for all unique values')
                else:
                    order_dict = {}
                    for i in range(len(ranked_order)):
                        order_dict[ranked_order[i]] = len(ranked_order) - i
                    dataframe[v] = dataframe[v].map(order_dict)
            elif (var_type == 'Categorical'):
                category_dummies = pd.get_dummies(dataframe[v])
                dataframe = dataframe.join(category_dummies, how='outer')
                unique_categories = dataframe[v].unique().tolist()
                final_inp_vars.extend(unique_categories)
                final_inp_vars.remove(v)
        
        '''Transformed Dataset:'''
        dataframe

        training_vars = st.multiselect('Choose the final input variables:', final_inp_vars, default=final_inp_vars)
        output_var = st.selectbox('Output Variables:', list(set(col_names) - set(input_vars)))

        chosen_model = st.selectbox('Choose a Machine Learning Model:', ['Linear Regression', 'Lasso Regression'])
        ml_model_dict = {'Linear Regression': LinearRegression(), 'Lasso Regression': Lasso()}

        if (len(input_vars) > 0 and output_var):

            X_features = dataframe[training_vars]
            Y_target = dataframe[output_var]
            X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_target, test_size=0.5, random_state=0)

            if (chosen_model == 'Linear Regression'):
                ml_model = LinearRegression()
            elif (chosen_model == 'Lasso Regression'):
                ml_model = Lasso(alpha=0.1)

            try:
                ml_model.fit(X_train, Y_train)
                st.latex(f'R^{2} = {ml_model.score(X_test, Y_test)}')
            except:
                st.error("""One or more variable types are invalid. Please fix them to proceed.""")

    copyright_html()

except urllib.error.URLError as e:
    st.error(f"""Connection error: {e}""")