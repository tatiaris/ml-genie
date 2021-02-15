import streamlit as st
import pandas as pd
import altair as alt
import streamlit.components.v1 as components
import urllib
import numpy as np

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
    st.write('')
    st.markdown(f'<div class="copyright">Copyright © 2020 - 2021 <a href="https://tatiaris.com">Rishabh Tatia</a></div>', unsafe_allow_html=True)

def load_html_code(fname):
    homepage_html = open(f'frontend/{fname}', 'r').read()
    components.html(homepage_html, height=120)

try:
    remove_footer_hamburger()
    load_html_code('home.html')
    name = st.text_input('Please enter your name')
    greeting_msg = f'Hello {name}! ' if (name != "") else ''
    st.write(greeting_msg + 'Choose a number below to customize the random area plot.')
    x = st.slider("", 0, 100, value=10)
    st.write('Random data points per variable:', x)
    chart_data = pd.DataFrame(np.random.randn(x, 3), columns=['python', 'javascript', 'c++'])
    st.area_chart(chart_data)
    st.markdown('Cheat Sheet Link: [HERE](https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py)')
    copyright_html()

except urllib.error.URLError as e:
    st.error("""Connection error: %s""" % e.reason)