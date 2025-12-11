import streamlit as st
from streamlit_option_menu import option_menu

from pages.home import home_app
from pages.wikipedia import wikipedia_app
from pages.about_us import about_us_app

st.set_page_config(
    page_title="Object Detection AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stVerticalBlock"] > div:has(div.stOptionMenu) { margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = 0

query_params = st.query_params
if "nav" in query_params:
    target_nav = query_params["nav"]
    
    if target_nav == "Wiki Search":
        st.session_state["selected_index"] = 1
    elif target_nav == "About Us":
        st.session_state["selected_index"] = 2
    else:
        st.session_state["selected_index"] = 0
        
    del query_params["nav"]

selected = option_menu(
    menu_title=None,
    options=["Home", "Wiki Search", "About Us"],
    icons=["camera-video", "book", "people"], 
    menu_icon="cast",
    
    manual_select=st.session_state["selected_index"],
    
    default_index=0,
    orientation="horizontal",
    key="main_menu",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
        "nav-link-selected": {"background-color": "#FF4B4B"},
    }
)

mapping = {"Home": 0, "Wiki Search": 1, "About Us": 2}
if mapping[selected] != st.session_state["selected_index"]:
    st.session_state["selected_index"] = mapping[selected]
    st.rerun() # Tải lại trang để cập nhật trạng thái ngay lập tức

if selected == "Home":
    home_app()
elif selected == "Wiki Search":
    wikipedia_app()
elif selected == "About Us":
    about_us_app()
else:
    st.error("Page not found!")
    st.stop()