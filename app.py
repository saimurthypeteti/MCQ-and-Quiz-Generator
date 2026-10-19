"""Streamlit entry point; service logic stays in backend/src modules."""
import streamlit as st
from src.streamlit_ui import render_page

st.set_page_config(page_title="Quill Studio · MCQ Generator", page_icon="✳", layout="wide")
pages=[
    st.Page("pages/1_Dashboard.py",title="Dashboard",icon="🏠",default=True),
    st.Page("pages/2_Upload_Documents.py",title="Upload Documents",icon="📚"),
    st.Page("pages/3_Generate_Quiz.py",title="Generate Quiz",icon="✳"),
    st.Page("pages/4_Review_Questions.py",title="Question Review",icon="📝"),
    st.Page("pages/5_Take_Quiz.py",title="Take Quiz",icon="✅"),
    st.Page("pages/6_Results.py",title="Results",icon="📊"),
    st.Page("pages/7_Quiz_History.py",title="Quiz History",icon="🗂️"),
    st.Page("pages/8_Model_Evaluation.py",title="RAG & Model Evaluation",icon="🔎"),
    st.Page("pages/9_Settings.py",title="Settings",icon="⚙️"),
]
st.navigation(pages).run()
