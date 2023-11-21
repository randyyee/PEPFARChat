import streamlit as st

st.set_page_config(page_title="Introduction")

st.write("# Welcome to HIDMSB Large Language Model (LLM) Chatbot Demos! ")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This is a collection of demos using generative AI with PEPFAR documentation and data.
    ### Try out a demo from the sidebar!
    - Chat with PEPFAR documentation
    - Use an agent to analyze MER indicators
    - Suggest other projects
"""
)