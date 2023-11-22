import streamlit as st

st.set_page_config(page_title="Introduction")

st.write("# Welcome to the HIDMSB Large Language Model (LLM) Chatbot Demos! ")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This is a collection of demos using generative AI with PEPFAR documentation and data. 
    The demos show how you can use generative AI for knowledge management, task automation such as writing, data analysis, etc.
    ### Use cases
     -    
    ### Try out a demo from the sidebar!
    - Chat with PEPFAR documentation (document retrieval)
    - Use an agent to analyze MER indicators
    - Suggest other projects
    """
)