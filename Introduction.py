import streamlit as st

st.set_page_config(page_title="Introduction")

st.write("# Welcome PEPTALK: HIDMSB's Collection of AI Demos! ")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This is a collection of demos using generative AI with PEPFAR documentation and data. 
    The demos show how you can use generative AI for knowledge management, task automation such as writing, data analysis, etc.
    ### Demo use cases
     - Chat with PEPFAR documentation (COP/ROP guidance, technical considerations, reports to Congress, MER reference guides, etc.)
     - Evaluate protocol proposals with data management rubric
     - Evaluate NOFO submissions with rubric
     - Use an agent as a basic data analyst (descriptive stats, basic statistical tests, data visualization)
     - Use LLM for MER narratives analysis
     - Use LLM for writing (correspondence, reports, etc.)
    """
)