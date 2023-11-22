from langchain.callbacks import get_openai_callback
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from dotenv import load_dotenv

st.set_page_config(page_title="Chat with PEPFAR Data Using a CSV Agent from LangChain")
st.title("Chat with PEPFAR Data Using a CSV Agent from LangChain")

document_library = {
    "FY19-Current PEPFAR Operating Unit by Fine Age and Sex": "resources/OperatingUnit_FineAge_Sex.csv"
}


with st.sidebar:
    st.subheader("PEPFAR Data Analysis Chatbot")
    st.markdown(
        "This is an app to demo a PEPFAR data analysis chatbot. "
        "Select your doc(s) and check \"Done!\" then the chatbox will appear."
        "Data from [https://data.pepfar.gov/]."
    )

    user_data = st.selectbox("Select a data file",
                             options=("resources/FY19-Current PEPFAR Operating Unit by Fine Age and Sex"))

    accept = st.checkbox("Done!")

load_dotenv()

agent = create_csv_agent(
    OpenAI(temperature=0),
    "resources/OperatingUnit_FineAge_Sex.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

user_query = st.text_input("Enter your question here.")

if st.button("Submit") and len(user_query) > 0:
    with get_openai_callback() as cb:
        response = agent.run(user_query)
    st.write(response)
    st.write(f"Total Tokens: {cb.total_tokens}")
    st.write(f"Prompt Tokens: {cb.prompt_tokens}")
    st.write(f"Completion Tokens: {cb.completion_tokens}")
    st.write(f"Total Cost (USD): ${cb.total_cost}")