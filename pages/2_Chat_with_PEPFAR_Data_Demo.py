from langchain.callbacks import get_openai_callback
import streamlit as st
import pandas as pd
import re
import json
from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import matplotlib.pyplot as plt

from dotenv import load_dotenv

data_library = {
    "FY19-Current PEPFAR Operating Unit by Fine Age and Sex": "resources/OperatingUnit_FineAge_Sex.csv"
}


def extract_code_from_response(response):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)

    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None


def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df1 = pd.DataFrame(data)
        df1.set_index("columns", inplace=True)
        st.bar_chart(df1)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df1 = pd.DataFrame(data)
        df1.set_index("columns", inplace=True)
        st.line_chart(df1)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df1 = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df1)


load_dotenv()
st.set_page_config(page_title="Chat with PEPFAR Data Using a CSV Agent from LangChain")
st.title("Chat with PEPFAR Data Using an Agent from LangChain")
st.markdown(
    '''
    Example queries:
    - Using the HTS_TST in the indicator column, summarize the total number of tests performed by each operating unit in 2021Q1?
    - Using the HTS_TST in the indicator column, make a bar chart for the total number of tests performed by each operating unit in 2021Q1?
    - Filter for HTS_TST in the indicator column and filter for HTS_TST_POS in the indicator column then calculate the sum for each and calculate the yield (HTS_TST_POS/HTS_TST)?
    - Using HTS_TST_POS in the indicator column, can you compare the mean of positives in 2023Q1 to 2022Q1 between each operating unit using a statistical test?
    '''
)

with st.sidebar:
    st.subheader("PEPFAR Data Analysis Chatbot")
    st.markdown(
        "This is an app to demo a PEPFAR data analysis chatbot. "
        "Select or upload your data file (csv) then the chatbox will appear. "
        "Note: The OpenAI agent costs $ to run (though not much). "
        "Selecting OpenAI will display the tokens used as well as the cost. "
        "Data from PEPFAR Panorama Spotlight (https://data.pepfar.gov/). "
    )

    user_data = st.selectbox("Select a data file",
                             options=data_library.keys())
    user_agent = st.selectbox("Select a model to use",
                              options=("gpt-3.5-turbo-0613", "google/flan-t5-xxl"))

data_path = data_library[user_data]
df = pd.read_csv(data_path)
st.dataframe(df, use_container_width=True)  # Show data
user_query = st.text_input("Enter your question here.")

if st.button("Submit") and len(user_query) > 0:

    if user_agent == "gpt-3.5-turbo-0613":
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            data_path,  # file path
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )

        with get_openai_callback() as cb:
            response = agent.run(user_query)
        decoded = extract_code_from_response(response)
        # write_response(decoded)
        st.write(decoded)
        st.write(f"Total Tokens: {cb.total_tokens}")
        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
        st.write(f"Completion Tokens: {cb.completion_tokens}")
        st.write(f"Total Cost (USD): ${cb.total_cost}")

    elif user_agent == "google/flan-t5-xxl":
        st.write("Under Construction")
    #     agent = create_pandas_dataframe_agent(
    #         HuggingFaceHub(
    #             repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5}),
    #         df,
    #         verbose=True
    #     )
    #
    #     #response = agent.run(user_query)
    #     try:  # huggingfacehub seems to give errors, this is a temp workaround to show the response
    #         response = agent.run(user_query)
    #         st.write(response)
    #     except Exception as e:
    #         response = str(e)
    #         if response.startswith("Could not parse LLM output: `"):
    #             response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    #             print(response)
    #             st.write(response)
