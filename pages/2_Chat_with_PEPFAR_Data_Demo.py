from langchain.callbacks import get_openai_callback
import streamlit as st
import pandas as pd
import re
from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv

data_library = {
    "FY19-Current PEPFAR Operating Unit by Fine Age and Sex": "resources/OperatingUnit_FineAge_Sex.csv"
}


load_dotenv()
st.set_page_config(page_title="Chat with PEPFAR Data Using a CSV Agent from LangChain")
st.title("Chat with PEPFAR Data")


with st.expander("Example queries", expanded=True):
    st.markdown(
        '''
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
        "The default settings use a CSV agent from OpenAI ."
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

with st.expander("Dataset preview"):
    st.dataframe(df, use_container_width=True)  # Show data


user_query = st.text_input("Enter your question here.")


if st.button("Submit") and len(user_query) > 0:

    if user_agent == "gpt-3.5-turbo-0613":
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            data_path,  # File path
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )

        with get_openai_callback() as cb:
            try:
                response = agent.run(user_query)
                print(response)
            except Exception as e:  # Will throw error for python code
                response = str(e)

                # Use regular expression to find content between brackets
                match = re.search(r"\{([^}]+)}", response)
                if match:
                    content_between_brackets = match.group(1)

                    # Use regular expression to find content between double quotes
                    quoted_content = re.findall(r'"([^"]*)"', content_between_brackets)  # Returns a list
                    response = quoted_content[0].encode().decode('unicode_escape')  # Result is in list with another backslash for the newline so will need to remove
        if response:
            try:  # Response is code
                # Making df available for execution in the context
                exec(response)
                #exec(response, globals(), {"df": df, "plt": plt})
                fig = plt.gcf()  # Get current figure
                st.pyplot(fig)  # Display using Streamlit
                st.code(response)
            except Exception as e:  # Response is not code
                print(f"Error executing code: {e}")
                st.write(response)
        st.write("## Costs are:")
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
