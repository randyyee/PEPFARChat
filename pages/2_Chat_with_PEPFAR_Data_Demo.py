from langchain_community.callbacks import get_openai_callback
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv

data_library = {
    "Vietnam PSNUxIM": "resources/Vietnam_PSNU_IM_FY22-24.csv",
    "FY19-Current PEPFAR Operating Unit by Fine Age and Sex": "resources/OperatingUnit_FineAge_Sex.csv",
}


load_dotenv()
st.set_page_config(page_title="Chat with PEPFAR Data")
st.title("Chat with PEPFAR Data")


with st.expander("Example queries", expanded=True):
    st.markdown(
        '''
        - The testing yield is HTS_TST_POS/HTS_TST. Using this calculation, can you figure out which mech_name has the lowest yield for period 2023 cumulative? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by and summarize by mech_name before adding the yield.
        - Using the HTS_TST column, make a bar chart for the total number of tests performed by each psnu in 2021 qtr4? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by psnu.
        - The linkage is calculated using TX_NEW/HTS_TST_POS. Using this calculation, can you figure out which mech_name has the lowest linkage for the period 2023 cumulative? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by and summarize by mech_name before adding the linkage.
        - The linkage is calculated using TX_NEW/HTS_TST_POS. Using this calculation, can you compare the ratio in 2023 qtr4 between age_2019 using a statistical test? Be sure to group by and summarize after filtering and before calculating the linkage.
        '''
    )


with st.sidebar:
    st.subheader("PEPFAR Data Analysis Chatbot")
    st.markdown(
        "This is an app to demo a PEPFAR data analysis chatbot. "
        "The default settings use Azure OpenAI. "
        "Select your data file then the chatbox will appear. "
        "Note: The Azure OpenAI agent costs $ to run (though not much). "
        "Selecting Azure OpenAI will display the tokens used as well as the cost. "
        "Data from PEPFAR Panorama (https://data.pepfar.gov/). "
    )

    user_data = st.selectbox("Select a data file",
                             options=data_library.keys())
    user_agent = st.selectbox("Select a model to use",
                              options=("edav-chatapp-share-gpt4-32k-tpm25kplus-v0613-dfilter", "edav-api-share-gpt35-turbo-16k-tpm25plus-v0613-dfilter"))


data_path = data_library[user_data]
df = pd.read_csv(data_path, low_memory=False)

with st.expander("Dataset preview"):
    st.dataframe(df, use_container_width=True)  # Show data


user_query = st.text_input("Enter your question here.")
# user_viz = st.checkbox("Visualize")

if st.button("Submit") and len(user_query) > 0:

    azure_llm = AzureChatOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_KEY"]
    )

    agent = create_pandas_dataframe_agent(
        azure_llm,
        df,
        verbose=True,
        #agent_type=AgentType.OPENAI_FUNCTIONS,
        #handle_parsing_errors=True
    )

    with get_openai_callback() as cb:

        response = agent.invoke(user_query)
        print(response)
        st.write(response)

        # if user_viz:
        #     response = agent.invoke(user_query + "Please provide only the Python code")
        #     exec(response)
        #     fig = plt.gcf()
        #     st.pyplot(fig)
        #     st.code(response)
        # else:
        #     response = agent.invoke(user_query)
        #     st.write(response)

    #     try:
    #         response = agent.invoke(user_query)
    #         print(response)
    #     except Exception as e:  # Will throw error for python code
    #         response = str(e)

    #         # Use regular expression to find content between brackets
    #         match = re.search(r"\{([^}]+)}", response)
    #         if match:
    #             content_between_brackets = match.group(1)

    #             # Use regular expression to find content between double quotes
    #             quoted_content = re.findall(r'"([^"]*)"', content_between_brackets)  # Returns a list
    #             response = quoted_content[0].encode().decode('unicode_escape')  # Result is in list with another backslash for the newline so will need to remove
    # if response:
    #     try:  # Response is code
    #         # Making df available for execution in the context
    #         exec(response)
    #         #exec(response, globals(), {"df": df, "plt": plt})
    #         fig = plt.gcf()  # Get current figure
    #         st.pyplot(fig)  # Display using Streamlit
    #         st.code(response)
    #     except Exception as e:  # Response is not code
    #         print(f"Error executing code: {e}")
    #         st.write(response)
    st.write("## Costs are:")
    st.write(f"Total Tokens: {cb.total_tokens}")
    st.write(f"Prompt Tokens: {cb.prompt_tokens}")
    st.write(f"Completion Tokens: {cb.completion_tokens}")
    st.write(f"Total Cost (USD): ${cb.total_cost}")