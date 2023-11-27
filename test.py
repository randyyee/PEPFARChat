from langchain.callbacks import get_openai_callback
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms.huggingface_hub import HuggingFaceHub
import pandas as pd
from dotenv import load_dotenv
import re


load_dotenv()

data_library = {
    "FY19-Current PEPFAR Operating Unit by Fine Age and Sex": "resources/OperatingUnit_FineAge_Sex.csv"
}

df = pd.read_csv("resources/OperatingUnit_FineAge_Sex.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    #"resources/OperatingUnit_FineAge_Sex.csv",
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
with get_openai_callback() as cb:
    try:
        response = agent.run("Using the HTS_TST in the indicator column, make a bar chart for total number of tests performed in 2021Q1 and 2021Q2?")
        print(response)
    except Exception as e:  # Will throw error for python code
        response = str(e)

        # Use regular expression to find content between brackets
        match = re.search(r"\{([^}]+)}", response)
        if match:
            content_between_brackets = match.group(1)

            # Use regular expression to find content between double quotes
            quoted_content = re.findall(r'"([^"]*)"', content_between_brackets)  # Returns a list
            clean_content = quoted_content[0].encode().decode('unicode_escape')
            exec(clean_content)  # Resulting content is in the form of a list and will have another backslash for the newline so will need to remove
            print(clean_content)
        else:
            print("No content found between curly brackets.")
print(f"Total Tokens: {cb.total_tokens}")
print(f"Prompt Tokens: {cb.prompt_tokens}")
print(f"Completion Tokens: {cb.completion_tokens}")
print(f"Total Cost (USD): ${cb.total_cost}")

#exec(response)


