from langchain.callbacks import get_openai_callback
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv


load_dotenv()

agent = create_csv_agent(
    OpenAI(temperature=0),
    "resources/OperatingUnit_FineAge_Sex.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
with get_openai_callback() as cb:
#agent.run("Using HTS_TST_POS in the indicator column, can you compare the mean of positives in 2023Q1-R to 2022Q1-R between each operating unit using a statistical test?")
    response = agent.run("Filter for HTS_TST in the indicator column and filter for HTS_TST_POS in the indicator column then calculate the sum for each and calculate the yield (HTS_TST_POS/HTS_TST)?")
print(f"Total Tokens: {cb.total_tokens}")
print(f"Prompt Tokens: {cb.prompt_tokens}")
print(f"Completion Tokens: {cb.completion_tokens}")
print(f"Total Cost (USD): ${cb.total_cost}")