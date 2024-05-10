# Chat with PEPFAR docs.
# Demo uses the following:
# 1) Interface with streamlit
#    Use streamlit run 1_Chat_with_PEPFAR_Docs_Demo.py and ctrl + c to exit
# 2) Utilize LangChain to create knowledge base (pdf chunking)
# 3) OpenAI or embedding model for embeddings
# 4) FAISS for vectorstore
# 5) Functions to handle user input, bot response, and conversation history
# TODO Final release save persistent embeddings and vectorstore

import os
import streamlit as st
from dotenv import load_dotenv  # load .env so langchain can access secrets
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings  # OpenAI costs $$$!
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI


document_library = {
    "FY23 PEPFAR Country and Regional Operational Plan": "resources/PEPFAR-2023-Country-and-Regional-Operational-Plan.pdf",
    "FY24 PEPFAR Technical Considerations": "resources/FY-2024-PEPFAR-Technical-Considerations.pdf",
    "PEPFAR 5 Year Strategy": "resources/PEPFARs-5-Year-Strategy_WAD2022_FINAL_COMPLIANT_3.0.pdf"
}


def get_pdf_text(pdf_docs):
    text = ""  # variable to store text
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # create pdf object
        for page in pdf_reader.pages:  # loop through pdfs
            text += page.extract_text()  # add text to text
    return text


# def main():
load_dotenv()
st.set_page_config(page_title="Ask PEPFAR")

with st.sidebar:
    st.subheader("PEPFAR Knowledge Chatbot")
    st.markdown(
        "This is an app to demo a PEPFAR documentation chatbot. "
        "Select your doc(s) and check \"Done!\" then the chatbox will appear."
    )

    doc_names = list(document_library.keys())  # get a list of the keys from the documents
    pdf_docs = st.sidebar.multiselect(  # add keys to the multiselect so the user can select
        'Choose your doc(s)', doc_names
    )
    selected_doc_list = []
    for d in pdf_docs:  # loop through the list of selected docs and get the docs
        selected_docs = document_library.get(d)
        selected_doc_list.append(selected_docs)

    user_size = st.number_input(
        "Chunk Size", value=1000, step=100
    )
    user_overlap = st.number_input(
        "Chunk Overlap", value=200, step=100
    )
    # user_embedding = st.selectbox(
    #     "Select embedding", options=("edav-api-share-text-embeding-ada-tpm100plus-v002-dfilter")
    # )
    user_llm = st.selectbox(
        "Select LLM", options=("edav-chatapp-share-gpt4-32k-tpm25kplus-v0613-dfilter", "edav-api-share-gpt35-turbo-16k-tpm25plus-v0613-dfilter")
    )
    accept = st.button("Ready!")
    st.markdown(
        '''
        This demo is based on:
        - langchain tutorials
        - streamlit tutorials
        - https://github.com/alejandro-ao/ask-multiple-pdfs
        '''
    )  # end of sidebar

st.title("Chat with PEPFAR Documentation")


with st.expander("Example queries", expanded=True):
    st.markdown(
        '''
        - Try asking something specific, something you can easily check...
        - Summarize PEPFAR's 5 year strategy.
        - What is this document's executive summary?
        - What does the document say about data?
        '''
    )

if accept and len(selected_doc_list) > 0:  # detect if user selects anything from the document library
    # 1) Read doc
    raw_text = get_pdf_text(selected_doc_list)

    # 2) Chunk doc (fixed-size chunking)
    # TODO Test out different chunking parameters and methods; contextual chunk headers for multiple documents
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=user_size,  # number of characters
        chunk_overlap=user_overlap,
        length_function=len
    )
    chunked_text = text_splitter.split_text(raw_text)

    # 3) Run embeddings and store in vector database
    # TODO Test out different embedding models and vector databases
    embeddings = AzureOpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunked_text, embedding=embeddings)

    # 4) Setup LLM
    # TODO Test out different LLMs and parameters
    llm = AzureChatOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_KEY"]
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # 5) Chat section

    #client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if user_llm not in st.session_state:
        st.session_state["openai_model"] = user_llm

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            #stream = client.chat.completions.create(
            stream = llm.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})