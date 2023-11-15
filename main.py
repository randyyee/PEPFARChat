# Chat with PEPFAR docs.
# Demo uses the following:
# 1) Interface with streamlit
#    Use streamlit run main.py and ctrl + c to exit
# 2) Utilize LangChain to create knowledge base (pdf chunking)
# 3) OpenAI or embedding model for embeddings
# 4) FAISS for vectorstore
# 5) Functions to handle user input, bot response, and conversation history


import streamlit as st
from dotenv import load_dotenv  # langchain can access secrets
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings  # OpenAI costs $$$!
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from chathtml import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""  # variable to store text
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # create pdf object
        for page in pdf_reader.pages:  # loop through pdfs
            text += page.extract_text()  # add text to text
    return text

# TODO Test out different chunking parameters and methods
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # number of characters
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# TODO Tes out different embedding models and vector databases?
def get_vectorstore(chunked_text):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=chunked_text, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()  # Can switch language models here, ex. huggingface
    #llm =
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']  # create new session state variable for chat history

    for i, message in enumerate(st.session_state.chat_history):  # loop thr history for index and content
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PEPFAR Documentation")

    st.write(css, unsafe_allow_html=True)

    # Initialize session variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask PEPFAR!")
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("PEPFAR Documentation")
        st.markdown(
            "This is an app to demo a PEPFAR documentation chatbot. "
            "Upload your pdfs in the box below then press 'Process' "
            "After processing, you can start to ask questions in the main app area. "
            "This demo is based on https://github.com/alejandro-ao/ask-multiple-pdfs."
        )
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get text
                raw_text = get_pdf_text(pdf_docs)

                # chunk text
                chunked_text = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(chunked_text)

                # create conversation chain using memory, generate new msgs after conversation...
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
