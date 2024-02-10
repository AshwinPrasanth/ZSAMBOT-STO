import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu
from streamlit import session_state as ss  # Import session_state

import yt_dlp
import assemblyai as aai
load_dotenv()

# Function to get PDF text from uploaded files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display chat history
def handle_userinput(user_question):
    response = ss.conversation({'question': user_question})
    ss.chat_history = response['chat_history']

    for i, message in enumerate(ss.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="ZSAMBOT SOLUTIONS 💣", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🌠   :violet[_ZSAMBOT STO SOCIOVERSE_]")
        add_vertical_space(5)
        # Initialize session state variables if not present
        ss.conversation = ss.get("conversation", None)
        ss.chat_history = ss.get("chat_history", None)
        selected = option_menu(
            menu_title=" Menu",
            options=["Home 🏠", "Syllabus 📥 ", "Chatbot 📚", "Developers 🛠️"]
        )
    
    
    if selected == "Developers 🛠️":
        st.header("Crafted with ❤️ and passion by [ZSAM](https://www.linkedin.com/in/ashwin-prasanth-7b8066252)")
        st.write("Software Information:")
        st.write("Version 2a :blue[22nd November,2023]")
        st.write("New Release :red[Transcriber support with youtube]")
        st.write(''' Our Features:
    - Multiple Document uploading facility
    - Chat History 
    - Answers within the Syllabus
    ''')
        add_vertical_space(5)
        st.write("Next Update: Version 2b :blue[2nd February,2024]")
        st.write(''' Upcoming Features:
    - Translator
    ''')
        st.write(":blue[Staaaaay tuuuned!!!!]")

    if selected == "Home 🏠":
        st.title("Welcome to ZSAM BOT")
        st.write(''' I am your Syllabus assistant and I am made using:
    - [Streamlit](https://docs.streamlit.io)
    - [Langchain](https://python.langchain.com/docs/get_started/introduction)
    - [OpenAI](https://platform.openai.com/docs/introduction)
    ## Upload your documents and start your session
    All the best for the subject toppers!!
    ''')

    if selected == "Syllabus 📥 ":
        st.header("Upload Your Syllabus Here! 📩")
        if ss.conversation is None:
            ss.conversation = None
        if ss.chat_history is None:
            ss.chat_history = None
        
        pdf_docs = st.file_uploader(
            "Upload your Syllabus here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                ss.conversation = get_conversation_chain(vectorstore)

        URL = st.text_input("Upload Youtube Link Here", type="default")
        if st.button("upload"):
            with st.spinner("uplaoding"):
                with yt_dlp.YoutubeDL() as ydl:
                    info = ydl.extract_info(URL, download=False)
                for format in info["formats"][::-1]:
                    if format["resolution"] == "audio only" and format["ext"] == "m4a":
                        url = format["url"]
                        break
                aai.settings.api_key = 'ee1353a3aac74d90bacba963e1df0605'
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(url)
                text_chunks = get_text_chunks(transcript.text)
                vectorstore = get_vectorstore(text_chunks)
                ss.conversation = get_conversation_chain(vectorstore)
                st.write(transcript.text)

    if selected == "Chatbot 📚":
        st.header("🏁 QA Session Begins!! 🏁")
        user_question = st.text_input("User Query")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
