import dotenv
dotenv.load_dotenv()
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
import streamlit as st
import textwrap
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
st.title("YouTube Assistant")

def create_vector_db_from_youtube(videourl : str):

    loader = YoutubeLoader.from_youtube_url(videourl)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs,embeddings)
    return db

def get_response_from_query(db, query, k):

    docs = db.similarity_search(query, k = k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model = "text-davinci-003")

    prompt = PromptTemplate(input_variable = ["question","docs"], 
                            template = """ You are helpful assistant that can answer questions about videos
                                        based on the video's transcript.
                                         Answer the following question : {question} 
                                         By searching the following video transcript : {docs}
                                         Only factual information from the transcript to answer the question 
                                         
                                         if you feel like i don't have enough information to answer the question say
                                         "I Don't Know".
                                         your answers should be detailed """)
    chain = LLMChain(llm = llm, prompt = prompt)

    response = chain.run(question = query, docs = docs_page_content)
    response = response.replace("\n", "")
    return response

with st.sidebar:
    with st.form(key = "my_form"):
        youtube_url = st.sidebar.text_area(label = "Enter the YouTube URL", max_chars = 50)

        query = st.sidebar.text_area(label = "Enter question you need to ask?", max_chars = 50, key = "query")

        submitbutton = st.form_submit_button(label = "Submit")


if query and youtube_url:

    db = create_vector_db_from_youtube(youtube_url)

    response, docs = get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width = 80))


