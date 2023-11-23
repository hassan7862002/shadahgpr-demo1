import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import  OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Qdrant
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer
# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
import logging
from utils import load_local_vectordb_using_qdrant,get_conversation_chain,create_new_vectorstore_qdrant, get_text_chunks,loadDocuments, translation_query_conversation,eng_to_arabic_prompt,arabic_to_eng_prompt,get_recursive_chunks,create_line_file,preprocess_quran_file,preprocess_english_Quran_csv,prepare_arabic_docs,similarity_final_answer

openapi_key = st.secrets["OPENAI_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]


logging.basicConfig(filename="IslamGPTQdrant.log", format='%(asctime)s %(message)s', filemode='a')

logger = logging.getLogger("Chatbot.log")
logger.setLevel(level=logging.DEBUG)
logger.info("Test Message from app.py")



# embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
# embedding_model_name = "BAAI/bge-small-en-v1.5"
# embedding_model_name = "all-MiniLM-L6-v2"
# embedding_model_name = "intfloat/multilingual-e5-base"
embedding_model_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model= embedding_model_name)
# "with" notation
# embeddings = HuggingFaceEmbeddings(
#         model_name=embedding_model_name)

logger.info(f"Embedding Model = {embedding_model_name}")










def handel_userinput(user_question,vectorstore):
    # user_question += ". Specify the answer with in the given {context}. If you don't find anythin related in context then respond with i don't know"
    # print(user_question)
    # arabic_query = eng_to_arabic_prompt(user_question)
    # logger.info(f"ARABIC QUerY = {arabic_query}")
    final_resp = ""
    source_list = []
    books_info ={1:{"name":"The Meanings of The Quran","author":"Saheeh International"}, 3:{"name":"Interpretation of the meaning of the Qur'an in the English Language","author":"Dr. Muhammad Taqi-ud-Din Al-Hilali & Dr. Muhammad Muhsin Khan"}, 4:{"name":"The Message of THE QUR'ĀN","author":"Muḥammad Asad"}, 6:{"name":"QURAN ARABIC ENGLISH", "author":"Talal Itani (Translation) & Tanzil.net (Arabic Text)"}, 7:{"name": "oxford world's classics THE QUR'AN", "author":"M. A. S. ABDEL HALEEM"}}
    eng_source_response="Sources:\n\n"
    with get_openai_callback() as cb:
        result = st.session_state.conversation({'query': user_question})
        response = result['result']
        logger.info(f"Result={result}")
        source_docs = result['source_documents']
        try:
            for source in source_docs:
                source_name = source.metadata['source']
                source_name_split = source_name.split('/')
                print(source_name_split[2])
                book_name = source_name_split[2]
                book_id = book_name[0]
                if book_name not in source_list:
                    source_list.append(book_name)
                    print("IN IF")
                # else:
                #     print("IN ELSE")
                #     book_info = books_info[book_id]
                #     print("BOOKS INFO",book_info)
                #     book_name_final = book_info['name']
                    
                #     book_author_final = book_info['author']
                #     print(book_name_final,book_author_final)



                
                print(source_name)
                # source_list=source_list+source.metadata['source']
            uniquelist = list(set(source_list))
            for book in uniquelist:
                book_id = book[0]
                book_info = books_info[int(book_id)]
                print("BOOKS INFO",book_info)
                book_name_final = book_info['name']
                
                
                book_author_final = book_info['author']
                eng_source_response = eng_source_response+ book_name_final+'\n'+'('+book_author_final+')'+'\n\n'
                print(book_name_final,book_author_final)

        except Exception as e:
            logger.exception(e)

        print(source_list)

        if response == "I don't know." or response == "I don't know":
            final_resp= "I could not find any relevant information related to your question."
        else:
            reference_response = similarity_final_answer(user_question,vectorstore)
            # final_resp=final_resp+'\n\n' + eng_source_response+'\n'
            eng_resp = "Following are the Arabic References from Quran\n\n"
            final_resp = response + '\n\n'+eng_source_response+'\n\n\n'+eng_resp
            
            final_resp = final_resp + reference_response
        
        # source = result['source_documents'][0].metadata['source']
    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(f"{final_resp} ")
    
    
    # try:
    #     st.session_state.chat_history = response['chat_history']
    # except Exception as e:
    #     pass
    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            # print(messages)
            logger.info(f"Answer= {messages}")
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))



# vectordb_folder_path = '1_en-translation-of-the-meanings-of-the-quran'
    # print(embedding_folder_path)
    # qdrantobj = create_new_vectorstore_qdrant(text_chunks,embeddings,vectordb_folder_path,qdrant_url,qdrant_api_key)

# vetorestore = load_local_vectordb_using_qdrant(vectordb_folder_path,embeddings,qdrant_url,qdrant_api_key)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")

    st.header("Shahada GPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "versedb" not in st.session_state:
        st.session_state.versedb = None

    with st.sidebar:
        # uploaded_files = st.file_uploader("Upload your file", type=[
        #                                   'pdf'], accept_multiple_files=False)
        openai_api_key = openapi_key
        # openai_api_key = st.text_input("OpenAI API Key", key=openapi_key , type="password")
        # process = st.button("Process")
        st.sidebar.write(""" <div style="text-align: center"> The chatbot will bring answers only from the book Meaning of Quran' by Noor International which is  is a profound and comprehensive exploration of the Quran.This work meticulously delves into the Quranic verses, unraveling their meanings, historical context, and relevance to contemporary life. 'Meaning of Quran' serves as an invaluable resource for those seeking spiritual insight, knowledge, and a more profound connection to the teachings of the Quran.</div>""", unsafe_allow_html=True)
        # process = st.text_area("This GPT is about the book Meaning of Quran' by Noor International which is  is a profound and comprehensive exploration of the Quran, offering readers a deeper understanding of its divine wisdom and guidance. This work meticulously delves into the Quranic verses, unraveling their meanings, historical context, and relevance to contemporary life. Noor International's approach is characterized by scholarly expertise and a commitment to promoting a more profound comprehension of the Quran's message, making it accessible to both Muslims and those interested in the study of Islamic scriptures. 'Meaning of Quran' serves as an invaluable resource for those seeking spiritual insight, knowledge, and a more profound connection to the teachings of the Quran. ",disabled=True)
    # if process:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    # documents = loadDocuments()

    # st.write("Documents loaded...")
    # get text chunks
    
    
    # text_chunks = get_text_chunks(documents)
    # final_chunks = text_chunks
    
    # text_chunks = get_recursive_chunks(documents)
    
    # print(type(text_chunks))
    # print(len(text_chunks))
    # print(text_chunks[0],text_chunks[1],text_chunks[2])
    # new_text_chunks = create_line_file('./docs/_quran-simple-enhanced-With-Ayah-Numbers.txt.txt')
    # print(type(new_text_chunks))
    # print(len(new_text_chunks))
    # print(new_text_chunks[0],new_text_chunks[1],new_text_chunks[2])
    
    
    # final_chunks = preprocess_quran_file(new_text_chunks)
    # final_chunks = preprocess_english_Quran_csv()
    # final_chunks = prepare_arabic_docs()
    # print(final_chunks[0])
    # print(final_chunks[1])
    # print(final_chunks[2])
    # st.write("file chunks created...")
    # create vetore stores
    # vectordb_folder_path = 'Quran-Arabic-English-and-meanings-of-quran' + '-' + "Bge-small-v1.5"
    # vectordb_folder_path = 'Quran-Arabic-English' + '-' + "E5-base-multilingual"
    # vectordb_folder_path = 'Quran-Arabic-only' + '-' + 'E5-base-multilingual'
    # vectordb_folder_path = 'Ayat-only-Quran-Arabic-enhanced' + '-' + 'intfloat-multilingual-e5-base'
    
    # vectordb_folder_path = 'Final-NoboleQuran-03-openai-textada-002'
    vectordb_folder_path = 'All-English-Books-openai-textada-002'
    versedb_path = 'Noble-english-ayats-openai-textada-002'
    # vectordb_folder_path = '1_en-translation-of-the-meanings-of-the-quran' + '-' + "Bge-small-v1.5"
    # vectordb_folder_path = 'Quran-Arabic-English-and-meanings-of-quran' + '-' + "multilingual-e5-base"
    # for val in final_chunks:
    #     print(val)
    # vectordb_folder_path = '1_en-translation-of-the-meanings-of-the-quran' + '-' + "Bge-base-v1.5"
    # vectordb_folder_path = '1_en-translation-of-the-meanings-of-the-quran' + '-' + "Bge-large-v1.5"
    # # print(embedding_folder_path)
    # qdrantobj = create_new_vectorstore_qdrant(final_chunks,embeddings,vectordb_folder_path,qdrant_url,qdrant_api_key)

    if st.session_state.versedb == None:
        versedb = load_local_vectordb_using_qdrant(versedb_path,embeddings,qdrant_url,qdrant_api_key)
        st.session_state.versedb = versedb
    if st.session_state.vectorstore == None:
        vetorestore = load_local_vectordb_using_qdrant(vectordb_folder_path,embeddings,qdrant_url,qdrant_api_key)
        st.session_state.vectorstore = vetorestore
    # st.write("Vectore Store Created...")
    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        st.session_state.vectorstore, openai_api_key)  # for openAI
    # st.session_state.conversation =get_conversation_chain(st.session_state.vectorstore)

    st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        logger.info(f"User Question= {user_question}")
        if user_question:
            handel_userinput(user_question,st.session_state.versedb)


if __name__ == '__main__':
    main()