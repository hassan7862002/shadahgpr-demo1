import os
import logging
import csv
from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS,Qdrant
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from PyPDF2 import PdfReader
import openai

logging.basicConfig(filename="IslamGPTQdrant_utils.log", format='%(asctime)s %(message)s', filemode='a')

logger = logging.getLogger("Chatbot_utils.log")
logger.setLevel(level=logging.DEBUG)
logger.info("Test Message from Utils.py")



def loadDocuments():
    documents = []
    for file in os.listdir('./docs/'):
        # logger.info(f"File found in docs: {file}")
        if file.endswith('.pdf'):
            pdf_path = './docs/' + file
            # return pdf_path
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        # elif file.endswith('.docx') or file.endswith('.doc'):
        #     doc_path = './docs/' + file
        #     loader = Docx2txtLoader(doc_path)
        #     documents.extend(loader.load())
        elif file.endswith('.txt'):
            txt_path = './docs/' + file
            loader = TextLoader(txt_path)
            documents.extend(loader.load())
    return documents

def load_vectorstore(folder_path, embeddings):
    
    if os.path.exists(folder_path):
        vectordb = FAISS.load_local(folder_path,embeddings)
        return vectordb
    else:
        print("Path Not Found")


def get_prompt():
    prompt_template = """
        you are an Islamic scholar. As an islamic scholar, you are expert in extracting information related to islamic queries. You are trained to analyze the given paragraphs and extract relevant information from the query.
        If you able to find any relevant information, then you must analyze the paragraphs and make a detailed information just using those paragraphs. If you don't find any relevant information simple return "I don't know".  
        paragraphs: {context}
        query: {question}
        Relevant information: 
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    logger.info(f"Prompt_Template= {prompt_template}")
    return PROMPT

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(doc):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    logger.info("Before SPlitting")
    chunks = text_splitter.split_documents(doc)
    logger.info("After Splitting")
    return chunks


def get_recursive_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=100,length_function=len)
    chunks = text_splitter.split_documents(doc)
    return chunks

def get_conversation_chain(vetorestore, openai_api_key):
    # llm = ChatOpenAI(openai_api_key=openai_api_key,
    #                  model_name='gpt-3.5-turbo-16k', temperature=0)
    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)
    chain_type_kwargs = {"prompt": get_prompt()}
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), chain_type="stuff", retriever = vetorestore.as_retriever(search_type="similarity", search_kwargs={'k': 5}),
                                           chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vetorestore.as_retriever(),
    #     memory=memory
    # )
    return qa_chain


def create_new_vectorstore_qdrant( doc_list, embed_fn, COLLECTION_NAME,qdrant_url, qdrant_api_key ):
    try:
        qdrant = Qdrant.from_documents(
            documents = doc_list,
            embedding = embed_fn,
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name=COLLECTION_NAME,
        )
        logger.info("Successfully created the vectordb")
        return qdrant
    except Exception as ex:
        logger.exception("Vectordb Failed:"+str(ex))
        # return JSONResponse({"Error": str(ex)})

def load_local_vectordb_using_qdrant( vectordb_folder_path, embed_fn, qdrant_url, qdrant_api_key):
    try:
        qdrant_client = QdrantClient(
            url=qdrant_url, 
            prefer_grpc=True,
            api_key=qdrant_api_key,
        )
        logger.info("Qdrant client loaded Successfully")
        qdrant_store= Qdrant(qdrant_client, vectordb_folder_path, embed_fn)
        logger.info("Successfully loaded vectordb")
        return qdrant_store   
    except Exception as e:
        logger.critical(f"error while loading vectordb:'{str(e)}'")
        raise Exception(f"error while loading vectordb:'{str(e)}'")
    
#similarity search
def similarity(vetorestore,text):
    '''Simple Similarity search using Vectorstore. Returns top 5 results.'''
    text=vetorestore.similarity_search(text,k=5)
    # print(text)
    document = text
    page_content = ""
    for doc in document:
        # print("verse= ", doc.page_content)
        # print("metadata= ", doc.metadata)
        page_content = page_content + '\n' + doc.page_content
    print(page_content)
    return page_content

def similarity_with_score(vetorestore,text):
    text = vetorestore.similarity_search_with_score(text,k=3)
    # print(text)
    all_docs = []
    document = text
    page_content = ""
    # print(type(document))
    for doc in document:
        # print("DOC IS ", doc)
        # print(type(doc))
        # print(len(doc))
        a,b = doc
        # print('a is ', a)
        # print('b is ', b)
        # print("verse=", doc.page_content)
        # print("metadata= ", doc.metadata)
        # print("score= ", doc.score)
        # page_content = page_content + '\n' + doc.page_content
    # toplist = return_top_k(document,7)
    # print(len(toplist))
    # print("TOPLIST= ", toplist)
    # print("LENGTH OF DOCUMENT LIST FOR this query", len(document))
    return document

def return_top_k(list_of_tuples_docs,k):
    '''Takes a list of tuples and returns top "k" elements on the basis of score'''
    sorted_tuples = sorted(list_of_tuples_docs, key=lambda t: t[1], reverse=True)
    top_k_tuples = sorted_tuples[:k]

    return top_k_tuples


def generate_Instagram_content(topic):
    '''Convertion process for English to Arabic Language'''
    messages = [
    {"role": "system", "content": f"""You are trained to analyze {topic} and convert this {topic} into arabic language."""},
    {"role": "user", "content": f"""You are trained to analyze {topic} and convert this {topic} into arabic language.Response should be in (arabic) language."""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 440, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()

    return response_text

def arabic_to_eng_convertion(topic):
    '''Convertion process for Arabic to English Language'''
    messages = [
    {"role": "system", "content": f"""You are trained to analyze {topic} and convert this {topic} into english language."""},
    {"role": "user", "content": f"""You are trained to analyze {topic} and convert this {topic} into english language.Response should be in (english) language."""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 440, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()

    return response_text







def translation_query_conversation(prompt):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",prompt=prompt, max_tokens=440, temperature=0.5)
    print(response)
    logger.info(f"Translated Text= {response}")
    return response

def arabic_to_eng_prompt(arabic_text):
    '''Translates the Arabic Text to English using OpenAI'''
    prompt_template = f"Translate this text to english {arabic_text}"
    # PROMPT = PromptTemplate(template=prompt_template,input_variables=["arabic_text"])
    translated_text = arabic_to_eng_convertion(arabic_text)
    return translated_text
    
def eng_to_arabic_prompt(eng_text):
    '''Translate the English Text to Arabic using OpenAI'''
    prompt_template = f"Translate this text to arabic {eng_text}"
    # PROMPT = PromptTemplate(template=prompt_template,input_variables=["eng_text"])
    translated_text = generate_Instagram_content(eng_text)
    return translated_text


def create_line_file():
    lines = []
    filetxt = './other documents/_quran-simple-enhanced-With-Ayah-Numbers.txt.txt'
    # filetxt = ''
    with open(filetxt,"r") as f:
        for line in f.readlines():
            lines.append(line)
    # print("lines are ", lines)
    return lines 

def create_surah_list(filename):
    lines = []
    # filetxt = './docs/_quran-simple-enhanced-With-Ayah-Numbers.txt.txt'
    # filetxt = ''
    with open(filename,"r") as f:
        for line in f.readlines():
            lines.append(line)
    # print("lines are ", lines)
    return lines        


def get_surah_dictionary():
    '''Creates a Surah Dictionary, containing Surah numbers as key and Surah names as value'''
    surah_names = create_surah_list('./surah_list.txt')
    surah_dict = {}
    for name in surah_names:
        # print(name)
        number, s_name = name.split('.')
        # print(number, s_name)
        surah_dict[number] = s_name
    return surah_dict

def preprocess_quran_file(text_lines):
    B_verse = 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ'
    
    documents = []
    for line in text_lines:
        # print("THIS IS A LINE")
        # print(line)
        new_line = line.replace(B_verse,"")
        try:
            surah_no, verse_no, verse = new_line.split("|")
            surah_dict = get_surah_dictionary()
            surah_name = surah_dict[surah_no]
            metadata = {}
            metadata['surah_number'] = surah_no
            metadata['surah_name'] = surah_name
            metadata['verse_number'] = verse_no
            new_doc = Document(page_content=verse, metadata=metadata)
            del metadata
            documents.append(new_doc)
        except Exception as e:
            logger.exception(e)
    # for i in documents:
    #     print(i)
    return documents

def preprocess_english_Quran_csv():
    documents = []
    with open('./other documents/Quran_English.csv', 'r') as file:
        csvreader = csv.reader(file)
        surah_dict = get_surah_dictionary()
        for row in csvreader:
            try:
                surah_no = row[1]
                verse_no = row[2]
                verse = row[3]
                surah_name = surah_dict[surah_no]
                metadata = {}
                metadata['surah_number'] = surah_no
                metadata['surah_name'] = surah_name
                metadata['verse_number'] = verse_no
                new_doc = Document(page_content=verse,metadata=metadata)
                del metadata
                documents.append(new_doc)
            except Exception as e:
                logger.exception(e)
    return documents
            
        

def prepare_arabic_docs():
    lines = create_line_file()
    doc = preprocess_quran_file(lines)
    return doc

def create_tuple_dictionary_surah_verses():
    lines = create_line_file()
    B_verse = 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ'
    
    # documents = []
    verse_tuple_dictionary = {}
    for line in lines:
        # print("THIS IS A LINE")
        # print(line)
        try:
            new_line = line.replace(B_verse,"")
            
            surah_no, verse_no, verse = new_line.split("|")
            key = (surah_no,verse_no)
            verse_tuple_dictionary[key] = verse
            # surah_dict = get_surah_dictionary()
            # surah_name = surah_dict[surah_no]
            # metadata = {}
        except Exception as e:
            logger.exception(e)
    return verse_tuple_dictionary


def unique_tuple_list(listtuples):
    unique_strings = set()
    unique_tuples = []
    for tuple in listtuples:
        string_value = tuple[0]
        string_value = string_value.page_content
        if string_value not in unique_strings:
            unique_strings.add(string_value)
            unique_tuples.append(tuple)
    return unique_tuples
# def generate_alternate_queries(query):
#     messages = [
#     {"role": "system", "content": f"""You are trained to analyze {query} and generate an alternate to this {query}."""},
#     {"role": "user", "content": f"""You are trained to analyze {query} and generate an alternate to this {query}. Response should be (simple) yet (meaningful)."""}
#     ]

#     response = openai.ChatCompletion.create(
#                         model="gpt-3.5-turbo",
#                         messages=messages, 
#                         max_tokens= 440, 
#                         n=4, 
#                         stop=None, 
#                         temperature=0.6)

#     response_text = response.choices[0].message.content.strip()
#     return response_text



def generate_multi_prompt_from_master(topic):
    generated_response=[]
    try:
        messages = [
        {"role": "system", "content": f"""You are trained to analyze the different characteristics and different aspects of {topic} and generate 4 different (queries).Response must only contain queries.Do not add (description) of each (queries) in (response). All (queries) should (vary) from each others.Every query should contain different aspects, characteristic and view of {topic}.All queries must not be same. The new queries must not be identical and they should generate different content.
         As we need 4 queries in return. So, The List of queries should be in the form: [1. Query1,2. Query2,3. Query3,4. Query4].
        """},
        {"role": "user", "content": f"""you are trained to analyze the different characteristics and different aspects of {topic} and generate 4 different (queries).All (queries) should (vary) from each others.Every query should contain different aspects, characteristic and view of {topic}.All queries must not be same. The new queries must not be identical and they should generate different content."""}
        ]

        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=messages, 
                            max_tokens= 440, 
                            n=1, 
                            stop=None, 
                            temperature=0.6)
    except Exception as e:
        print(f'Error : {str(e)}')
    prompt_list = response['choices'][0]['message']['content'].strip()
    # print(prompt_list)
    prompt_list = prompt_list.replace("\\","")
    prompt_list = prompt_list.split("\n")
    # response_without_backslashes = prompt_list.replace("\\", "")
    if prompt_list is not None:
        text = [item for item in prompt_list if item]
        # print("---------------------")
        # print(text)
        # print(len(text))
        # print(type(text))
        for item in text:
            # print(item)
            newstr = item[3:]
            # print(newstr)
            generated_response.append(newstr)
        # print("---------------------")
        # print(type(generated_response))
        # frequency=int(frequency)
        # generated_response=text[:frequency]
        # print("---------------------")
        # print(generated_response)
        # print(len(generated_response))
        # print("---------------------")
        return generated_response 
    else:
        return None
        


def similarity_final_answer(query,vectorstore):
    if query:
        query_alternates = generate_multi_prompt_from_master(query)
        query_alternates.append(query)
        final_response = ""
        arabic_ayats = ""
        all_doc_list = []
        arabic_dict = create_tuple_dictionary_surah_verses()
        # print("ALL Queries", all_queries)
        # print("QUERY ALTERNATES=",query_alternates)
        # print("query", query)
        for q in query_alternates:
            
            doc_list_single= similarity_with_score(vectorstore,q)
            for doc in doc_list_single:
                all_doc_list.append(doc)
        # print("FINAL LIST OF DOCS= ", all_doc_list)
        # print("Length is = ", len(all_doc_list))
        top_k = return_top_k(all_doc_list,6)
        # print("TOP RES= ",top_k)
        newlinechar = '\n'
        tabchar = '\t'
        # print(len(top_k))
        duplicate_removed_results = unique_tuple_list(top_k)


        for doc in duplicate_removed_results:
            content,metadata = doc
            # print("Content is= ", content)
            # print("TYPE=", type(content))
            # print("VERSE is = ", content.page_content)
            # print("META = ", content.metadata)
            # print("Surah= ", content.metadata['surah_name'])
            # print("Surah No= ", content.metadata['surah_number'])
            # print("Verse No= ", content.metadata['verse_number'])
            # print("Score= ", metadata)
            key = (content.metadata['chapter_number'],content.metadata['verse_number'])
            arabic_verse = arabic_dict[key]
            # final_response = f"{final_response} {}"
            final_response = f"{final_response} {newlinechar} {arabic_verse} {newlinechar} {content.page_content} {newlinechar} Surah Name:{content.metadata['chapter_name']} {tabchar} Surah Number:{content.metadata['chapter_number']} {newlinechar}Verse Number:{content.metadata['verse_number']} {newlinechar}"
        
        return final_response


        







