from langchain_community.document_loaders import PyPDFLoader
import chromadb
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import LLMChain


#Loading the data
loader = PyPDFLoader("Data/PG-Manual.pdf")
pages = loader.load_and_split()

#Data cleaning
cleaned_pages = []

for page in pages:
    # Remove newline characters
    page_text = page.page_content
    page_text = page_text.replace("\n", "")
    # Remove tab characters
    page_text = page_text.replace("\t", "")
    # Add more cleaning operations if necessary
    
    # Append cleaned page to the list
    cleaned_pages.append(page_text)


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.add(
    documents=cleaned_pages,ids=[str(i) for i in range(len(cleaned_pages))])

DB_collection = collection



prompt_template="""
Based on the given releated information of question try to answer the question given below.If you don't know the answer
return I dont know, don't to generate your own answer
information :{context}
question:{question}
return the helpful answer that set , dont return anything else
Helpful answer :"""

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama",config={"temperature":0.7,'context_length' : 2048},device="cuda")

output =LLMChain(llm=llm,prompt=PROMPT,output_key="OUTPUT",verbose=True)