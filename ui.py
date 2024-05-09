import streamlit as st
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import LLMChain
from support import DB_collection,output
import traceback

st.title("PG-Manual chatbot")

with st.form("user input"):
    question=st.text_input("Insert your question",max_chars=200)
    button=st.form_submit_button("Generate answer")

if button and question:
    with st.spinner("Loading...."):
        try:
            result = DB_collection.query(
                            query_texts=question,
                            n_results=2)
            Output = output({"context":result["documents"],"question":question})
            



        except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")
        else:
             if Output["OUTPUT"] is not None:
                   st.text_area(label="Answer", value=Output["OUTPUT"])



elif button and not question:
     st.write("Please enter the question and then submit")