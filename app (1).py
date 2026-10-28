import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time   #added for rate limit handeling
from io import BytesIO

from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
#from langchain_community.chains import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#configure the api_key with respect to what ever google api 
#we have loaded


#left ->pdf's display
#right->text

def get_pdf_text(pdf_docs):
    #with this function whatever pdf text you are uploading it takes and we will make it read and 
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
    
    #now we will return text extracted from pdf and then we will do chunking, embedding and storing

#now divide into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#Take the textchunks and convert into embedding from google genai models
def get_vector_store(text_chunks):
    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    # vector_store=FAISS.from_texts(text_chunks, embedding=embedding)
    # vector_store.save_local("faiss_index")
    #embedded vectors created into local file and which we can't see
        # ✅ batch embedding to avoid rate limit mismatch
    batch_size = 2
    all_embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = embedding.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        time.sleep(1)

    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(text_chunks, all_embeddings)),
        embedding=embedding
    )
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    #prompt_template to take answer
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question:\n{question}\n
    
    Answer:
    """

    # Replace in get_conversational_chain()
    from langchain_groq import ChatGroq

    model = ChatGroq(
        model="llama-3.3-70b-versatile",  # free and fast
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = prompt | model | StrOutputParser()
    #chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    new_db=FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    #calling the conversational cain that we created for the prompt_template the return answer
    #using the model and prompt_template by chaining it
    chain=get_conversational_chain()

    # ✅ Bug 2 fixed — LCEL chain uses .invoke() with context string, not input_documents
    context = "\n\n".join([doc.page_content for doc in docs])
    response=chain.invoke({
        "context": context,
        "question": user_question
    })

    print(response)
    st.write("Reply:", response)

#Creating the streamlit app
def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini")

    user_question=st.text_input("Ask a Question from the PDF Files")

    #if we are calling this user_question function then this function will get executed immediately
    if user_question:
        user_input(user_question)

    #there is a side bar where you can upload a pdf and convert it into vector
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process",
            accept_multiple_files=True  # ✅ Bug 3 fixed
        )
        if st.button("Submit & Process"):
            #when button is pressed call the pages reader, chunk convert and then convert into vectors
            with st.spinner("Processing...."):
                if not pdf_docs:  # ✅ guard against no files uploaded
                    st.warning("Please upload at least one PDF file!")
                else:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:  # ✅ guard against empty text extraction
                        st.warning("Could not extract text from the uploaded PDF(s)!")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")


if __name__ == "__main__":
    main()
