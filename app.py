from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Google API configuration
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks for vector storage
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store vector data from text chunks
def get_vector_store(text_chunks):
    # Embedding model for vector creation
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create a new FAISS index from the text chunks dynamically
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
    
    return vector_store  # No need to save it locally, return the index directly

# Function to create the conversational chain
def get_conversational_chain():
    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template="""
        You are an AI assistant that provides answers based on the given context.

        Please follow these guidelines when answering:
        1. **Contextual Understanding**: Read the provided context carefully and ensure that your answer is directly related to it.
        2. **Detail and Clarity**: Answer the question as thoroughly as possible. Include all relevant details and explanations.
        3. **When Information is Missing**: If the answer cannot be found in the context, clearly state: "Answer is not available in the context." Avoid guessing or providing inaccurate information.
        4. **Formatting**: Use clear and concise language, and organize your response for better readability.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )
    
    # Use Google Generative AI for the model
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    
    # Load the question-answering chain with the model and the prompt template
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt_template)
    
    return chain

# Function to handle user input and get the response
def user_input(user_question, vector_store):
    # Use the vector store to search for similar documents
    docs = vector_store.similarity_search(user_question)
    
    # Get the conversational chain to handle the question
    chain = get_conversational_chain()
    
    # Generate the response using the retrieved documents and the user question
    response = chain({'input_documents': docs, 'question': user_question}, return_only_outputs=True)
    
    # Display the answer in the Streamlit app
    st.write('Answer:', response['output_text'])

# Streamlit app main function
def main():
    st.set_page_config(page_title='Chat with Book/PDF')
    st.header('Gup Shup with Your Book/PDF File')

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title('PDF Menu 📖 📄')
        pdf_docs = st.file_uploader('Upload your PDF file here', accept_multiple_files=True)
        
        if st.button('Submit'):
            with st.spinner('Processing...'):
                # Extract text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # Split the text into manageable chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create a FAISS index dynamically from the text chunks
                vector_store = get_vector_store(text_chunks)
                
                st.success('PDF processed and FAISS index created!')

    # User input section for asking questions
    user_question = st.text_input('Ask a question from the book/pdf:')
    
    # Handle user question and response generation
    if user_question and 'vector_store' in locals():
        user_input(user_question, vector_store)

if __name__ == "__main__":
    main()
