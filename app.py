from pydoc import doc
import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

st.session_state["selected_urls"]=""
openai_api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def caching(selected_urls):
    if st.session_state["selected_urls"]:
        loader = AsyncChromiumLoader(selected_urls)
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)

        # Split the texts
        st.spinner("Spliting text....")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs_transformed)

        # Retrieve relevant documents
        st.spinner("Create retriver....")
        retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

        llm = OpenAI(temperature=0)  # Note: Ensure you have necessary imports and setup for OpenAI
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        return compression_retriever
    return

def pretty_print_docs(docs):
    for d in docs:
        st.write(str(d.metadata["source"]) +":\n\n" + d.page_content)


def main():
    st.title("URL Scraper and Searcher")
    
    # Step 1: Input URL and scrape href links
    url = st.text_input("Enter a URL:","")
    if not url:
        st.warning("Please input a URL.")
        return
    
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    
    urls = [url + link.get('href') for link in soup.find_all('a')]
    
    # Display checkboxes for each link
    selected_urls = st.multiselect("Choose links:", urls, default=urls,label_visibility="visible")
    st.write('You selected:', selected_urls)
    st.session_state["selected_urls"]=selected_urls

    
    # Step 2: Prompt to search
    query = st.text_input("Enter your query:")
    if not query:
        st.warning("Please enter a query.")
        return

    if st.button("Search"):
        # Load and transform the documents
        compression_retriever = caching(selected_urls)

        compressed_docs = compression_retriever.get_relevant_documents(query)
        pretty_print_docs(compressed_docs)
        


if __name__ == "__main__":
    main()
