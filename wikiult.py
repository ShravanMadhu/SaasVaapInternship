import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

def search_wikipedia(query):
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
    try:
        response = requests.get(search_url)
        if response.status_code != 200:
            return []
        data = response.json()
        search_results = data['query']['search']
        return search_results
    except requests.exceptions.RequestException as e:
        return f"An error occurred while making the request: {e}"

def get_wikipedia_content(page_id):
    page_url = f"https://en.wikipedia.org/wiki?curid={page_id}"
    try:
        response = requests.get(page_url)
        if response.status_code != 200:
            return "No content available."
        
        soup = BeautifulSoup(response.content, 'html.parser')
        content_elements = soup.find_all(['p', 'h2', 'h3'])
        
        sections = []
        current_section = []
        for element in content_elements:
            if element.name in ['h2', 'h3']:
                if current_section:
                    sections.append(current_section)
                current_section = [element]
            else:
                current_section.append(element)
        if current_section:
            sections.append(current_section)
        
        formatted_content = ""
        for section in sections:
            heading = section[0].text.strip()
            text = "\n".join([p.text.strip() for p in section[1:]])

            text = text.replace('$', '\\$').replace('_', '\\_')
            formatted_content += f"**{heading}**\n{text}\n\n"
        
        return formatted_content
    except requests.exceptions.RequestException as e:
        return f"An error occurred while making the request: {e}"
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = OpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    st.set_page_config(page_title='Wikipedia Search and Q&A', page_icon=':books:')

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'search'
    
    if 'current_url' not in st.session_state:
        st.session_state.current_url = None
    
    if 'current_title' not in st.session_state:
        st.session_state.current_title = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
        st.session_state.vectorstore = None

    if st.session_state.current_page == 'search':
        st.title("Wikipedia Semantic Search")

        search_query = st.text_input("Enter your query:")

        if search_query:
            results = search_wikipedia(search_query)
            if not results:
                st.write("No articles found.")
            else:
                st.write("Search Results:")
                for result in results:
                    title = result['title']
                    page_id = result['pageid']
                    url = f"https://en.wikipedia.org/wiki?curid={page_id}"
                    col1, col2 = st.columns([9, 1])
                    with col1:
                        st.write(f"**{title}**")
                        st.write(f"URL: {url}")  
                    with col2:
                        if st.button("Start Q&A", key=page_id):
                            st.session_state.current_page = 'chat'
                            st.session_state.current_url = url
                            st.session_state.current_title = title
                            # Get content and initialize vectorstore and conversation chain
                            raw_text = get_wikipedia_content(page_id)
                            text_chunks = get_text_chunks(raw_text)
                            embeddings = OpenAIEmbeddings()
                            st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                            st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectorstore)
                            st.session_state.chat_history = []
                            st.rerun()

    elif st.session_state.current_page == 'chat':
        st.subheader(f"Chatbot for: {st.session_state.current_title} ({st.session_state.current_url})")

        user_question = st.chat_input("Ask a question...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])

            with st.spinner('Generating response...'):
                if st.session_state.conversation_chain:
                    response = st.session_state.conversation_chain.invoke({'question': user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": response['chat_history'][-1].content})
                    st.chat_message("assistant").write(response['chat_history'][-1].content)
                else:
                    st.write("Conversation chain is not initialized.")
        
        if st.button("Back to Search"):
            st.session_state.current_page = 'search'
            st.session_state.chat_history = []
            st.rerun()
            
if __name__ == "__main__":
    main()
