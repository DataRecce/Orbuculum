import os

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from orbuculum.database import orbuculum_metadata, DATA_PATH, load_documents, split_documents, add_to_chroma, \
    CHROMA_PATH, clear_database
from orbuculum.embedding import model_map as embedding_model_map
from orbuculum.llm import model_map as llm_model_map
from orbuculum.rag import query_orbuculum

status_map = {
    'init': {
        'label': 'No PDFs provided yet',
        'state': 'error',
    },
    'recharging': {
        'label': 'Recharging Orbuculum...',
        'state': 'running',
    },
    'reset': {
        'label': 'Reset Orbuculum...',
        'state': 'running',
    },
    'ready': {
        'label': 'Orbuculum is ready',
        'state': 'complete',
    },
    'thinking': {
        'label': 'Orbuculum is thinking...',
        'state': 'running',
    },
}


def get_number_of_pdf_files():
    abs_path = os.path.abspath(DATA_PATH)
    if os.path.exists(abs_path):
        return len([f for f in os.listdir(abs_path) if f.endswith('.pdf')])


def dump_pdf_to_data_store(file: UploadedFile):
    with open(os.path.join(DATA_PATH, file.name), 'wb') as f:
        f.write(file.getvalue())


def is_orbuculum_ready() -> bool:
    if os.path.exists(DATA_PATH) is False:
        return False
    if os.path.exists(CHROMA_PATH) is False:
        return False
    return True


# Initialize
if 'pdf_files_in_data_store' not in st.session_state:
    st.session_state.pdf_files_in_data_store = get_number_of_pdf_files()

if "messages" not in st.session_state:
    st.session_state.messages = []

is_ready = is_orbuculum_ready()
st.title("Orbuculum")
st.caption("Give me PDFs and I'll give you insights")
status = st.status('Orbuculum is ready', state='complete') if is_ready else st.status('No PDFs provided yet',
                                                                                      state='error')
error = st.container()

with st.sidebar:
    def recharge():
        status.update(label='Recharging Orbuculum...', state='running')
        try:
            clear_database()
            documents = load_documents()
            print(f'Loaded {len(documents)} documents.')
            chunks = split_documents(documents)
            print(f'Split into {len(chunks)} chunks.')
            add_to_chroma(chunks)
            orbuculum_metadata.save()
        except Exception as e:
            status.update(label='Error occurred while recharging Orbuculum', state='error')
            error.write(e)
            return
        status.update(label='Orbuculum ready', state='complete')


    def clear():
        status.update(label='Clearing Orbuculum...', state='running')
        try:
            abs_path = os.path.abspath(DATA_PATH)
            if os.path.exists(abs_path):
                for f in os.listdir(abs_path):
                    if f.endswith('.pdf'):
                        os.remove(os.path.join(abs_path, f))
        finally:
            status.update(label='No PDFs provided yet', state='error')


    st.header("PDF Library")
    pdf_counts = st.caption(f"Collection of PDF files: {st.session_state.pdf_files_in_data_store}")
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'], accept_multiple_files=False)
    embed_model = st.selectbox("Embedding Model",
                               list(embedding_model_map.keys()),
                               index=0)
    orbuculum_metadata.embedding_model = embed_model
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("Recharge"):
    #         recharge()
    # with col2:
    #     if st.button("Clear", type='primary'):
    #         clear()
    #         st.session_state.pdf_files_in_data_store = 0
    #         pdf_counts.caption(f"Collection of PDF files: {get_number_of_pdf_files()}")

    st.header("LLM Model")
    llm_model = st.selectbox("LLM Model",
                             list(llm_model_map.keys()),
                             index=0)
    st.caption('Model Configuration')
    api_key = None
    if embed_model == 'ffm' or llm_model == 'llama3-ffm':
        api_key = st.text_input('API Key for FFM', type='password')
        st.caption('Please set the API Key for FFM related models.')
        orbuculum_metadata.api_key = api_key

    if uploaded_file:
        dump_pdf_to_data_store(uploaded_file)
        st.session_state.pdf_files_in_data_store = get_number_of_pdf_files()
        pdf_counts.caption(f"Collection of PDF files: {st.session_state.pdf_files_in_data_store}")
        recharge()

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if is_orbuculum_ready() is True:
    if prompt := st.chat_input("Come on, ask me anything!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status.update(label='Thinking...', state='running')
            response = query_orbuculum(
                prompt,
                model=llm_model
            )
            st.write(response)
            st.session_state.messages.append({"role": "ai", "content": response})
            status.update(label='Orbuculum is ready', state='complete')
