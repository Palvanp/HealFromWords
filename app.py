import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tensorflow as tf
import numpy as np
from transformers import TFBertModel, BertTokenizer
import os

# Check if vector store exists
if not os.path.exists("faiss_index"):
    # Load the PDF files from the directory
    loader = DirectoryLoader(
        r"E:\Projects\Intel_ai_hackathon\HealingFromWords\data", glob="*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )

    # Create and save vectorstore
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
else:
    # Load existing vectorstore with dangerous deserialization allowed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create LLM with optimized settings
llm = CTransformers(
    model=r"E:\Projects\MindScope\HealingFromWords\model\llama-2-7b-chat.ggmlv3.q4_1.bin",
    model_type="llama",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    config={
        'max_new_tokens': 100,
        'temperature': 0.01,
        'context_length': 2048,
        'threads': 8,
        'batch_size': 512
    }
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Build the conversational chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    memory=memory,
    return_source_documents=False
)

st.title("HealthCare ChatBot 🧑🏽‍⚕")

# Load the emotion detection model (.h5 file)
@st.cache_resource
def load_emotion_model():
    # Use custom object scope to load the model
    with tf.keras.utils.custom_object_scope({'TFBertModel': TFBertModel}):
        return tf.keras.models.load_model('E:\Projects\Intel_ai_hackathon\model\emotion_detector.h5')

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Predict emotion
def predict_emotion(emotion_model, text):
    # Tokenize and pad the input text with max_length=40
    inputs = tokenizer(text, return_tensors='tf', padding='max_length', truncation=True, max_length=40)
    
    input_data = inputs['input_ids']
    attention_mask = inputs['attention_mask']  # Add attention_mask
    
    # Predict emotion
    predictions = emotion_model.predict([input_data, attention_mask])  # Pass both inputs to the model
    
    # Assuming the model predicts a probability distribution for each emotion
    EMOTION_LABELS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    emotion_idx = np.argmax(predictions, axis=-1)  # Get the index of the highest probability
    return EMOTION_LABELS[emotion_idx[0]], predictions[0]

# Define chatbot functionality
@st.cache_data(ttl=3600)
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Display chat history
def display_chat_history(emotion_model):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input(
                "Question:", placeholder="Ask about your Mental Health", key='input'
            )
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                # Chatbot response
                output = conversation_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

                # Emotion detection
                emotion, probabilities = predict_emotion(emotion_model, user_input)
                st.write(f"Emotion Detected: {emotion}")
                st.bar_chart(probabilities)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Load models
emotion_model = load_emotion_model()

# Run the app
initialize_session_state()
display_chat_history(emotion_model)
