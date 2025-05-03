import os
import uuid
from typing import List, Dict

import sounddevice as sd
import streamlit as st
from deepgram import DeepgramClientOptions, DeepgramClient, LiveTranscriptionEvents, LiveOptions
from deepgram.utils import verboselogs
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# Initialize session state
if 'transcriptions' not in st.session_state:
    st.session_state["transcriptions"] = []

# Configuration
st.sidebar.header("Configuration")
deepgram_api_key = st.sidebar.text_input("Deepgram API Key", type="password", value=os.getenv("DEEPGRAM_API_KEY"))
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY"))
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY"))
zilliz_uri = st.sidebar.text_input("Zilliz URI", value=os.getenv("ZILLIZ_URI"))
zilliz_token = st.sidebar.text_input("Zilliz Token", type="password", value=os.getenv("ZILLIZ_TOKEN"))

# Audio device selection
devices = sd.query_devices()
input_devices = [f"{d['name']} ({d['max_input_channels']} in, {d['max_output_channels']} out)" for d in devices]
selected_device = st.sidebar.selectbox("Select Audio Input Device", input_devices)

# Language selection (example languages supported by Deepgram)
languages = ["en", "id"]
selected_language = st.sidebar.selectbox("Select Language", languages)

# RAG configuration
citation_limit = st.sidebar.number_input("Citation Limit", value=15)


# Transcription handling
def transcribe_audio():
    # Find the selected input device index
    device_idx = None
    device_info = None
    for i, d in enumerate(devices):
        if f"{d['name']} ({d['max_input_channels']} in, {d['max_output_channels']} out)" == selected_device:
            device_idx = i
            device_info = d
            break

    if device_idx is None:
        st.error("Selected audio device not found.")
        return

    # Use device's native parameters or fallback to defaults
    dtype = 'int16'
    chunk = 1024 * 8
    samplerate = device_info['default_samplerate']
    channels = device_info['max_input_channels']
    if channels == 0:
        channels = device_info['max_output_channels']

    # Initialize Deepgram client
    config = DeepgramClientOptions(verbose=verboselogs.DEBUG, options={"keepalive": "true"})
    deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=config)

    # Set up Deepgram parameters
    deepgram_options: LiveOptions = LiveOptions(
        diarize=True,
        language=selected_language,
        punctuate=True,
        model="nova-3",
        encoding="linear16",
        sample_rate=samplerate
    )

    connection = deepgram_client.listen.websocket.v("1")

    def on_message(self, result, **kwargs):
        alternatives = result.get("channel", {}).get("alternatives", [])
        if len(alternatives) > 0 and "words" in alternatives[0]:
            transcript = alternatives[0]["transcript"]
            words = alternatives[0]["words"]
            start = words[0]["start"] if words else None
            end = words[-1]["end"] if words else None
            speakers = [word['speaker'] for word in words]
            transcription: Dict = {
                'transcript': transcript,
                'timestamp_start': start,
                'timestamp_end': end,
                'speakers': speakers
            }
            update_transcriptions(transcription)

    def on_error(self, error, **kwargs):
        print(f"Handled Error: {error}")

    connection.on(LiveTranscriptionEvents.Transcript, on_message)
    connection.on(LiveTranscriptionEvents.Error, on_error)

    connection.start(options=deepgram_options)

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        audio_data = indata.tobytes()
        connection.send(audio_data)

    stream = sd.InputStream(
        samplerate=samplerate,
        device=device_idx,
        channels=channels,
        callback=audio_callback,
        dtype=dtype,
        blocksize=chunk,
    )
    stream.start()


if st.sidebar.button("Configure"):
    os.environ["DEEPGRAM_API_KEY"] = deepgram_api_key
    os.environ["COHERE_API_KEY"] = cohere_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key
    st.sidebar.success("Configured!")
    transcribe_audio()


# Ingestion pipeline with Langgraph
class IngestionState(BaseModel):
    chunk: dict = None
    embedding: list = None
    vector_id: str = None


def embed_chunk(state: IngestionState):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    state.embedding = embeddings.embed_documents([state.chunk['transcription']])[0]
    return state


def store_chunk(state: IngestionState):
    vector_store = Milvus(
        embedding_function=CohereEmbeddings(model="embed-v4.0"),
        connection_args={"uri": zilliz_uri, "token": zilliz_token},
        collection_name="transcriber-rag"
    )
    state.vector_id = str(uuid.uuid4())
    vector_store.add_texts(
        texts=[state.chunk['transcription']],
        embeddings=[state.embedding],
        metadatas=[state.chunk]
    )
    return state


ingestion_graph = StateGraph(IngestionState)
ingestion_graph.add_node("embed", embed_chunk)
ingestion_graph.add_node("store", store_chunk)
ingestion_graph.add_edge("embed", "store")
ingestion_graph.add_edge("store", END)
ingestion_graph.set_entry_point("embed")
ingestion_graph.set_finish_point("store")
ingestion_app = ingestion_graph.compile()


# QNA pipeline with Langgraph
class QNAState(BaseModel):
    query: str = None
    embedding: list = None
    documents: list = None
    answer: str = None


def embed_query(state: QNAState):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    state.embedding = embeddings.embed_query(state.query)
    return state


def retrieve_documents(state: QNAState):
    vector_store = Milvus(
        embedding_function=CohereEmbeddings(model="embed-v4.0"),
        connection_args={"host": zilliz_uri, "token": zilliz_token},
        collection_name="transcriptions"
    )
    state.documents = vector_store.similarity_search_by_vector(state.embedding, k=citation_limit)
    return state


def generate_answer(state: QNAState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    context = "\n".join([doc.page_content for doc in state.documents])
    prompt = f"Context: {context}\nQuestion: {state.query}\nAnswer:"
    state.answer = llm.invoke(prompt).content
    return state


qna_graph = StateGraph(QNAState)
qna_graph.add_node("embed_query", embed_query)
qna_graph.add_node("retrieve", retrieve_documents)
qna_graph.add_node("generate", generate_answer)
qna_graph.add_edge("embed_query", "retrieve")
qna_graph.add_edge("retrieve", "generate")
qna_graph.add_edge("generate", END)
qna_graph.set_entry_point("embed_query")
qna_graph.set_finish_point("generate")
qna_app = qna_graph.compile()

# Main Streamlit app
st.title("transcriber-rag")

# Transcription display
st.header("Transcription History")
transcription_container = st.empty()


def update_transcriptions(transcription: Dict):
    transcriptions: List[Dict] = st.session_state["transcriptions"]
    transcriptions.append(transcription)
    ingestion_state = IngestionState()
    ingestion_state.chunk = transcription
    ingestion_app.invoke(ingestion_state)

    with transcription_container:
        for transcription in transcriptions:
            st.write(transcription)


# Q&A section
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Submit"):
    qna_state = QNAState()
    qna_state.query = question
    result = qna_app.invoke(qna_state)
    st.write(f"Answer: {result.answer}")
