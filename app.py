import asyncio
import os
import queue
import uuid
from datetime import datetime
from queue import Queue

import sounddevice as sd
import streamlit as st
from deepgram import DeepgramClientOptions, DeepgramClient, LiveTranscriptionEvents, LiveOptions
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Initialize session state
if 'transcriptions' not in st.session_state:
    st.session_state["transcriptions"] = []
if 'transcription_queue' not in st.session_state:
    st.session_state["transcription_queue"] = queue.Queue()
if 'configured' not in st.session_state:
    st.session_state["configured"] = False

# Configuration
st.sidebar.header("Configuration")
deepgram_api_key = st.sidebar.text_input("Deepgram API Key", type="password", value=os.getenv("DEEPGRAM_API_KEY"))
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY"))
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY"))
zilliz_uri = st.sidebar.text_input("Zilliz URI", value=os.getenv("ZILLIZ_URI"))
zilliz_token = st.sidebar.text_input("Zilliz Token", type="password", value=os.getenv("ZILLIZ_TOKEN"))

# Audio device selection
devices = sd.query_devices()
input_devices = [d['name'] for d in devices if d['max_input_channels'] > 0]
selected_device = st.sidebar.selectbox("Select Audio Input Device", input_devices)

# Language selection (example languages supported by Deepgram)
languages = ["en", "id"]
selected_language = st.sidebar.selectbox("Select Language", languages)

# RAG configuration
citation_limit = st.sidebar.number_input("Citation Limit", value=15)

if st.sidebar.button("Configure"):
    os.environ["DEEPGRAM_API_KEY"] = deepgram_api_key
    os.environ["COHERE_API_KEY"] = cohere_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key
    st.session_state["configured"] = True
    st.sidebar.success("Configured!")


# Transcription handling
async def transcribe_audio():
    if "configured" not in st.session_state or not st.session_state["configured"]:
        st.error("Please configure first.")
        return

    # Find the selected input device index
    device_idx = None
    for i, d in enumerate(devices):
        if d['name'] == selected_device and d['max_input_channels'] > 0:
            device_idx = i
            break

    if device_idx is None:
        st.error("Selected audio device not found.")
        return

    # Audio parameters
    samplerate = 16000
    channels = 1
    dtype = 'int16'

    # Audio buffer
    audio_queue = asyncio.Queue()

    # Callback function for audio capture
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        audio_queue.put_nowait(bytes(indata))

    # Initialize Deepgram client
    config = DeepgramClientOptions(options={"keepalive": "true"})
    dg = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"], config=config)

    # Set up Deepgram parameters
    deepgram_options: LiveOptions = LiveOptions(
        diarize=True,
        language=selected_language,
        punctuate=True,
        model="nova-3",
        encoding="linear16",
        sample_rate=samplerate
    )

    connection = dg.listen.asyncwebsocket.v("1")

    async def on_message(self, result, **kwargs):
        alternatives = result.get("channel", {}).get("alternatives", [])
        if len(alternatives) > 0 and "words" in alternatives[0]:
            for word in alternatives[0]["words"]:
                transcription_queue: Queue = st.session_state["transcription_queue"]
                transcription_queue.put({
                    'text': word['word'],
                    'timestamp': word.get('start', 0),
                    'speaker': word.get('speaker', 0)
                })

    connection.on(LiveTranscriptionEvents.Transcript, on_message)

    await connection.start(options=deepgram_options)

    # Start the audio stream
    stream = sd.InputStream(
        samplerate=samplerate,
        device=device_idx,
        channels=channels,
        callback=audio_callback,
        dtype=dtype
    )
    stream.start()

    while True:
        audio_data = await audio_queue.get()
        await connection.send(audio_data)
        await asyncio.sleep(0.1)


# Ingestion pipeline with Langgraph
class IngestionState(BaseModel):
    chunk: dict = None
    embedding: list = None
    vector_id: str = None


def embed_chunk(state: IngestionState):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    state.embedding = embeddings.embed_documents([state.chunk['text']])[0]
    return state


def store_chunk(state: IngestionState):
    vector_store = Milvus(
        embedding_function=CohereEmbeddings(model="embed-v4.0"),
        connection_args={"uri": zilliz_uri, "token": zilliz_token},
        collection_name="transcriber-rag"
    )
    state.vector_id = str(uuid.uuid4())
    vector_store.add_texts(
        texts=[state.chunk['text']],
        embeddings=[state.embedding],
        metadatas=[{'timestamp': state.chunk['timestamp'], 'speaker': state.chunk['speaker']}]
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


# Background processing
def process_transcriptions():
    while True:
        try:
            transcription_queue: Queue = st.session_state["transcription_queue"]
            chunk = transcription_queue.get_nowait()
            st.session_state["transcriptions"].append(chunk)
            ingestion_state = IngestionState()
            ingestion_state.chunk = chunk
            ingestion_app.invoke(ingestion_state)
        except queue.Empty:
            break


# Main Streamlit app
st.title("transcriber-rag")

# Transcription display
st.header("Transcription History")
transcription_container = st.empty()


async def update_transcriptions():
    process_transcriptions()
    with transcription_container:
        for t in st.session_state["transcriptions"]:
            st.write(f"[{datetime.fromtimestamp(t['timestamp'])}] Speaker {t['speaker']}: {t['text']}")


if st.session_state["configured"]:
    if "transcribe_audio_task" in st.session_state and st.session_state["transcribe_audio_task"]:
        st.session_state["transcribe_audio_task"].cancel()

    st.session_state["transcribe_audio_task"] = loop.create_task(transcribe_audio())

    if "update_transcription_task" in st.session_state and st.session_state["update_transcription_task"]:
        st.session_state["update_transcription_task"].cancel()

    st.session_state["update_transcription_task"] = loop.run_until_complete(update_transcriptions())

# Q&A section
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Submit"):
    qna_state = QNAState()
    qna_state.query = question
    result = qna_app.invoke(qna_state)
    st.write(f"Answer: {result.answer}")
