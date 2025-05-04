import asyncio
import math
import os
import time
import uuid
from typing import List, Dict, Tuple

import nest_asyncio
import sounddevice as sd
import streamlit as st
from deepgram import DeepgramClientOptions, DeepgramClient, LiveTranscriptionEvents, LiveOptions
from deepgram.utils import verboselogs
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langgraph.graph import StateGraph
from pydantic import BaseModel

nest_asyncio.apply()

if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state["event_loop"] = loop

loop = st.session_state["event_loop"]
asyncio.set_event_loop(loop)

if 'transcriptions' not in st.session_state:
    st.session_state["transcriptions"] = []

transriptions = st.session_state["transcriptions"]

st.sidebar.header("Configuration")
deepgram_api_key = st.sidebar.text_input("Deepgram API Key", type="password", value=os.getenv("DEEPGRAM_API_KEY"))
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY"))
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY"))
zilliz_uri = st.sidebar.text_input("Zilliz URI", value=os.getenv("ZILLIZ_URI"))
zilliz_token = st.sidebar.text_input("Zilliz Token", type="password", value=os.getenv("ZILLIZ_TOKEN"))

devices = sd.query_devices()
input_devices = [f"{d['name']} ({d['max_input_channels']} in, {d['max_output_channels']} out)" for d in devices]
selected_device = st.sidebar.selectbox("Select Audio Input Device", input_devices)

languages = ["en", "id"]
selected_language = st.sidebar.selectbox("Select Language", languages)

citation_limit = st.sidebar.number_input("Citation Limit", value=15)

st.title("transcriber-rag")

st.header("Transcription History")
transcription_container = st.empty()


async def transcribe_audio():
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

    dtype = 'int16'
    chunk = 1024 * 8
    sample_rate = 16000
    channels = device_info['max_input_channels']
    if channels == 0:
        channels = device_info['max_output_channels']

    config = DeepgramClientOptions(verbose=verboselogs.DEBUG, options={"keepalive": "true"})
    deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=config)

    deepgram_options: LiveOptions = LiveOptions(
        diarize=True,
        language=selected_language,
        model="nova-3",
        no_delay=True,
        encoding="linear16",
        sample_rate=sample_rate,
        smart_format=True,
    )

    connection = deepgram_client.listen.asyncwebsocket.v("1")

    async def on_message(self, result, **kwargs):
        transriptions.append(transcription)
        print("transriptions 1")
        print(transriptions)
        ingestion_state = IngestionState()
        ingestion_state.chunk = transcription
        ingestion_app.invoke(ingestion_state)

    async def on_error(self, error, **kwargs):
        print(f"Handled Error: {error}")

    connection.on(LiveTranscriptionEvents.Transcript, on_message)
    connection.on(LiveTranscriptionEvents.Error, on_error)

    await connection.start(options=deepgram_options)

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        audio_data = indata.tobytes()
        loop.run_until_complete(connection.send(audio_data))

    stream = sd.InputStream(
        samplerate=sample_rate,
        device=device_idx,
        channels=channels,
        callback=audio_callback,
        dtype=dtype,
        blocksize=chunk,
    )
    stream.start()


def update_transcriptions():
    while True:
        with transcription_container:
            print("transriptions 2")
            print(transriptions)
            for index, transcription in enumerate(transriptions):
                start = transcription["start"]
                end = start + transcription["duration"]
                alternatives = transcription.get("channel", {}).get("alternatives", [{}])[0]
                transcript = alternatives.get("transcript", "")
                words = alternatives.get("words", [])
                language = transcription.get("metadata", {}).get("language", "?")

                # attaching speaker diarization below transcript words
                transcript_words = []
                transcript_speakers = []
                for word in words:
                    speaker = f"{word.get('speaker', '?')}"
                    suffix = " " * (int(math.fabs(len(word["punctuated_word"]) - len(speaker))))
                    transcript_speaker = speaker if len(speaker) >= len(word["punctuated_word"]) else speaker + suffix
                    transcript_speakers.append(transcript_speaker)
                    if len(word["punctuated_word"]) >= len(speaker):
                        transcript_word = word["punctuated_word"]
                    else:
                        transcript_word = word["punctuated_word"] + suffix
                    transcript_words.append(transcript_word)

                separator = ","
                prefix = "- "
                subtitle = (
                    f"{index + 1}\n"
                    f"{subtitle_time_formatter(start, separator)} --> "
                    f"{subtitle_time_formatter(end, separator)}\n"
                    f"{prefix}{' '.join(transcript_words)}\n"
                    f"{prefix}{' '.join(transcript_speakers)}\n"
                    f"{prefix}{language}\n\n"
                )
                st.write(subtitle)
        time.sleep(1)


if st.sidebar.button("Configure"):
    os.environ["DEEPGRAM_API_KEY"] = deepgram_api_key
    os.environ["COHERE_API_KEY"] = cohere_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key
    st.sidebar.success("Configured!")
    loop.run_until_complete(transcribe_audio())
    update_transcriptions()


class IngestionState(BaseModel):
    chunk: Dict = None


async def store_chunk(state: IngestionState):
    embedder = CohereEmbeddings(
        model="embed-v4.0",
        cohere_api_key=cohere_api_key,
    )
    vector_store = Milvus(
        embedding_function=embedder,
        connection_args={"uri": zilliz_uri, "token": zilliz_token},
        collection_name="transcriber-rag"
    )
    document = Document(
        id=uuid.uuid4(),
        page_content=state.chunk['transcription'],
        metadata=state.chunk
    )
    await vector_store.aadd_documents(
        ids=[document.id],
        documents=[document]
    )
    return state


ingestion_graph = StateGraph(IngestionState)
ingestion_graph.add_node("store", store_chunk)
ingestion_graph.set_entry_point("store")
ingestion_graph.set_finish_point("store")
ingestion_app = ingestion_graph.compile()


class QNAState(BaseModel):
    query: str = None
    embedding: List = None
    documents: List[Tuple[Document, float]] = None
    answer: str = None


async def retrieve_documents(state: QNAState):
    embedder = CohereEmbeddings(
        model="embed-v4.0",
        cohere_api_key=cohere_api_key,
    )
    vector_store = Milvus(
        embedding_function=embedder,
        connection_args={"uri": zilliz_uri, "token": zilliz_token},
        collection_name="transcriber-rag"
    )
    state.documents = await vector_store.asimilarity_search_with_score(query=state.query, k=citation_limit)
    return state


def generate_answer(state: QNAState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        google_api_key=google_api_key,
    )
    citations = []
    for index, (document, score) in enumerate(state.documents):
        citation = {
            "type": "text",
            "text": f"""
                <citation_{index + 1}>
                {document.page_content}
                </citation_{index + 1}>
                """
        }
        citations.append(citation)

    message_content = [
        {
            "type": "text",
            "text": f"""
                <instruction>
                Answer the following query using ONLY the provided citations.
                You MUST include the citations in your answer, i.e., [1, 2, 3, etc.].
                If you cannot find the answer in the citations, respond with "I don't have enough information to answer that query."
                Do not make up any information.
                </instruction>
                """
        },
        {
            "type": "text",
            "text": f"""
                <query>
                {state.query}
                </query>
                """
        },
        {
            "type": "text",
            "text": "<citations>"
        },
        *citations,
        {
            "type": "text",
            "text": "</citations>"
        },
    ]
    message = HumanMessage(content=message_content)
    response = llm.invoke([message])
    state.answer = response.content
    return state


qna_graph = StateGraph(QNAState)
qna_graph.add_node("retrieve", retrieve_documents)
qna_graph.add_node("generate", generate_answer)
qna_graph.add_edge("retrieve", "generate")
qna_graph.set_entry_point("retrieve")
qna_graph.set_finish_point("generate")
qna_app = qna_graph.compile()


def subtitle_time_formatter(seconds, separator):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Submit"):
    qna_state = QNAState()
    qna_state.query = question
    result = qna_app.invoke(qna_state)
    st.write(f"Answer: {result.answer}")
