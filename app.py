import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from threading import Event
from typing import List, Tuple

import nest_asyncio
import pyaudiowpatch as pyaudio
import streamlit as st
from deepgram import DeepgramClientOptions, DeepgramClient, LiveTranscriptionEvents, LiveOptions, LiveResultResponse
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

if "stop_event" not in st.session_state:
    stop_event = Event()
    stop_event.set()
    st.session_state["stop_event"] = stop_event

stop_event = st.session_state["stop_event"]


def get_audio_devices():
    p = pyaudio.PyAudio()
    devices = []

    for device in p.get_device_info_generator():
        if device["maxInputChannels"] > 0:
            devices.append(device)

    return devices


if 'transcriptions' not in st.session_state:
    st.session_state["transcriptions"] = []

transcriptions: List[LiveResultResponse] = st.session_state["transcriptions"]

st.sidebar.header("Configuration")
deepgram_api_key = st.sidebar.text_input("Deepgram API Key", type="password", value=os.getenv("DEEPGRAM_API_KEY"))
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY"))
google_api_key = st.sidebar.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY"))
zilliz_uri = st.sidebar.text_input("Zilliz URI", value=os.getenv("ZILLIZ_URI"))
zilliz_token = st.sidebar.text_input("Zilliz Token", type="password", value=os.getenv("ZILLIZ_TOKEN"))

devices = get_audio_devices()
input_devices = [f"{device['name']} ({device['index']})" for device in devices]
selected_device = st.sidebar.selectbox("Audio Device", input_devices)

languages = ["en", "id", "multi"]
selected_language = st.sidebar.selectbox("Language", languages)
collection_name = st.sidebar.text_input("Collection Name", value="transcriber_rag")
ingestion_batch_size = st.sidebar.number_input("Ingestion Batch Size", value=15)
citation_limit = st.sidebar.number_input("Citation Limit", value=15)

embedder = CohereEmbeddings(
    model="embed-v4.0",
    cohere_api_key=cohere_api_key,
)
vector_store = Milvus(
    embedding_function=embedder,
    connection_args={"uri": zilliz_uri, "token": zilliz_token},
    collection_name=collection_name,
    enable_dynamic_field=True,
    index_params={"metric_type": "COSINE"},
    search_params={"metric_type": "COSINE"},
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
)

st.title("transcriber-rag")
status = st.empty()


class IngestionState(BaseModel):
    chunks: List[LiveResultResponse] = None


async def store_chunk(state: IngestionState):
    document = Document(
        id=str(uuid.uuid4()),
        page_content="",
        metadata={
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "sources": [],
        }
    )

    for chunk in state.chunks:
        transcript = chunk.channel.alternatives[0].transcript
        document.page_content = f"{document.page_content} {transcript}"
        document.metadata["sources"].append(chunk.to_dict())

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
    state.documents = await vector_store.asimilarity_search_with_score(query=state.query, k=citation_limit)
    return state


async def generate_answer(state: QNAState):
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
    response = await llm.ainvoke([message])
    state.answer = response.content
    return state


qna_graph = StateGraph(QNAState)
qna_graph.add_node("retrieve", retrieve_documents)
qna_graph.add_node("generate", generate_answer)
qna_graph.add_edge("retrieve", "generate")
qna_graph.set_entry_point("retrieve")
qna_graph.set_finish_point("generate")
qna_app = qna_graph.compile()


async def transcribe_audio():
    device_info = None
    for device in devices:
        if selected_device == f"{device['name']} ({device['index']})":
            device_info = device
            break

    if device_info is None:
        st.error("Selected audio device not found.")
        return

    format = pyaudio.paInt16
    chunk = 8000
    sample_rate = 48000
    channels = 1
    deepgram_config = DeepgramClientOptions(verbose=verboselogs.DEBUG, options={"keepalive": "true"})
    deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=deepgram_config)

    deepgram_live_options: LiveOptions = LiveOptions(
        diarize=True,
        language=selected_language,
        model="nova-3" if selected_language in ["multi", "en"] else "nova-2",
        no_delay=True,
        smart_format=True,
        encoding="linear16",
        sample_rate=sample_rate,
        channels=channels,
    )
    deepgram_connection = deepgram_client.listen.asyncwebsocket.v("1")

    async def on_transcript(self, result: LiveResultResponse, *args, **kwargs):
        if (
                len(result.channel.alternatives) == 0
                or result.is_final is False
                or result.channel.alternatives[0].transcript == ""
        ):
            return

        transcriptions.append(result)

        if len(transcriptions) % ingestion_batch_size == 0:
            ingestion_state = IngestionState()
            ingestion_state.chunks = transcriptions[-ingestion_batch_size:]
            await ingestion_app.ainvoke(ingestion_state)

    async def on_error(self, error, *args, **kwargs):
        print(f"Handled Error: {error}")

    async def on_close(self, close, *args, **kwargs):
        print(f"Handled Close: {close}")

    deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    deepgram_connection.on(LiveTranscriptionEvents.Error, on_error)
    deepgram_connection.on(LiveTranscriptionEvents.Close, on_close)
    await deepgram_connection.start(options=deepgram_live_options)

    chunk_count = 0

    audio = pyaudio.PyAudio()

    def audio_callback(input_data, *args, **kwargs):
        nonlocal chunk_count

        if stop_event.is_set():
            deepgram_connection.finish()
            audio.terminate()
            return input_data, pyaudio.paComplete

        loop.run_until_complete(deepgram_connection.send(input_data))
        chunk_count += 1
        print(f"Sent audio chunk {chunk_count} to Deepgram with length {len(input_data)} bytes")
        return input_data, pyaudio.paContinue

    audio_stream = audio.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=device_info["index"],
        frames_per_buffer=chunk,
        stream_callback=audio_callback,
    )
    audio_stream.start_stream()


def subtitle_time_formatter(seconds, separator):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


if st.sidebar.button("Start", use_container_width=True):
    stop_event.clear()
    loop.run_until_complete(transcribe_audio())

    with status:
        st.success("Started successfully!")

if st.sidebar.button("Stop", use_container_width=True):
    stop_event.set()

    with status:
        st.success("Stopped successfully!")

if st.sidebar.button("Reset", use_container_width=True):
    if vector_store.col:
        vector_store.col.drop()

    for key in st.session_state.keys():
        del st.session_state[key]

    with status:
        st.success("Reset successfully!")


def transcribe_page():
    st.header("Transcribe")
    transcription_container = st.empty()

    while True:
        if stop_event.is_set():
            st.text("No transcriptions yet, please start it first.")
            break

        with transcription_container:
            subtitles = []

            for index, transcription in enumerate(transcriptions):
                start = transcription.start
                end = start + transcription.duration
                alternative = transcription.channel.alternatives[0]
                words = alternative.words

                # attaching speaker diarization below transcript words
                # Create aligned display of words and speakers using monospace formatting
                words_line = ""
                speakers_line = ""

                for i, word in enumerate(words):
                    padding = max(len(word.punctuated_word), len(str(word.speaker)))
                    formatted_word = f"{word.punctuated_word:<{padding}} "
                    formatted_speaker = f"{str(word.speaker):<{padding}} "

                    words_line += formatted_word
                    speakers_line += formatted_speaker

                separator = ","
                subtitle = (
                    f"{index + 1}\n"
                    f"{subtitle_time_formatter(start, separator)} - {subtitle_time_formatter(end, separator)}\n"
                    f"{words_line}\n"
                    f"{speakers_line}\n"
                )
                subtitles.insert(0, subtitle)

            if len(subtitles) == 0:
                st.spinner("Waiting for transcriptions...")
            else:
                # Use markdown with code block to ensure monospace font for alignment
                st.markdown("```\n" + "\n".join(subtitles) + "\n```")

        time.sleep(0.5)


@st.dialog(title="Citation Details", width="large")
def citation_details(document: Document):
    st.write(document.model_dump())


def qna_page():
    st.header("Question Answering")
    query = st.text_area("Enter your query:")
    if st.button("Submit"):
        qna_state = QNAState()
        qna_state.query = query
        result = loop.run_until_complete(qna_app.ainvoke(qna_state))
        st.session_state["qna_result"] = result

    if "qna_result" in st.session_state:
        result = st.session_state["qna_result"]
        st.write("**Answer:**")
        st.write(result["answer"])
        st.write("**Citations:**")
        for index, (document, score) in enumerate(result["documents"]):
            if st.button(label=f"Citation {index + 1}: {score}"):
                citation_details(document)
            st.text(document.page_content)


pages = {
    "Transcribe": transcribe_page,
    "Question Answering": qna_page,
}

selected_page = st.sidebar.selectbox("Page", list(pages.keys()), index=0)
pages[selected_page]()
