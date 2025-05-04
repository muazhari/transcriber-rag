import asyncio
import math
import os
import uuid
from typing import List, Tuple

import nest_asyncio
import pyaudiowpatch as pyaudio
import streamlit as st
from deepgram import DeepgramClientOptions, DeepgramClient, LiveTranscriptionEvents, LiveOptions, LiveResultResponse, \
    AsyncListenWebSocketClient
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


def get_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    devices = []

    for loopback in p.get_loopback_device_info_generator():
        devices.append(loopback)

    for i in range(0, info.get('deviceCount')):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            devices.append(p.get_device_info_by_host_api_device_index(0, i))

    return devices


if 'transcriptions' not in st.session_state:
    st.session_state["transcriptions"] = []

transcriptions: List[LiveResultResponse] = st.session_state["transcriptions"]

st.sidebar.header("Configuration")
deepgram_api_key = st.sidebar.text_input("Deepgram API Key", type="password", value=os.getenv("DEEPGRAM_API_KEY"))
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY"))
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY"))
zilliz_uri = st.sidebar.text_input("Zilliz URI", value=os.getenv("ZILLIZ_URI"))
zilliz_token = st.sidebar.text_input("Zilliz Token", type="password", value=os.getenv("ZILLIZ_TOKEN"))

devices = get_audio_devices()
input_devices = [f"{device['name']} ({device['index']})" for device in devices]
selected_device = st.sidebar.selectbox("Select Audio Input Device", input_devices)

languages = ["en", "id"]
selected_language = st.sidebar.selectbox("Select Language", languages)

citation_limit = st.sidebar.number_input("Citation Limit", value=15)

if "embedder" not in st.session_state:
    embedder = CohereEmbeddings(
        model="embed-v4.0",
        cohere_api_key=cohere_api_key,
    )
    st.session_state["embedder"] = embedder

embedder = st.session_state["embedder"]

if "vector_store" not in st.session_state:
    vector_store = Milvus(
        embedding_function=embedder,
        connection_args={"uri": zilliz_uri, "token": zilliz_token},
        collection_name="transcriber_rag",
        enable_dynamic_field=True,
        index_params={"metric_type": "COSINE"},
        search_params={"metric_type": "COSINE"},
    )
    st.session_state["vector_store"] = vector_store

vector_store = st.session_state["vector_store"]

st.title("transcriber-rag")
status = st.empty()


class IngestionState(BaseModel):
    chunk: LiveResultResponse = None


async def store_chunk(state: IngestionState):
    document = Document(
        id=uuid.uuid4(),
        page_content=state.chunk.channel.alternatives[0].transcript,
        metadata=state.chunk.to_dict()
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
    state.documents = await vector_store.asimilarity_search_with_score(query=state.query, k=citation_limit)
    return state


async def generate_answer(state: QNAState):
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


@st.dialog(title="Citation Details", width="large")
def citation_details(document: Document):
    st.write(document.model_dump())


st.header("Ask a Question")
question = st.text_area("Enter your question:")
if st.button("Submit"):
    qna_state = QNAState()
    qna_state.query = question
    result = loop.run_until_complete(qna_app.ainvoke(qna_state))
    st.write("**Answer:**")
    st.write(result["answer"])
    st.write("**Citations:**")
    for index, (document, score) in enumerate(result["documents"]):
        if st.button(label=f"Citation {index + 1}: {score}"):
            citation_details(document)
        st.text(document.page_content)

st.header("Transcriptions")
transcription_container = st.empty()


async def transcribe_audio():
    device_info = None
    for device in devices:
        if selected_device == f"{device['name']} ({device['index']})":
            device_info = device
            break

    if device_info is None:
        st.error("Selected audio device not found.")
        return

    dtype = 'int16'
    format = pyaudio.paInt16
    chunk = 1024 * 16
    sample_rate = int(device_info["defaultSampleRate"])
    channels = device_info["maxInputChannels"]
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

    async def on_message(self, result: LiveResultResponse, **kwargs):
        if len(result.channel.alternatives) == 0:
            return

        if result.channel.alternatives[0].transcript == "":
            return

        transcriptions.append(result)
        ingestion_state = IngestionState()
        ingestion_state.chunk = result
        await ingestion_app.ainvoke(ingestion_state)

    async def on_error(self, error, **kwargs):
        print(f"Handled Error: {error}")

    connection.on(LiveTranscriptionEvents.Transcript, on_message)
    connection.on(LiveTranscriptionEvents.Error, on_error)

    await connection.start(options=deepgram_options)

    def audio_callback(input_data, frames, time, status):
        loop.run_until_complete(connection.send(input_data))
        return (input_data, pyaudio.paContinue)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=device_info["index"],
        frames_per_buffer=chunk,
        stream_callback=audio_callback,
    )
    stream.start_stream()

    st.session_state["audio_stream"] = stream
    st.session_state["deepgram_connection"] = connection


def subtitle_time_formatter(seconds, separator):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


async def update_transcriptions():
    while True:
        if st.session_state.get("stop", False):
            break

        with transcription_container:
            subtitles = []
            for index, transcription in enumerate(transcriptions):
                start = transcription.start
                end = start + transcription.duration
                alternative = transcription.channel.alternatives[0]
                words = alternative.words

                # attaching speaker diarization below transcript words
                transcript_words = []
                transcript_speakers = []
                for word in words:
                    speaker = str(word.speaker)
                    suffix = "   " * (int(math.fabs(len(word.punctuated_word) - len(speaker))))
                    transcript_speaker = speaker if len(speaker) >= len(word.punctuated_word) else speaker + suffix
                    transcript_speakers.append(transcript_speaker)
                    if len(word["punctuated_word"]) >= len(speaker):
                        transcript_word = word["punctuated_word"]
                    else:
                        transcript_word = word["punctuated_word"] + suffix
                    transcript_words.append(transcript_word)

                separator = "."
                subtitle = (
                    f"{index + 1}\n"
                    f"{subtitle_time_formatter(start, separator)} - {subtitle_time_formatter(end, separator)}\n"
                    f"{' '.join(transcript_words)}\n"
                    f"{' '.join(transcript_speakers)}\n"
                )
                subtitles.insert(0, subtitle)

            st.text("\n".join(subtitles))
        await asyncio.sleep(1)


if st.sidebar.button("Start", use_container_width=True):
    st.session_state["stop"] = False
    loop.run_until_complete(transcribe_audio())
    loop.run_until_complete(update_transcriptions())

    with status:
        st.success("Started successfully!")

if st.sidebar.button("Stop", use_container_width=True):
    st.session_state["stop"] = True
    if "audio_stream" in st.session_state:
        audio_stream: pyaudio.Stream = st.session_state["audio_stream"]
        audio_stream.stop_stream()

    if "deepgram_connection" in st.session_state:
        deepgram_connection: AsyncListenWebSocketClient = st.session_state["deepgram_connection"]
        loop.run_until_complete(deepgram_connection.finish())

    with status:
        st.success("Stopped successfully!")

if st.sidebar.button("Reset", use_container_width=True):
    if vector_store.col:
        vector_store.col.drop()

    for key in st.session_state.keys():
        del st.session_state[key]

    with status:
        st.success("Reset successfully!")
