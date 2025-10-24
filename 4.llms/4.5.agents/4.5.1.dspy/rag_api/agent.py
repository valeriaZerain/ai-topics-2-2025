import wikipediaapi
import os
from dotenv import load_dotenv
import dspy
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext,
    load_index_from_storage,
    Document
)
load_dotenv()

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

PERSIST_DIR = "lyrics_index"

def get_artist_documents(filename: str) -> list[Document]:
    with open(filename) as file:
        data = file.read()
    songs = data.split("===")
    artist = songs.pop(0).strip()
    
    documents = [
        Document(
            text=song,
            metadata={
                "category":"music",
                "artist": artist,
            }
        )
        for song in songs
    ]    
    return documents

ARTIST_FILES = [
    "8988_Kjarkas.txt",
    "8989_SaviaAndina.txt",
    "8990_GrupoMarka.txt",
    "8991_RicoRico.txt",
    "8992_SayaAfroBoliviana.txt",
]

if not os.path.exists(PERSIST_DIR):
    all_documents = []
    
    for filename in ARTIST_FILES:
        docs = get_artist_documents(filename)
        all_documents.extend(docs)
    
    index = VectorStoreIndex.from_documents(all_documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever()

def get_artist_wikipedia_info(artist: str) -> str:
    wiki = wikipediaapi.Wikipedia("Test Wikipedia", "es")
    page = wiki.page(artist)
    if not page.exists():
        return f"Wikipedia page for {artist} does not exist"
    return f"{page.summary}\n{page.sections[0]}\n{page.sections[1]}"


def get_artist_relevant_song_lyrics(topic: str) -> list[str]:
    nodes = retriever.retrieve(topic)
    return [node.text for node in nodes]


class ContextQA(dspy.Signature):
    # context: str = dspy.InputField(desc="Context for the response, the context is composed of song lyrics")
    history: dspy.History = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Should be detailed, outputing excerpts from the original context")


class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ReAct(
            signature=ContextQA, 
            tools=[get_artist_wikipedia_info, get_artist_relevant_song_lyrics],
            max_iters=4
        )
        self.refine_question = dspy.Predict("question -> main_topic")
        self.chat_history = dspy.History(messages=[])

    def forward(self, question):
        outputs = self.respond(history=self.chat_history, question=question)
        self.chat_history.messages.append({"question": question, **outputs})
        return outputs.answer