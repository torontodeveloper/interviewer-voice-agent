import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

nltk.download("punkt", quiet=True)

load_dotenv()


class RAGDataBase:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.connect()

    def connect(self):
        index_name = "interviewer"
        if not self.pc.has_index(index_name):
            index = self.pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"},
                },
            )
            self.index = index
        else:
            self.index = self.pc.Index(index_name)
        print("Type of Index", self.pc.describe_index("interviewer"))

    async def search_query(self, query):
        chunks = await asyncio.to_thread(
            self.index.search,
            namespace="interviewer",
            inputs={
                "text": query,
            },
            top_k=5,
            fields=["chunk_text"],
        )
        return "".join([hit.fields["chunk_text"] for hit in chunks.result.hits])

    def chunk_with_nltk(self, text, max_sentences_per_chunk=5):
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + max_sentences_per_chunk])
            chunks.append(chunk)
        return chunks

    def is_ingested(self, namespace):
        stats = self.index.describe_index_stats()
        return namespace in (stats.namespaces or {})

    def upsert(self, file):
        reader = PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
        chunks = self.chunk_with_nltk(text)
        formatted_chunks = []
        for index, chunk in enumerate(chunks):
            formatted_chunk = {"id": f"chunk_{index}", "chunk_text": chunk}
            formatted_chunks.append(formatted_chunk)
        result = self.index.upsert_records(
            namespace="interviewer",
            records=formatted_chunks,
        )



if __name__ == "__main__":
    rag = RAGDataBase()
    rag.upsert("./ManulifeJD.pdf")
