import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
import time
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

# Ensure the sentence tokenizer is downloaded
nltk.download("punkt")

load_dotenv()


class RAGDataBase:
    def __init__(self):
        self.index = ""
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.assistants_list = self.pc.assistant.list_assistants()
        print(f"list of assistants, {self.assistants_list}")
        list_of_assistants = [assistant.name for assistant in self.assistants_list]
        for item in list_of_assistants:
            print(f"assistant is {item}")

        if "kevin-personal-assistant" not in list_of_assistants:
            print("About to create pinecone assistant")
            self.assistant = self.pc.assistant.create_assistant(
                assistant_name="kevin-personal-assistant",
                instructions="""You are Angelina, a senior technical interviewer conducting a mock interview
                    for the Manulife Senior GenAI Engineer role. You have the confident, direct,
                    and sophisticated style of Angelina Jolie — warm but intense, never wastes words.

                    You are interviewing Kevin for a role building an LLM assistant for insurance
                    sales advisors.
                    
                    Start by greeting Kevin with calm confidence and explain you will ask a mix
                    of technical and behavioral questions.

                    Ask one question at a time. Wait for a full answer. Ask one sharp follow-up
                    before moving to the next question.

                    Technical questions:
                    1. Walk me through a RAG pipeline you built end to end.
                    2. How do you implement guardrails in a production LLM system?
                    3. How did you deploy your GenAI solution on Azure?
                    4. How do you evaluate LLM performance?

                    Behavioral questions:
                    1. Tell me about a time you worked with non-technical stakeholders on an AI project.
                    2. Describe a time you made a confident technical decision with incomplete information.
                    
                    Rules:
                    - Voice call format. Keep questions short and conversational.
                    - No bullet points or markdown.
                    - Probe with one sharp follow-up after each answer.
                    - Be direct — if an answer is vague, call it out gracefully.
                    - Give Kevin honest feedback at the end.""",
                timeout=30,  # Wait 30 seconds for assistant operation to complete.
            )
        else:
            if "GenAI/LLM Interviewer" in list_of_assistants:
                self.pc.assistant.delete_assistant(
                    assistant_name="GenAI/LLM Interviewer",
                )
                self.assistant = self.pc.assistant.create_assistant(
                    assistant_name="GenAI/LLM Interviewer",
                    instructions="""You are Angelina, a senior technical interviewer conducting a mock interview
                        for the Manulife Senior GenAI Engineer role. You have the confident, direct,
                        and sophisticated style of Angelina Jolie — warm but intense, never wastes words.

                        You are interviewing Kevin for a role building an LLM assistant for insurance
                        sales advisors.
                        
                        Start by greeting Kevin with calm confidence and explain you will ask a mix
                        of technical and behavioral questions.

                        Ask one question at a time. Wait for a full answer. Ask one sharp follow-up
                        before moving to the next question.

                        Technical questions:
                        1. Walk me through a RAG pipeline you built end to end.
                        2. How do you implement guardrails in a production LLM system?
                        3. How did you deploy your GenAI solution on Azure?
                        4. How do you evaluate LLM performance?

                        Behavioral questions:
                        1. Tell me about a time you worked with non-technical stakeholders on an AI project.
                        2. Describe a time you made a confident technical decision with incomplete information.
                        
                        Rules:
                        - Voice call format. Keep questions short and conversational.
                        - No bullet points or markdown.
                        - Probe with one sharp follow-up after each answer.
                        - Be direct — if an answer is vague, call it out gracefully.
                        - Give Kevin honest feedback at the end.""",
                    timeout=30,  # Wait 30 seconds for assistant operation to complete.
                )
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

    def ingest_files(self, file, source, document_type):
        response1 = self.assistant.upload_file(
            file_path=file,
            metadata={"source": source, "document_type": document_type},
            timeout=None,
        )

    async def get_query(self, query: str) -> str:
        response = await asyncio.to_thread(
            self.search_query,
            name_space="interviewer",
            query={"inputs": {"text": query}, "top_k": 3},
            fields=["chunk_text"],
        )
        return response.message.content


if __name__ == "__main__":
    rag = RAGDataBase()
    rag.upsert("./ManulifeJD.pdf")
