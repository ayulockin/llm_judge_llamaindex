import os
import wandb

wandb.require("core")
import weave
import pathlib

from llama_index.core import PromptTemplate
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb


WANDB_PROJECT = "llamaindex-weave"

if wandb.run is None:
    run = wandb.init(project=WANDB_PROJECT)
else:
    run = wandb.run

weave.init(f"{WANDB_PROJECT}")

documents_artifact = run.use_artifact(
    f"{WANDB_PROJECT}/wandb_docs:latest", type="dataset"
)
data_dir = "data/wandb_docs"

docs_dir = documents_artifact.download(data_dir)

docs_dir = pathlib.Path(docs_dir)
docs_files = sorted(docs_dir.rglob("*.md"))
print(f"Number of files: {len(docs_files)}\n")

SYSTEM_PROMPT_TEMPLATE = """
Answer to the following question about W&B. Provide an helful and complete answer based only on the provided contexts.

User Query: {query_str}
Context: {context_str}
Answer: 
"""


class SimpleRAGPipeline(weave.Model):
    chat_llm: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    similarity_top_k: int = 2
    chunk_size: int = 512
    chunk_overlap: int = 128
    prompt_template: str = SYSTEM_PROMPT_TEMPLATE
    query_engine: RetrieverQueryEngine = None

    def _get_llm(self):
        return OpenAI(
            model=self.chat_llm,
            temperature=0.0,
            max_tokens=4096,
        )

    def _get_embedding_model(self):
        return OpenAIEmbedding(model=self.embedding_model)

    def _get_text_qa_template(self):
        return PromptTemplate(self.prompt_template)

    def _load_documents_and_chunk(self, files: pathlib.PosixPath):
        documents = []
        for file in files:
            content = file.read_text()
            documents.append(
                Document(
                    text=content,
                    metadata={
                        "source": str(file.relative_to(docs_dir)),
                        "raw_tokens": len(content.split()),
                    },
                )
            )
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    def _download_chroma_index(self):
        index_artifact = run.use_artifact(
            f"{WANDB_PROJECT}/chroma_index:latest", type="index"
        )

        self.embedding_model = index_artifact.metadata["openai_embedding_model"]
        index_dir = index_artifact.download("data/chroma_db")
        return index_dir

    def _setup_chroma_vectorstore(self, index_dir: str):
        db = chromadb.PersistentClient(path=index_dir)
        chroma_collection = db.get_or_create_collection("simple_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store

    def _create_vector_index(self, nodes):
        try:
            index_dir = self._download_chroma_index()
            print(f"Downloaded Chroma Index to: {index_dir}")
            vector_store = self._setup_chroma_vectorstore(index_dir)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self._get_embedding_model(),
            )
        except:
            print("Failed to download Chroma Index. Creating new index.")
            vector_store = self._create_chroma_index(index_dir="data/chroma_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes,
                embed_model=self._get_embedding_model(),
                show_progress=True,
                insert_batch_size=128,
                storage_context=storage_context,
            )

        return index

    def _get_retriever(self, index):
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.similarity_top_k,
        )
        return retriever

    def _get_response_synthesizer(self):
        llm = self._get_llm()
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="compact",
            text_qa_template=self._get_text_qa_template(),
        )
        return response_synthesizer

    def build_query_engine(self):
        nodes = self._load_documents_and_chunk(docs_files)
        index = self._create_vector_index(nodes)
        retriever = self._get_retriever(index)
        response_synthesizer = self._get_response_synthesizer()

        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    @weave.op()
    def predict(self, question: str):
        response = self.query_engine.query(question)
        return {
            "response": response.response,
            "context_str": "\n-----------\n".join(
                [
                    f"Source ID: {source_node.node.node_id}\n\n{source_node.node.text}"
                    for source_node in response.source_nodes
                ]
            )
            + "\n-----------\n",
        }


if __name__ == "__main__":
    rag_pipeline = SimpleRAGPipeline()
    rag_pipeline.build_query_engine()

    response = rag_pipeline.predict("What is Weigths and Biases?")
    print(response["response"])
