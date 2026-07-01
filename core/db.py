import uuid
from abc import ABC, abstractmethod
import numpy as np

# Import configuration
import config

class BaseVectorStore(ABC):
    """
    Abstract interface for our Vector Database, defining essential RAG operations.
    """
    @abstractmethod
    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        """
        Embed and insert a list of text chunks with their associated metadata.
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        """
        Query the database with a question and return the top k matching document chunks.
        Returns a list of dicts, each with 'text' and 'metadata'.
        """
        pass

    @abstractmethod
    def get_count(self) -> int:
        """
        Returns the total number of text chunks currently indexed in this store.
        """
        pass


# ==============================================================================
# Embedding Generator Helper
# ==============================================================================
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates embedding vectors for a list of texts using the configured LLM provider.
    This avoids local heavy dependencies (like PyTorch or sentence-transformers).
    """
    if not texts:
        return []

    # Clean text to ensure no empty values
    texts = [t if t.strip() else "empty text chunk" for t in texts]

    if config.LLM_PROVIDER == "gemini":
        from google import genai
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing in your environment/config. Please set it in .env.")
        
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        
        # Google Gen AI SDK supports lists of texts
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=texts
        )
        
        # In the new SDK, the result contains a list of ContentEmbedding objects under `embeddings`
        # each has a `values` property containing the float list.
        if hasattr(response, 'embeddings'):
            return [emb.values for emb in response.embeddings]
        elif isinstance(response, dict) and 'embeddings' in response:
            return [emb['values'] for emb in response['embeddings']]
        else:
            # Fallback/alternative if format varies
            return [getattr(emb, 'values', emb) for emb in response]

    elif config.LLM_PROVIDER == "openai_compatible":
        from openai import OpenAI
        client = OpenAI(base_url=config.OPENAI_API_BASE, api_key=config.OPENAI_API_KEY or "no-key")
        
        response = client.embeddings.create(
            input=texts,
            model=config.OPENAI_EMBEDDING_MODEL_NAME
        )
        # OpenAI returns an array of data objects, each containing an embedding
        return [data.embedding for data in response.data]
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")


# ==============================================================================
# ChromaDB Local Implementation
# ==============================================================================
class ChromaStore(BaseVectorStore):
    def __init__(self):
        import chromadb
        # Initialize Persistent Local Client
        self.client = chromadb.PersistentClient(path=str(config.DB_DIR))
        # Create or fetch the collection
        # We handle embeddings manually, so we don't supply an embedding_function
        self.collection = self.client.get_or_create_collection(
            name="med_subspecialty_index"
        )

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        if not texts:
            return
            
        # 1. Generate Embeddings using our API client
        embeddings = get_embeddings(texts)
        
        # 2. Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # 3. Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        # 1. Embed query
        query_embeddings = get_embeddings([query])
        if not query_embeddings:
            return []
            
        # 2. Query Chroma
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=k
        )
        
        # 3. Format results into structured dictionary list
        formatted = []
        if results and 'documents' in results and results['documents']:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(documents)
            
            for doc, meta in zip(documents, metadatas):
                formatted.append({
                    "text": doc,
                    "metadata": meta
                })
        return formatted

    def get_count(self) -> int:
        return self.collection.count()


# ==============================================================================
# Pinecone Cloud Implementation
# ==============================================================================
class PineconeStore(BaseVectorStore):
    def __init__(self):
        from pinecone import Pinecone, ServerlessSpec
        
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing. Please set it in your .env file.")
            
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        
        # Determine embedding dimension based on provider
        dimension = 768 if config.LLM_PROVIDER == "gemini" else 1536  # Default dimensions
        if config.LLM_PROVIDER == "openai_compatible" and "nomic" in config.OPENAI_EMBEDDING_MODEL_NAME:
            dimension = 768  # nomic-embed-text standard dimension is 768

        # Create index if it does not exist
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index '{self.index_name}' (dimension={dimension})...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=config.PINECONE_ENVIRONMENT
                )
            )
            
        self.index = self.pc.Index(self.index_name)

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        if not texts:
            return
            
        # 1. Generate Embeddings
        embeddings = get_embeddings(texts)
        
        # 2. Package into Pinecone upsert format: list of tuples of (id, vector, metadata)
        # Note: metadata values in Pinecone must be simple types (str, int, float, bool, or list of str)
        upsert_data = []
        for i, (text, vector, meta) in enumerate(zip(texts, embeddings, metadatas)):
            chunk_id = str(uuid.uuid4())
            # Inject the actual text content directly into the metadata so we can retrieve it
            cleaned_meta = {k: v for k, v in meta.items() if v is not None}
            cleaned_meta["text"] = text
            upsert_data.append((chunk_id, vector, cleaned_meta))
            
        # 3. Upsert in batches of 100
        batch_size = 100
        for idx in range(0, len(upsert_data), batch_size):
            batch = upsert_data[idx:idx + batch_size]
            self.index.upsert(vectors=batch)

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        # 1. Embed query
        query_embeddings = get_embeddings([query])
        if not query_embeddings:
            return []
            
        # 2. Query Pinecone
        response = self.index.query(
            vector=query_embeddings[0],
            top_k=k,
            include_metadata=True
        )
        
        # 3. Format response
        formatted = []
        if 'matches' in response:
            for match in response['matches']:
                meta = match.get('metadata', {})
                # Extract text stored in metadata
                text = meta.pop('text', '')
                formatted.append({
                    "text": text,
                    "metadata": meta
                })
        return formatted

    def get_count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
            return stats.get('total_vector_count', 0)
        except Exception as e:
            print(f"Error checking Pinecone stats: {e}")
            return 0


# ==============================================================================
# Database Factory Loader
# ==============================================================================
def get_vector_store() -> BaseVectorStore:
    """
    Returns the initialized vector store configured in environment variables.
    """
    if config.VECTOR_DB_TYPE == "pinecone":
        return PineconeStore()
    else:
        return ChromaStore()
