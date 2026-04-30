import chromadb
import time
from utils.logger import logger

class DexterMemory:
    def __init__(self, persist_directory="./memory_db"):
        logger.info("Waking up Dexter's Long-Term Memory (ChromaDB)...")
        # ChromaDB creates a local folder to store vector embeddings
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Collection is like a table in a database
        self.collection = self.client.get_or_create_collection(name="dexter_memory")
        logger.info(f"Long-Term Memory loaded. {self.collection.count()} memories on file.")

    def remember(self, text: str, role: str = "user"):
        """Stores an interaction into the vector database for future recall."""
        try:
            doc_id = f"msg_{int(time.time() * 1000)}"
            self.collection.add(
                documents=[text],
                metadatas=[{"role": role, "timestamp": time.time()}],
                ids=[doc_id]
            )
            logger.debug(f"Saved memory: {text[:60]}...")
        except Exception as e:
            logger.error(f"Memory save error (non-fatal): {e}")

    def recall_context(self, query: str, n_results: int = 3) -> str:
        """
        Searches the vector database for relevant past memories.
        Injects them into the LLM prompt for context. Costs 0 API tokens.
        """
        try:
            if self.collection.count() == 0:
                return ""
                
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            memories = results['documents'][0]
            if not memories:
                return ""
                
            context = "PAST RELEVANT MEMORIES (Use these to understand the user's context):\n" 
            context += "\n".join([f"- {m}" for m in memories])
            return context
            
        except Exception as e:
            logger.error(f"Memory recall error: {e}")
            return ""

    def get_memory_count(self) -> int:
        """Returns the total number of stored memories."""
        return self.collection.count()
