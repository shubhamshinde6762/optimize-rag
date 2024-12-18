from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingGenerator:
    def __init__(self, chunkSize=512, chunkOverlap=50):
        self.textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunkSize, chunk_overlap=chunkOverlap
        )
        self.embeddingModel = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.chromaDB = None

    def processDocs(self, rawDocs):
        docChunks = self.textSplitter.split_text(" ".join(rawDocs))
        return docChunks

    def createEmbeddings(self, docChunks):
        self.chromaDB = Chroma.from_texts(
            texts=docChunks, embedding=self.embeddingModel, metadatas=[{"chunkId": i} for i in range(len(docChunks))]
        )
        return self.chromaDB

    def retrieveResults(self, query, fusionResults=5):
        return self.chromaDB.similarity_search_with_relevance_scores(query, k=fusionResults)
