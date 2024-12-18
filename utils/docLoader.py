from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

class DocumentLoader:
    def loadDocs(self, filePath):
        if filePath.endswith(".pdf"):
            loader = PyPDFLoader(filePath)
        elif filePath.endswith(".docx"):
            loader = Docx2txtLoader(filePath)
        elif filePath.endswith(".txt"):
            loader = TextLoader(filePath)
        else:
            raise ValueError("Unsupported file format")
        documents = loader.load()
        return [doc.page_content for doc in documents]
