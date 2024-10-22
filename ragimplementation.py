import os
import ollama
import pdfplumber
import fitz
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedModel = 'nomic-embed-text:latest'
llmmodel = 'llama3.1:8b'
sdir = "c:\\codes\\rag\\"
def extract_pdf_content(filePath):
    pageList = []
    with pdfplumber.open(filePath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text_content = page.extract_text()  # Extract page text
            # Extract tables (if any)
            tables = page.extract_tables()
            table_content = ''
            if tables:
                for table in tables:
                    # Handle None values in table rows
                    table_content += "\n".join(
                        ["\t".join([str(cell) if cell is not None else '' for cell in row]) for row in table]
                    ) + "\n"
            page_content = text_content + "\n" + table_content  # Append tables to text
            page_metadata = {'source': os.path.basename(filePath), 'page': page_num + 1}
            pageList.append({'content': page_content, 'metadata': page_metadata})
    return pageList


# Extract images from PDF using PyMuPDF
def extract_images_from_pdf(filePath):
    doc = fitz.open(filePath)
    images = []
    for page_num in range(len(doc)):
        for img in doc.get_page_images(page_num):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                'image': base_image['image'],
                'image_metadata': {'page': page_num + 1, 'source': os.path.basename(filePath)}
            })
    return images

# Main PDF processing loop
fileNames = [fileName for fileName in os.listdir(sdir) if fileName.endswith('.pdf')]
pageList = []

for file in fileNames:
    filePath = os.path.join(sdir, file)
    
    # Extract text and table data
    page_data = extract_pdf_content(filePath)
    pageList.extend(page_data)

    # Extract images (if needed)
    image_data = extract_images_from_pdf(filePath)
    # You can save or process image data separately. For this example, we're only extracting them.

# Improved text splitting with metadata for better context
from langchain.text_splitter import RecursiveCharacterTextSplitter

textSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
textsplits = []
textSplitsMetadata = []

for page in pageList:
    split = textSplitter.split_text(page['content'])
    textsplits.extend(split)
    PM = page['metadata']
    for i in range(len(split)):
        textSplitsMetadata.append(PM)

# Embed each text chunk
embeddings = []
for split in textsplits:
    embedding = ollama.embeddings(model=embedModel, prompt=split)
    embeddings.append(embedding)

# Create Document objects for embedding
DocumentObjectList = [Document(page_content=data[0], metadata=data[1]) for data in zip(textsplits, textSplitsMetadata)]
vectorDataBase = Chroma.from_documents(
    documents=DocumentObjectList, 
    embedding=OllamaEmbeddings(model=embedModel, show_progress=True)
)

# Load LLM model
model = ChatOllama(model=llmmodel)

# Optimize query rewriting prompt
queryPrompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an AI language model assistant. Your task is to generate alternative versions of the given user question to retrieve documents more effectively from a vector database. 
    Ensure the alternative questions are rephrased to account for different perspectives without diverging from the topic.
    Original question: {question}
    """
)

retriever = MultiQueryRetriever.from_llm(
    vectorDataBase.as_retriever(),
    ChatOllama(model=llmmodel),
    prompt=queryPrompt
)

# Refined prompt to avoid using LLM knowledge unless absolutely necessary
templateRAG = """
Based only on the following context, answer the question as accurately as possible:
{context}
Question: {question}

If you cannot find the answer in the context, indicate that the information isn't available in the provided documents. Do not provide information from outside the context.
"""

# Create the chain to execute RAG process
prompt = ChatPromptTemplate.from_template(templateRAG)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

# Ask question
question = '''What is torqe force and what is Drag force'''
response = chain.invoke(question)

# Output the result
print(response.content)

# Save response to a file
with open('output.txt', 'w', encoding="utf-8") as text_file:
    text_file.write(response.content)
'''fileNames = [fileName for fileName in os.listdir(sdir) if fileName.endswith('.pdf')]
pageList=[]

for file in fileNames:
    filePath = os.path.join(sdir,file)
    loader = PyPDFLoader(file_path = filePath)
    pages = loader.load()
    pageList.extend(pages)

textSplitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50, add_start_index= True)

textsplits=[]
textSplitsMetadata=[]

for page in pageList:
    split = textSplitter.split_text(page.page_content)
    textsplits.extend(split)
    PM = page.metadata
    for i in range(len(split)):
        textSplitsMetadata.append(PM)

embeddings = []

for split in textsplits:
    embedding = ollama.embeddings(model=embedModel,prompt=split)
    embeddings.append(embedding)

DocumentObjectList = [Document(page_content=data[0], metadata = data[1]) for data in zip(textsplits,textSplitsMetadata)]
vectorDataBase = Chroma.from_documents(documents= DocumentObjectList,embedding= OllamaEmbeddings(model=embedModel, show_progress= True),)


model = ChatOllama(model=llmmodel)

queryPrompt = PromptTemplate(input_variables=["question"],
                             template="""You are an AI language model assisant. Your task is to generate different versions of the given user question to retrive documents from a vector database. 
                             Ensure the alternative questions are rephrased to account for different perspectives without diverging from the topic.
                             Original question: {question}""",)

retriever = MultiQueryRetriever.from_llm(vectorDataBase.as_retriever(),
                                         ChatOllama(model=llmmodel),
                                         prompt=queryPrompt)

templateRAG = """First try to answer the question based ONLY on the following context:
{context} Question:{question} and if you cannot answer then use LLM knowledge to help
If you cannot find the answer in the context, indicate that the information isn't available in the provided documents. Do not provide information from outside the context."""

prompt = ChatPromptTemplate.from_template(templateRAG)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

question = """Can you explain what is deeplearning? What are topics in it?"""


response = chain.invoke(question)

print(response.content)

with open('output.txt','w',encoding="utf-8") as text_file:
          text_file.write(response.content)'''