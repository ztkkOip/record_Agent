from datetime import datetime
from typing import List
import os
from pathlib import Path

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from record_agent.db_config import session_scope
from record_agent.models import User, ChatHistory, ChatSession

default_memory_config = {
    "maxlen": 30
}
default_model = "qwen3-max"
default_system_prompt = "你是ztkk,一个智能手账助手，语气尽量温和有力。只需要回答最新的问题,不用每次都介绍自己，需要的时候才介绍"

role_dict = {
    0: "user",
    1: "assistant",
    2: "system"
}
default_chroma_db_filePath = "../chroma_db"
default_topK = 4


def save_rag(filePath: str, collectionName: str) ->bool:
    """
    保存文档到向量数据库（支持文件和目录）

    Args:
        filePath: 文件路径或目录路径
        collectionName: 向量集合名称
        userId: 用户ID（可选，用于关联用户）

    Returns:
        str: 保存结果信息
    """
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        if os.path.isfile(filePath):
            file_ext = Path(filePath).suffix.lower()

            if file_ext in ['.pdf']:
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(filePath)
                    documents = loader.load()
                except ImportError:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(filePath)
                    documents = [type('Document', page_content=page.extract_text(), metadata={'source': filePath}) for page in reader.pages]

            elif file_ext in ['.docx']:
                try:
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(filePath)
                    documents = loader.load()
                except ImportError:
                    from docx import Document
                    doc = Document(filePath)
                    documents = [type('Document', page_content='\n'.join([para.text for para in doc.paragraphs]), metadata={'source': filePath})]

            else:
                loader = TextLoader(filePath, encoding='utf-8')
                documents = loader.load()

            texts = text_splitter.split_documents(documents)

            if len(texts) == 0:
                return "没有文档内容可保存"

            api_key = os.environ.get('DASHSCOPE_API_KEY')
            if api_key:
                embedding_function = DashScopeEmbeddings(dashscope_api_key=api_key)
            else:
                embedding_function = DashScopeEmbeddings()

            vs = Chroma(
                collection_name=collectionName,
                embedding_function=embedding_function,
                persist_directory=default_chroma_db_filePath
            )
            vs.add_documents(texts)

            print( f"文件已保存，共分割成 {len(texts)} 个文档块")
            return True

        elif os.path.isdir(filePath):
            all_documents = []
            file_count = 0

            for root, dirs, files in os.walk(filePath):
                for file in files:
                    file_ext = Path(file).suffix.lower()

                    if file_ext in ['.pdf', '.docx', '.txt', '.md', '.json', '.xml']:
                        try:
                            if file_ext == '.pdf':
                                from langchain_community.document_loaders import PyPDFLoader
                                loader = PyPDFLoader(os.path.join(root, file))
                                docs = loader.load()
                            elif file_ext == '.docx':
                                from langchain_community.document_loaders import Docx2txtLoader
                                loader = Docx2txtLoader(os.path.join(root, file))
                                docs = loader.load()
                            else:
                                from langchain_community.document_loaders import TextLoader
                                loader = TextLoader(os.path.join(root, file), encoding='utf-8')
                                docs = loader.load()

                            all_documents.extend(docs)
                            file_count += 1
                        except Exception:
                            continue

            if file_count == 0:
                print( "目录为空或没有支持的文件")
                return True

            texts = text_splitter.split_documents(all_documents)

            if len(texts) == 0:
                print( "没有文档内容可保存")
                return True

            embedding_function = DashScopeEmbeddings()

            vs = Chroma(
                collection_name=collectionName,
                embedding_function=embedding_function,
                persist_directory=default_chroma_db_filePath
            )
            vs.add_documents(texts)
            print(f"目录已保存，共处理 {file_count} 个文件，分割成 {len(texts)} 个文档块")
            return True
        else:
            return "路径不存在或不是有效的文件/目录"

    except Exception as e:
        print( f"处理失败: {str(e)}")
        return False

def get_rag(input:str,collectionName: str):
    embedding_function = DashScopeEmbeddings()
    vs = Chroma(
        collection_name=collectionName,
        embedding_function=embedding_function,
        persist_directory=default_chroma_db_filePath
    )
    res = []
    docs = vs.similarity_search(input,default_topK)
    for doc in docs:
        res.append(doc.page_content)
    return res


def get_history(userId: int, sessionId: int):
    with session_scope() as session:
        chatHistorys = session.query(ChatHistory).filter(
            ChatHistory.user_id == userId,
            ChatHistory.session_id == sessionId
        ).order_by(ChatHistory.create_time).all()
        if None == chatHistorys or len(chatHistorys) == 0:
            return []
        res = []
        for chatHistory in chatHistorys:
            res.append((role_dict[chatHistory.role], chatHistory.content))
        return res


def save_history(userId: int, sessionId: int, chatHistorys: List[ChatHistory]):
    with session_scope() as session:
        if None != sessionId:
            checkSession = session.query(ChatSession).filter(
                ChatSession.user_id == userId,
                ChatSession.id == sessionId
            ).first()
        if None == sessionId or checkSession == None:
            newSession = ChatSession(
                user_id=userId,
                create_time=datetime.now(),
                update_time=datetime.now(),
                title="新对话"
            )
            session.add(newSession)
            session.flush()
            sessionId = newSession.id
        for chatHistory in chatHistorys:
            chatHistory.session_id = sessionId
        session.add_all(chatHistorys)
        chatSession = session.query(ChatSession).filter(
            ChatSession.id == sessionId,
            ChatSession.user_id == userId
        ).first()
        chatSession.update_time = datetime.now()
        session.commit()
        return sessionId


def chat(model_name: str, input, sessionId: int = None, userId: int = None, system_prompt: str = None):
    if sessionId is None:
        with session_scope() as session:
            newSession = ChatSession(
                user_id=userId,
                create_time=datetime.now(),
                update_time=datetime.now(),
                title="新对话"
            )
            session.add(newSession)
            session.flush()
            sessionId = newSession.id

    if None == model_name:
        model_name = default_model

    try:
        model = ChatTongyi(model=model_name)
    except Exception:
        return {"sessionId": sessionId, "answer": "模型初始化失败"}
    else:
        if system_prompt is None:
            system_prompt = default_system_prompt

        prompt = PromptTemplate.from_template("{system_prompt}\n请根据历史记录{chat_history}和搜寻到的资料{docs}作为用户喜好的参考不用回答历史问题，回答用户提问{{input}}")
        print(prompt)
        chat_history = get_history(userId, sessionId)
        chat = prompt | model
        docs = get_rag(input,"pet")
        answer = chat.invoke({"system_prompt":system_prompt,"input": input, "chat_history": chat_history,"docs": docs})
        saveHistorys = []

        if type(input) == str:
            saveHistorys.append(ChatHistory(
                user_id=userId,
                session_id=sessionId,
                content=input,
                role=0,
                model=model_name,
                create_time=datetime.now(),
            ))
        saveHistorys.append(ChatHistory(
            user_id=userId,
            session_id=sessionId,
            content=answer.content,
            role=1,
            model=model_name,
            create_time=datetime.now(),
        ))
        sessionId = save_history(userId, sessionId, saveHistorys)
        return {
            "sessionId": sessionId,
            "answer": answer.content,
        }


if __name__ == "__main__":
    # save_rag("../test","pet")
    print( chat(None,"我又买了一条小狗",1,1))
    print(2)
    print(chat(None, "给这条边牧取个名字吧", 1, 1))
