import json
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
    "maxlen": 20
}
default_model = "qwen3-max"
sum_model_name = "qwen3-max"
default_system_prompt = "你是ztkk,一个智能手账助手，语气尽量温和有力。只需要回答最新的问题,不用每次都介绍自己，需要的时候才介绍"

role_dict = {
    0: "user",
    1: "assistant",
    2: "system"
}
default_chroma_db_filePath = "../chroma_db"
default_topK = 4


def save_rag(filePath: str, collectionName: str) -> bool:
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

            print(f"文件已保存，共分割成 {len(texts)} 个文档块")
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
                print("目录为空或没有支持的文件")
                return True

            texts = text_splitter.split_documents(all_documents)

            if len(texts) == 0:
                print("没有文档内容可保存")
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
        print(f"处理失败: {str(e)}")
        return False


def get_rag(input: str, collectionName: str)->List[str]:
    embedding_function = DashScopeEmbeddings()
    vs = Chroma(
        collection_name=collectionName,
        embedding_function=embedding_function,
        persist_directory=default_chroma_db_filePath
    )
    res = []
    docs = vs.similarity_search(input, default_topK)
    for doc in docs:
        res.append(doc.page_content)
    return res


def get_history(userId: int, sessionId: int)->dict:
    with session_scope() as session:
        chatHistorys = session.query(ChatHistory).filter(
            ChatHistory.user_id == userId,
            ChatHistory.session_id == sessionId,
            ChatHistory.message_type == "text"
        ).order_by(ChatHistory.create_time).limit(default_memory_config['maxlen']).all()
        if None == chatHistorys or len(chatHistorys) == 0:
            return []
        history = []
        #获取摘要
        chatSummarry = session.query(ChatHistory).filter(
            ChatHistory.user_id == userId,
            ChatHistory.session_id == sessionId,
            ChatHistory.message_type == "summary"
        ).order_by(ChatHistory.create_time).first()
        if chatSummarry is not None:
            history.append((role_dict[1], chatSummarry.content))
            summarryJson = json.loads(chatSummarry.meta_data)
            lastIndex = summarryJson.get('lastIndex')
            count = 0
            for chatHistory in chatHistorys:
                if chatHistory.id > lastIndex:
                    count += 1
                history.append((role_dict[chatHistory.role], chatHistory.content))
            if count >= default_memory_config['maxlen']:
                sumFlag = True
            else:
                sumFlag = False
            return {
                "history": history,
                "sumFlag": sumFlag,
                "summary": chatSummarry.content
            }
        else:
            for chatHistory in chatHistorys:
                history.append((role_dict[chatHistory.role], chatHistory.content))

            return {
                "history": history,
                "sumFlag": True,
                "summary": ""
            }


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


def chat(model_name: str, input: str, sessionId: int = None, userId: int = None, system_prompt: str = None):
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
        yield {"sessionId": sessionId, "answer": "模型初始化失败", "done": True}
        return
    else:
        if system_prompt is None:
            system_prompt = default_system_prompt

        prompt = PromptTemplate.from_template("请根据历史记录{chat_history}和搜寻到的资料{docs}回答用户提问")
        print(prompt)
        history_dict = get_history(userId, sessionId)
        chat_history = history_dict["history"]
        chat_history.extend([
            (role_dict[0], input),
            (role_dict[2], system_prompt)
        ])
        chat = prompt | model
        docs = get_rag(input, "pet")

        # 保存用户问题
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

        full_answer = ""

        # 流式输出
        for chunk in chat.stream({"chat_history": chat_history, "docs": docs}):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_answer += content

            # 实时返回 chunk
            yield {
                "sessionId": sessionId,
                "answer": content,
                "done": False
            }

        # 流式结束后，保存完整的助手回答
        saveHistorys.append(ChatHistory(
            user_id=userId,
            session_id=sessionId,
            content=full_answer,
            role=1,
            model=model_name,
            create_time=datetime.now(),
        ))
        sessionId = save_history(userId, sessionId, saveHistorys)

        # 处理摘要 - 获取当前最后一条对话的 id 作为 lastIndex
        last_chat_id = None
        with session_scope() as session:
            last_chat_id = session.query(ChatHistory.id).filter(
                ChatHistory.user_id == userId,
                ChatHistory.session_id == sessionId,
                ChatHistory.message_type == "text"
            ).order_by(ChatHistory.id.desc()).first()

        sum_dict = {"lastIndex": last_chat_id[0] if last_chat_id else None}

        if history_dict.get("sumFlag") is not None and history_dict["sumFlag"]:
            print("[调试] 开始执行摘要处理")
            sum_model = ChatTongyi(model=sum_model_name)
            sum_prompt = PromptTemplate.from_template("根据历史记录{chat_history}和已有的摘要提炼一份摘要{summary}，语言简洁，只保留和用户有关的关键信息，不要自己加信息")
            chain_sum = sum_prompt | sum_model
            sum_res = chain_sum.invoke({"chat_history": chat_history, "summary": history_dict["summary"]})
            with session_scope() as session:
                sum_chat_history = ChatHistory(
                    create_time=datetime.now(),
                    message_type="summary",
                    meta_data=json.dumps(sum_dict, ensure_ascii=False),
                    content=sum_res.content,
                    user_id=userId,
                    session_id=sessionId,
                    role=1
                )
                session.add(sum_chat_history)
                print(sum_chat_history)
        else:
            print("[调试] 跳过摘要处理，sumFlag 不满足条件")

        # 标记流式结束
        yield {
            "sessionId": sessionId,
            "answer": "",
            "done": True
        }


if __name__ == "__main__":
    # save_rag("../test","pet")

    # 第一次对话 - 流式输出
    print("=== 第一次对话 ===")
    print("问题：你是谁？")
    print("回答：", end="", flush=True)
    for chunk in chat(None, "你是谁？", 1, 1):
        if not chunk["done"]:
            print(chunk["answer"], end="", flush=True)
    print("\n")

    # 第二次对话 - 流式输出
    print("=== 第二次对话 ===")
    print("问题：我的边牧叫wish")
    print("回答：", end="", flush=True)
    for chunk in chat(None, "我的边牧叫wish", 1, 1):
        if not chunk["done"]:
            print(chunk["answer"], end="", flush=True)
    print("\n")
