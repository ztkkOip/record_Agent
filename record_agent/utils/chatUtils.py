import json
from datetime import datetime
from typing import List
import os
from pathlib import Path

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from record_agent.db_config import session_scope
from record_agent.models import User, ChatHistory, ChatSession

default_memory_config = {
    "maxlen": 20
}
default_model = "qwen3-max"
sum_model_name = "qwen3-max"
default_system_prompt = "你是ztkk,一个智能手账助手，语气尽量温和专业。只需要回答最新的问题,不用每次都介绍自己，需要的时候才介绍"

role_dict = {
    0: "user",
    1: "assistant",
    2: "system"
}
default_chroma_db_filePath = "../chroma_db"
default_topK = 4

# Redis 配置
redis_url = os.environ.get('REDIS_URL', "redis://localhost:6380/0")

# 获取 Redis 客户端
import redis
redis_client = redis.from_url(redis_url)


def _get_redis_history(session_id: int, userId: int) -> List[tuple]:
    """
    从 Redis List 获取对话历史
    如果 Redis 中没有，则从数据库加载并同步到 Redis

    返回: List[(role, content)] 格式的历史记录
    """
    key = f"chat_history:user_{userId}:{session_id}"

    try:
        # 检查 Redis 中是否有历史记录
        history_count = redis_client.llen(key)
        if history_count > 0:
            # 从 Redis 获取所有历史记录
            history_data = redis_client.lrange(key, 0, -1)
            history = []
            for item in history_data:
                msg = json.loads(item)
                history.append((msg['role'], msg['content']))
            return history
    except Exception as e:
        print(f"获取 Redis 历史失败: {e}")

    # Redis 中没有数据，从数据库加载
    return _load_and_sync_to_redis(session_id, userId, key)


def _load_and_sync_to_redis(session_id: int, userId: int, key: str) -> List[tuple]:
    """
    从数据库加载最新的 maxlen 条记录并同步到 Redis List
    """
    with session_scope() as session:
        chatHistories = session.query(ChatHistory).filter(
            ChatHistory.user_id == userId,
            ChatHistory.session_id == session_id,
            ChatHistory.message_type == "text"
        ).order_by(ChatHistory.create_time.desc()).limit(default_memory_config['maxlen']).all()

        # 反转顺序（从旧到新）
        chatHistories = list(reversed(chatHistories))

        history = []

        if chatHistories:
            # 清空旧的 Redis List（如果存在）
            redis_client.delete(key)

            # 同步到 Redis List
            for chat in chatHistories:
                role = "user" if chat.role == 0 else "assistant"
                msg_json = json.dumps({"role": role, "content": chat.content}, ensure_ascii=False)
                redis_client.rpush(key, msg_json)
                history.append((role, chat.content))

    return history


def _save_to_redis(session_id: int, userId: int, user_message: str, assistant_message: str):
    """
    保存消息到 Redis List
    如果超过 maxlen，从队头移除最老的记录
    """
    key = f"chat_history:user_{userId}:{session_id}"
    maxlen = default_memory_config['maxlen']

    # 添加用户消息和助手消息到队尾
    user_msg_json = json.dumps({"role": "user", "content": user_message}, ensure_ascii=False)
    assistant_msg_json = json.dumps({"role": "assistant", "content": assistant_message}, ensure_ascii=False)

    redis_client.rpush(key, user_msg_json)
    redis_client.rpush(key, assistant_msg_json)

    # 检查长度，如果超过 maxlen，从队头移除最老的记录
    current_length = redis_client.llen(key)
    if current_length > maxlen:
        # 移除多余的记录（每次对话添加2条，所以可能需要移除2条）
        remove_count = current_length - maxlen
        redis_client.lpop(key, remove_count)



def save_rag(filePath: str, collectionName: str) -> bool:
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma

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
    from langchain_chroma import Chroma
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

    if model_name is None:
        model_name = default_model

    try:
        model = ChatTongyi(model=model_name)
    except Exception:
        yield {"sessionId": sessionId, "answer": "模型初始化失败", "done": True}
        return

    if system_prompt is None:
        system_prompt = default_system_prompt

    # 获取 Redis 中的对话历史
    redis_history_list = _get_redis_history(sessionId, userId)

    # 获取摘要（从数据库），保持原有摘要逻辑不变
    history_dict = get_history(userId, sessionId)
    chat_summary = history_dict["summary"]
    sumFlag = history_dict["sumFlag"]

    # 使用 Redis 的历史作为对话历史
    chat_history = redis_history_list.copy() if redis_history_list else []

    # 如果有摘要，添加到开头
    if chat_summary:
        chat_history.insert(0, (role_dict[1], chat_summary))

    # 添加当前问题和系统提示
    chat_history.extend([
        (role_dict[0], input),
        (role_dict[2], system_prompt)
    ])

    docs = get_rag(input, "pet")
    prompt = PromptTemplate.from_template("请根据历史记录{chat_history},历史摘要{chat_summary}和搜寻到的资料{docs}回答用户提问")
    print(prompt)
    chat = prompt | model

    # 保存用户问题到数据库
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
    for chunk in chat.stream({"chat_history": chat_history, "docs": docs,"chat_summary": chat_summary}):
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

    # 保存到 Redis（如果超过 maxlen 会自动清理）
    _save_to_redis(sessionId, userId, input, full_answer)

    # 处理摘要 - 获取当前最后一条对话的 id 作为 lastIndex
    last_chat_id = None
    with session_scope() as session:
        last_chat_id = session.query(ChatHistory.id).filter(
            ChatHistory.user_id == userId,
            ChatHistory.session_id == sessionId,
            ChatHistory.message_type == "text"
        ).order_by(ChatHistory.id.desc()).first()

    sum_dict = {"lastIndex": last_chat_id[0] if last_chat_id else None}

    if sumFlag:
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
