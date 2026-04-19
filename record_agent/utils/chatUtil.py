import json
from datetime import datetime
from typing import List
import os
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dashscope import Generation
from record_agent.db_config import session_scope
from record_agent.models import User, ChatHistory, ChatSession
from record_agent.utils.RAGUtil import get_rag

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


def chat(model_name: str, input: str, sessionId: int = None, userId: int = None, system_prompt: str = None) -> dict:
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

    # 构建 prompt
    prompt_text = f"请根据历史记录{chat_history},历史摘要{chat_summary}和搜寻到的资料{docs}回答用户提问"
    print(prompt_text)

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

    # 使用官方 dashscope SDK 调用
    try:
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话
        for role, content in chat_history:
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": str(content)})

        # 添加用户问题
        messages.append({"role": "user", "content": input})

        response = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=model_name,
            messages=messages,
            result_format="message",
            enable_thinking=True,
        )

        if response.status_code == 200:
            full_answer = response.output.choices[0].message.content

            # 获取 token 使用量
            total_input_tokens = response.usage.input_tokens
            total_output_tokens = response.usage.output_tokens
        else:
            print(f"API错误: {response.status_code}, {response.code}, {response.message}")
            return {
                "sessionId": sessionId,
                "answer": f"API错误: {response.message}",
                "done": True,
                "error": response.message
            }

    except Exception as e:
        print(f"获取响应失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "sessionId": sessionId,
            "answer": f"生成失败: {str(e)}",
            "done": True,
            "error": str(e)
        }

    # 保存完整的助手回答
    saveHistorys.append(ChatHistory(
        user_id=userId,
        session_id=sessionId,
        content=full_answer,
        role=1,
        model=model_name,
        tokens_used=total_output_tokens,
        create_time=datetime.now(),
    ))

    # 更新用户问题的 token 使用量
    if saveHistorys and len(saveHistorys) > 1:
        saveHistorys[0].tokens_used = total_input_tokens
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
        try:
            # 构建摘要消息
            summary_messages = [{"role": "system", "content": "根据历史记录和已有的摘要提炼一份摘要，语言简洁，只保留和用户有关的关键信息，不要自己加信息"}]
            summary_messages.extend([{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ["user", "assistant"]])

            sum_response = Generation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=sum_model_name,
                messages=summary_messages,
                result_format="message",
            )

            if sum_response.status_code == 200:
                summary_content = sum_response.output.choices[0].message.content
                with session_scope() as session:
                    sum_chat_history = ChatHistory(
                        create_time=datetime.now(),
                        message_type="summary",
                        meta_data=json.dumps(sum_dict, ensure_ascii=False),
                        content=summary_content,
                        user_id=userId,
                        session_id=sessionId,
                        role=1
                    )
                    session.add(sum_chat_history)
                    print(sum_chat_history)
        except Exception as e:
            print(f"摘要处理失败: {e}")
    else:
        print("[调试] 跳过摘要处理，sumFlag 不满足条件")

    return {
        "sessionId": sessionId,
        "answer": full_answer,
        "inputTokens": total_input_tokens,
        "outputTokens": total_output_tokens,
        "done": True
    }



if __name__ == "__main__":
    # save_rag("../test","pet")

    # 第一次对话
    print("=== 第一次对话 ===")
    print("问题：你是谁？")
    result = chat(None, "你是谁？", 1, 1)
    print(f"回答：{result['answer']}")
    print(f"输入Token: {result.get('inputTokens', 0)}, 输出Token: {result.get('outputTokens', 0)}\n")

    # 第二次对话
    print("=== 第二次对话 ===")
    print("问题：我的边牧叫wish")
    result = chat(None, "我的边牧叫wish", 1, 1)
    print(f"回答：{result['answer']}")
    print(f"输入Token: {result.get('inputTokens', 0)}, 输出Token: {result.get('outputTokens', 0)}\n")
