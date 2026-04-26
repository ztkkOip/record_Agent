import json
from datetime import datetime
from typing import List
import os

import asyncio
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from record_agent.db_config import session_scope
from record_agent.models import User, ChatHistory, ChatSession
from record_agent.utils import ImageUtil
from record_agent.utils.RAGUtil import get_rag
from langchain.tools import tool

default_memory_config = {
    "chat_max_len": 20,
    "tool_max_len": 3
}
default_model = "qwen3-max"
sum_model_name = "qwen3-max"
default_system_prompt = "你是ztkk,一个智能手账助手，语气尽量温和专业。只需要回答最新的问题,不用每次都介绍自己，需要的时候才介绍"
default_retry_count  =3
default_tool_retry = 3
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
@tool
def imageGenerate(input:str,imageFile:List[str]=[]):
    """
    用于图片的生成
    input: input-用户需要生成的图片的文本描述，imageFile-用户如果有上传图片就把上传图片的地址粘贴上
    output: 生成的图片下载链接
    """
    return ImageUtil.generateImage(input,imageFile)
def imageGenerateErrorHandler(e:Exception):
    print(e)
    return str(e)


async def get_tools_async():
    client = MultiServerMCPClient(
        {
            "amap-maps":{
                "command":"npx",
                "args":[
                    "-y",
                    "@amap/amap-maps-mcp-server"
                ],
                "env":{
                    "AMAP_MAPS_API_KEY": "a251175b85ab4823b4615ccdc731d58f"
                },
                "transport": "stdio"
            }
        }

    )
    tools = await client.get_tools()
    return tools
tool_dict = {"imageGenerate": imageGenerate}
tools=[imageGenerate]
mcp_tools = asyncio.run(get_tools_async())
if mcp_tools:
    for tool in mcp_tools:
        tool_dict[tool.name] = tool
        tools.append(tool)



def _get_redis_history(session_id: int, userId: int) -> List[tuple]:
    """
    从 Redis List 获取对话历史
    如果 Redis 中没有，则从数据库加载并同步到 Redis

    返回: List[(role, content)] 格式的历史记录
    """
    key = f"chat_history:user_{userId}:1{session_id}"

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
        ).order_by(ChatHistory.create_time.desc()).limit(default_memory_config['chat_max_len']).all()

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
    maxlen = default_memory_config['chat_max_len']

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
        ).order_by(ChatHistory.create_time).limit(default_memory_config['chat_max_len']).all()
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
            if count >= default_memory_config['chat_max_len']:
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
def use_tool(response,key:str)->bool:
    """
    :param response:tool_name-工具名称
    :return: 模型回答
    """
    res = []
    if(response!=None and hasattr(response, 'tool_calls') and response.additional_kwargs.get("tool_calls")!=None and len(response.additional_kwargs["tool_calls"])>0):
            tool_calls = response.additional_kwargs["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    select_tool=tool_dict.get(tool_name)
                    if(select_tool!=None):
                        # 1. 正确解析参数
                        tool_args = tool_call.get('args', {})

                        # 如果没有args，尝试从function字段解析
                        if not tool_args and 'function' in tool_call:
                            args_str = tool_call['function'].get('arguments', '{}')
                            try:
                                tool_args = json.loads(args_str)
                            except json.JSONDecodeError:
                                tool_args = {}
                        content = select_tool.invoke(tool_args)
                        res.append({
                            "tool_name": tool_name,
                            "success": True,
                            "content": content
                        })
                    else:
                        res.append({
                            "tool_name": tool_name,
                            "success": False,
                            "content": "调用失败，该工具不存在"
                        })
                        print("调用失败，该工具不存在")
                except Exception as e:
                    print(e)
                    res.append({
                        "tool_name": tool_name,
                        "success": False,
                        "content": f"调用失败，报错：{e}"
                    })
    else:
        return False
    #2.保存调用工具结果

    redis_client.lpush(key, json.dumps(res))
    key_len  = redis_client.llen(key)
    if(key_len>default_memory_config["tool_max_len"]):
        redis_client.rpop(key,key_len-default_memory_config["tool_max_len"])
    return True



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
    tool_key = f"tool_history:user_{userId}:session_{sessionId}"
    tool_history = redis_client.lrange(tool_key, 0, default_memory_config["tool_max_len"])
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
    model = ChatTongyi(model=model_name)
    model = model.bind_tools(tools=tools)
    prompt = PromptTemplate.from_template("请根据历史记录{chat_history},前几轮对话的工具调用结果{tool_history}，如果调用失败停止调用该工具,历史摘要{chat_summary}和搜寻到的资料{docs}回答用户提问")
    # print(prompt)


    # 保存用户问题到数据库
    saveHistorys = []
    if type(input) == str and input!="":
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
        chat = prompt|model
        response = chat.invoke({"chat_history": chat_history,"chat_summary": chat_summary,"docs": docs,"tool_history":tool_history})
        print(response)
        print(type(response.additional_kwargs))
        tool_flag = use_tool(response,tool_key)
        if response!=None:
            full_answer = response.content
            # 获取 token 使用量
            total_input_tokens = response.response_metadata["token_usage"]["input_tokens"]
            total_output_tokens = response.response_metadata["token_usage"]["output_tokens"]
        else:
            print(f"API错误: ")
            return {
                "sessionId": sessionId,
                "answer": f"API错误: {response.message}",
                "done": True,
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
    meta = [response.additional_kwargs,response.response_metadata]
    # 保存完整的助手回答
    saveHistorys.append(ChatHistory(
        user_id=userId,
        session_id=sessionId,
        content=full_answer,
        role=1,
        model=model_name,
        tokens_used=total_output_tokens,
        create_time=datetime.now(),
        meta_data=json.dumps(meta)
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
        sum_success=False
        for i in range(default_retry_count):
            if(sum_success):
                break
            try:
                # 构建摘要消息
                sum_model = ChatTongyi(model=model_name)
                sum_prompt = PromptTemplate.from_template(f"根据历史记录{chat_history}和已有的摘要{chat_summary}提炼一份摘要，语言简洁，只保留和用户有关的关键信息，不要自己加信息")
                sum_chat = sum_prompt|sum_model
                sum_response = sum_chat.invoke({"chat_history":chat_history,"chat_summary":chat_summary})
                sum_success=True
            except Exception as e:
                print(f"第{i}次摘要处理失败: {e}")
        if sum_success == True and sum_response != None and sum_response.content != "":
            summary_content = sum_response.content
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
    else:
        print("[调试] 跳过摘要处理，sumFlag 不满足条件")

    return {
        "sessionId": sessionId,
        "answer": full_answer,
        "inputTokens": total_input_tokens,
        "outputTokens": total_output_tokens,
        "done": True,
        "finish_reason":response.response_metadata["finish_reason"]
    }

def agent_loop(model_name: str, input: str, sessionId: int = None, userId: int = None, system_prompt: str = None):
    times = 0
    while True:
        if(times>=5):
            break
        res = chat(model_name, input, sessionId, userId, system_prompt)
        print(res)
        if res["finish_reason"]!="tool_calls":
            break
        times+=1
        input="根据上一轮调用的结果继续回答问题"
    return f"此次调用一共进行了{times}\n最终调用结果：{res}"



if __name__ == "__main__":
    # save_rag("../test","pet")

    # 第一次对话
    print("=== 第一次对话 ===")
    print("问题：给我家小狗设计一个动漫头像，二次元风格，尺寸一比一")
    result = agent_loop(None, "给我家小猫设计一个动漫头像", 1, 1)
    print(f"回答：{result}")
    print(f"输入Token: {result.get('inputTokens', 0)}, 输出Token: {result.get('outputTokens', 0)}\n")

    # # 第二次对话
    # print("=== 第二次对话 ===")
    # print("问题：我的边牧叫wish")
    # result = chat(None, "我的边牧叫wish", 1, 1)
    # print(f"回答：{result['answer']}")
    # print(f"输入Token: {result.get('inputTokens', 0)}, 输出Token: {result.get('outputTokens', 0)}\n")
    # print(tools)
