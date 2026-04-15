from flask import Flask, request, Response, jsonify
import json
from record_agent.utils.chatUtils import chat

app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    智能体对话接口

    请求参数:
    - userId: 用户ID (必填)
    - sessionId: 会话ID (可选，不传则创建新会话)
    - input: 用户输入内容 (必填)
    - model_name: 模型名称 (可选，默认使用 qwen3-max)
    - system_prompt: 系统提示词 (可选)

    返回: Server-Sent Events (SSE) 流式输出
    """
    data = request.get_json()

    # 参数验证
    userId = data.get('userId')
    sessionId = data.get('sessionId')
    user_input = data.get('input')
    model_name = data.get('model_name')

    if not userId or not user_input:
        return jsonify({
            'error': '缺少必要参数',
            'required': ['userId', 'input']
        }), 400

    def generate():
        try:
            for chunk in chat(
                model_name=model_name,
                input=user_input,
                sessionId=sessionId,
                userId=userId,
            ):
                # SSE 格式: data: {json}\n\n
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_chunk = {
                'sessionId': sessionId,
                'answer': f'服务错误: {str(e)}',
                'done': True
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'service': 'record-agent'
    })


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=14000,
        debug=True,
        threaded=True
    )