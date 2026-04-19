from flask import Flask, request, jsonify
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
    - model_name: 模型名称 (可选，默认使用 qwen3.5-27b)
    - system_prompt: 系统提示词 (可选)

    返回: JSON 格式
    """
    data = request.get_json()

    # 参数验证
    userId = data.get('userId')
    sessionId = data.get('sessionId')
    user_input = data.get('input')
    model_name = data.get('model_name')
    system_prompt = data.get('system_prompt')

    if not userId or not user_input:
        return jsonify({
            'error': '缺少必要参数',
            'required': ['userId', 'input']
        }), 400

    try:
        result = chat(
            model_name=model_name,
            input=user_input,
            sessionId=sessionId,
            userId=userId,
            system_prompt=system_prompt
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'sessionId': sessionId,
            'answer': f'服务错误: {str(e)}',
            'done': True,
            'error': str(e)
        }), 500


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