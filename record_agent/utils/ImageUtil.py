import json
from typing import List

import dashscope
import os
from dashscope import MultiModalConversation
import base64
import mimetypes


dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

def generateImage(input:str,imagePaths:List[str])->dict:

    # 获取图像的 Base64 编码
    # 调用编码函数，请将 "/path/to/your/image.png" 替换为您的本地图片文件路径，否则无法运行
    # image = encode_file("D:\\pythonTools\\code\\langchainLearn\\record_agent\\test\\boy.png")
    content = []
    content.append({"text": input})
    for path in imagePaths:
        image = encode_file(path)
        content.append({"image": image})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    api_key = os.getenv("DASHSCOPE_API_KEY")

    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen-image-2.0-pro",
        messages=messages,
        result_format='message',
        stream=False,
        n=2,
        watermark=True,
        negative_prompt=""
    )
    res ={}
    if response.status_code == 200:
        res["success"] = True
        res["content"] = []
        # 如需查看完整响应，请取消下行注释
        print(json.dumps(response, ensure_ascii=False))
        for i, content in enumerate(response.output.choices[0].message.content):
            res["content"].append(content)
    else:
        res["success"] = False
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")
    return res

if __name__ == "__main__":
    print(generateImage("把图上的男人变成他五十年后的样子",["D:\\pythonTools\\code\\langchainLearn\\record_agent\\test\\boy.png"]))