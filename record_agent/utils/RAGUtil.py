import os
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import DashScopeEmbeddings
import dashscope
from dashscope import Generation

# 设置 DashScope API URL
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

default_topK = 4
default_chroma_db_filePath = "../chroma_db"
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
    embedding_function = DashScopeEmbeddings()

    try:
        vs = Chroma(
            collection_name=collectionName,
            embedding_function=embedding_function,
            persist_directory=default_chroma_db_filePath,
            client_settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        # 使用 similarity_search 获取结果
        search_results = vs.similarity_search(input, k=default_topK)
        res = []
        for doc in search_results:
            if hasattr(doc, 'page_content'):
                res.append(doc.page_content)
            else:
                res.append(str(doc))
        return res
    except Exception as e:
        print(f"RAG搜索失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_rag_enhanced(input: str, collectionName: str, num_queries: int = 3) -> List[str]:
    """
    RAG 检索增强：多查询 + RRF 去噪

    Args:
        input: 用户输入
        collectionName: 向量集合名称
        num_queries: 生成查询数量（默认3个）

    Returns:
        去噪后的文档内容列表
    """
    # 1. 使用 LLM 生成多个不同的查询
    queries = _generate_queries(input, num_queries)

    # 2. 对每个查询进行检索，收集结果
    query_results: List[List[str]] = []
    for query in queries:
        docs = get_rag(query, collectionName)
        query_results.append(docs)

    # 3. 使用 RRF 合并去重结果
    ranked_docs = _rrf_fusion(query_results, k=default_topK)

    return ranked_docs


def _generate_queries(input: str, num_queries: int = 3) -> List[str]:
    """
    使用 LLM 生成多个不同的查询

    Args:
        input: 用户输入
        num_queries: 生成查询数量

    Returns:
        查询列表
    """
    prompt = f"""请根据用户问题生成 {num_queries} 个不同的检索查询。
要求：
1. 每个查询从不同角度扩展原始问题
2. 保持简洁，不超过 20 字
3. 直接输出，用换行符分隔

用户问题：{input}"""

    try:
        response = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 按换行符分割并清理
            queries = [q.strip() for q in content.strip().split('\n') if q.strip()]
            return queries[:num_queries]  # 确保返回指定数量
        else:
            return [input]  # 失败时返回原问题
    except Exception:
        return [input]  # 失败时返回原问题


def _rrf_fusion(query_results: List[List[str]], k: int = 4) -> List[str]:
    """
    Reciprocal Rank Fusion (RRF) 算法

    Args:
        query_results: 每个查询的检索结果列表
        k: RRF 参数（默认60）

    Returns:
        排序后的文档列表
    """
    if not query_results:
        return []

    # 收集所有唯一文档及其排名
    doc_scores: Dict[str, float] = {}

    for results in query_results:
        for rank, doc in enumerate(results, 1):
            if doc not in doc_scores:
                doc_scores[doc] = 0
            # RRF 公式：1 / (k + rank)
            doc_scores[doc] += 1.0 / (60 + rank)

    # 按分数降序排序
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # 返回前 k 个文档
    return [doc for doc, _ in ranked_docs[:k]]