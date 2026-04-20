import os
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import DashScopeEmbeddings
import dashscope
from dashscope import Generation
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
# 设置 DashScope API URL
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

default_topK = 4
# 获取当前文件的绝对路径，然后计算 chroma_db 的绝对路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
default_chroma_db_filePath = os.path.join(os.path.dirname(_current_dir), "chroma_db")

# 禁用 ChromaDB 遥测
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['DO_NOT_TRACK'] = 'true'
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
        # 调试：输出实际使用的路径
        db_path = os.path.abspath(default_chroma_db_filePath)
        print(f"[调试] 使用数据库路径: {db_path}")

        vs = Chroma(
            collection_name=collectionName,
            embedding_function=embedding_function,
            persist_directory=default_chroma_db_filePath,
            client_settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 调试：检查集合中是否有数据
        all_docs = vs.get()
        doc_count = len(all_docs.get('documents', []))
        print(f"[调试] 集合 {collectionName} 中有 {doc_count} 条文档")

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


def getRagByExtendUserQuery(input: str, collectionName: str, num_queries: int = 3) -> List[str]:
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
    queries = generate_queries(input, num_queries)

    # 2. 对每个查询进行检索，收集结果
    query_results: List[List[str]] = []
    for query in queries:
        docs = get_rag(query, collectionName)
        query_results.append(docs)

    # 3. 使用 RRF 合并去重结果
    ranked_docs = rrf_fusion(query_results, k=default_topK)

    return ranked_docs


def generate_queries(input: str, num_queries: int = 3) -> List[str]:
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


def rrf_fusion(query_results: List[List[str]], k: int = 4) -> List[str]:
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


def save_rag_with_parent_child(filePath: str, collectionName: str, parent_size: int = 2000, child_size: int = 300) -> bool:
    """
    保存文档到向量库，使用父子文档结构

    原理：将文档分成较大的父文档（保留完整上下文），再切分成小的子块（提高检索精度）
         检索时在子块中搜索，命中后返回对应的父文档

    Args:
        filePath: 文件或目录路径
        collectionName: 集合名称
        parent_size: 父文档大小（默认2000字符）
        child_size: 子块大小（默认300字符）

    Returns:
        是否成功
    """
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=50)

        if os.path.isfile(filePath):
            file_ext = Path(filePath).suffix.lower()
            documents = []
            if file_ext == '.pdf':

                documents = PyPDFLoader(filePath).load()
            elif file_ext == '.docx':

                documents = Docx2txtLoader(filePath).load()
            else:
                documents = TextLoader(filePath, encoding='utf-8').load()

            api_key = os.environ.get('DASHSCOPE_API_KEY')
            embedding_function = DashScopeEmbeddings(dashscope_api_key=api_key) if api_key else DashScopeEmbeddings()

            vs = Chroma(
                collection_name=collectionName,
                embedding_function=embedding_function,
                persist_directory=default_chroma_db_filePath
            )

            parent_docs = parent_splitter.split_documents(documents)
            all_docs = []

            for parent_idx, parent_doc in enumerate(parent_docs):
                parent_id = f"parent_{parent_idx}"
                parent_doc.metadata['doc_type'] = 'parent'
                parent_doc.metadata['parent_id'] = parent_id
                all_docs.append(parent_doc)

                child_docs = child_splitter.split_documents([parent_doc])
                for child_doc in child_docs:
                    child_doc.metadata['doc_type'] = 'child'
                    child_doc.metadata['parent_id'] = parent_id
                    all_docs.append(child_doc)

            if all_docs:
                vs.add_documents(all_docs)
                print(f"父子文档已保存，父文档 {len(parent_docs)} 个，子块 {len(all_docs) - len(parent_docs)} 个")
            return True

        elif os.path.isdir(filePath):
            all_documents = []
            for root, _, files in os.walk(filePath):
                for file in files:
                    file_ext = Path(file).suffix.lower()
                    if file_ext in ['.pdf', '.docx', '.txt', '.md', '.json', '.xml']:
                        try:
                            if file_ext == '.pdf':
                                docs = PyPDFLoader(os.path.join(root, file)).load()
                            elif file_ext == '.docx':
                                docs = Docx2txtLoader(os.path.join(root, file)).load()
                            else:
                                docs = TextLoader(os.path.join(root, file), encoding='utf-8').load()
                            all_documents.extend(docs)
                        except Exception:
                            continue

            if all_documents:
                embedding_function = DashScopeEmbeddings()
                vs = Chroma(
                    collection_name=collectionName,
                    embedding_function=embedding_function,
                    persist_directory=default_chroma_db_filePath
                )

                parent_docs = parent_splitter.split_documents(all_documents)
                all_docs = []

                for parent_idx, parent_doc in enumerate(parent_docs):
                    parent_id = f"parent_{parent_idx}"
                    parent_doc.metadata['doc_type'] = 'parent'
                    parent_doc.metadata['parent_id'] = parent_id
                    all_docs.append(parent_doc)

                    child_docs = child_splitter.split_documents([parent_doc])
                    for child_doc in child_docs:
                        child_doc.metadata['doc_type'] = 'child'
                        child_doc.metadata['parent_id'] = parent_id
                        all_docs.append(child_doc)

                vs.add_documents(all_docs)
                print(f"目录父子文档已保存，父文档 {len(parent_docs)} 个，子块 {len(all_docs) - len(parent_docs)} 个")
            return True

        return False

    except Exception as e:
        print(f"保存父子文档失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def getRagByParentChild(input: str, collectionName: str) -> List[str]:
    """
    使用父子文档检索：检索子块，返回对应的父文档

    Args:
        input: 用户查询
        collectionName: 集合名称

    Returns:
        父文档内容列表（去重）
    """
    embedding_function = DashScopeEmbeddings()

    try:
        vs = Chroma(
            collection_name=collectionName,
            embedding_function=embedding_function,
            persist_directory=default_chroma_db_filePath,
            client_settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # 只检索子块
        child_results = vs.similarity_search(
            input,
            k=default_topK * 2,
            filter={'doc_type': 'child'}
        )

        parent_ids = set()
        for child_doc in child_results:
            if hasattr(child_doc, 'metadata') and 'parent_id' in child_doc.metadata:
                parent_ids.add(child_doc.metadata['parent_id'])

        if not parent_ids:
            return []

        # 根据 parent_id 获取父文档
        parent_docs = []
        for parent_id in parent_ids:
            parent_results = vs.get(where={'parent_id': parent_id, 'doc_type': 'parent'})
            for parent_doc in parent_results.get('documents', []):
                if hasattr(parent_doc, 'page_content'):
                    parent_docs.append(parent_doc.page_content)

        return parent_docs[:default_topK]

    except Exception as e:
        print(f"父子文档检索失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def getRagByHybrid(input: str, collectionName: str, dense_weight: float = 0.7, sparse_weight: float = 0.3) -> List[str]:
    """
    混合检索：稠密向量检索 + BM25稀疏检索

    原理：
    - 稠密向量检索：通过语义相似度找到相关文档
    - BM25稀疏检索：通过关键词匹配找到相关文档
    - 混合融合：按权重合并两种结果，取最优文档

    Args:
        input: 用户查询
        collectionName: 集合名称
        dense_weight: 稠密检索权重（默认0.7）
        sparse_weight: BM25检索权重（默认0.3）

    Returns:
        混合融合后的文档内容列表
    """
    try:
        from rank_bm25 import BM25Okapi
        import jieba

        embedding_function = DashScopeEmbeddings()
        vs = Chroma(
            collection_name=collectionName,
            embedding_function=embedding_function,
            persist_directory=default_chroma_db_filePath,
            client_settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # 1. 稠密向量检索
        dense_results = vs.similarity_search(input, k=default_topK * 2)
        dense_scores = {doc.page_content if hasattr(doc, 'page_content') else str(doc): 1.0 - idx / (default_topK * 2)
                       for idx, doc in enumerate(dense_results)}

        # 2. BM25稀疏检索
        all_docs = vs.get()
        documents = all_docs.get('documents', [])
        if not documents:
            return []

        # 使用jieba分词
        corpus = []
        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            corpus.append(list(jieba.cut(content)))

        bm25 = BM25Okapi(corpus)
        query_tokens = list(jieba.cut(input))
        sparse_scores = bm25.get_scores(query_tokens)

        # 归一化BM25分数
        if sparse_scores.max() - sparse_scores.min() > 0:
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
        else:
            sparse_scores = sparse_scores * 0

        sparse_results = {}
        for idx, doc in enumerate(documents):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            sparse_results[content] = sparse_scores[idx]

        # 3. 混合融合
        all_contents = set(dense_scores.keys()) | set(sparse_results.keys())
        final_scores = {}

        for content in all_contents:
            dense_score = dense_scores.get(content, 0)
            sparse_score = sparse_results.get(content, 0)
            final_scores[content] = dense_weight * dense_score + sparse_weight * sparse_score

        # 按分数降序排序
        ranked_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:default_topK]]

    except Exception as e:
        print(f"混合检索失败: {e}")
        import traceback
        traceback.print_exc()
        return []

