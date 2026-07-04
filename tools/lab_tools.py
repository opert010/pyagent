"""Lab Agent 工具：工艺 SOP 检索。"""

from rag import search_knowledge_base


def search_process_sop(keyword: str) -> str:
    """检索实验/封装工艺 SOP 与表征方法。

    Args:
        keyword: 检索关键词，如 "BCB 固化"、"TGA 测试"、"ELN"
    """
    query = f"工艺 SOP 实验流程 {keyword}"
    return search_knowledge_base(query)
