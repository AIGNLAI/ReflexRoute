from reflexroute.retrieval import retrieve_balanced, tfidf_similarities
from reflexroute.schemas import HistoryRecord


def test_retrieval_is_ranked_and_balanced_per_model():
    history = [
        HistoryRecord("solve a quadratic equation", "math", 0.98),
        HistoryRecord("summarize an email", "math", 0.50),
        HistoryRecord("write a short summary", "fast", 0.90),
        HistoryRecord("factor a polynomial equation", "fast", 0.65),
        HistoryRecord("solve an equation", "not-a-candidate", 1.0),
    ]
    result = retrieve_balanced(
        "solve this polynomial equation", ["math", "fast"], history, top_k=1
    )
    assert list(result) == ["math", "fast"]
    assert result["math"][0].record.query == "solve a quadratic equation"
    assert result["fast"][0].record.query == "factor a polynomial equation"


def test_empty_documents_and_cjk_are_supported():
    assert tfidf_similarities("hello", []) == []
    similarities = tfidf_similarities("编写分布式限流器", ["实现分布式限流算法", "总结邮件"])
    assert similarities[0] > similarities[1]
