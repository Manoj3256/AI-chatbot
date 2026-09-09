from langchain_classic.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from app.retrieval.rag import rag_answer
from app.vision.image_search import search_images

def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

calc_tool = Tool(
    name="Calculator",
    description="Evaluates a math expression. Input should be a valid Python arithmetic expression, e.g. '1599 * 0.85'.",
    func=calculate,
)

def rag_tool_func(query: str) -> str:
    answer, _chunks = rag_answer(query)
    return answer

rag_tool = Tool(
    name="DocumentQA",
    description=(
        "Searches a knowledge base containing information specifically about "
        "the movie Interstellar (its plot, cast, director, production, box "
        "office, and awards). Use ONLY for questions about the movie "
        "Interstellar. Do NOT use for math, general knowledge, or unrelated topics."),
    func=rag_tool_func,
)
def image_search_func(query: str) -> str:
    results = search_images(query)
    return f"Found {len(results)} matching image(s) for '{query}'."

image_tool = Tool(
    name="ImageSearch",
    description="Searches for images matching a text description. Use this when the user asks to find or see an image.",
    func=image_search_func,
)

ddg_search = DuckDuckGoSearchResults()
def ddg_search_truncated(query: str) -> str:
    raw = ddg_search.run(query)
    return raw[:800]

search_tool = Tool(
    name="duckduck",
    description="A web search engine. Use this only for questions the document knowledge base can't answer.",
    func=ddg_search_truncated,
)