from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory
from app.agent.tools import calc_tool, rag_tool, image_tool, search_tool

react_template = """Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Previous conversation history:
{chat_history}
Begin!
Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
)
tools = [rag_tool, calc_tool, image_tool, search_tool]
agent_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    max_tokens=768,
    reasoning_format="hidden",
)
agent = create_react_agent(agent_llm, tools, prompt)
_sessions: dict[str, ConversationBufferWindowMemory] = {}
def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    return _sessions[session_id]
def run_agent(session_id: str, user_input: str) -> str:
    memory = get_memory(session_id)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors="Your response didn't follow the required format. You must respond with either 'Action:' and 'Action Input:' on separate lines, or 'Final Answer:' if you have enough information. Try again.",
        max_iterations=4,
        max_execution_time=60,
    )
    result = agent_executor.invoke({"input": user_input})
    return result["output"]