from research_agent.agent import Agent
from research_agent.llm import LLMResponse, ScriptedLLM, ToolCall
from research_agent.memory import Memory
from research_agent.tools import Notes, default_registry


def make_agent(responses, **kwargs):
    return Agent(ScriptedLLM(responses), default_registry(Notes()), **kwargs)


def test_answers_directly_without_tools():
    agent = make_agent([LLMResponse(text="42")])
    result = agent.run("question")
    assert result.answer == "42"
    assert result.steps == 1
    assert result.tool_calls == []
    assert result.finished


def test_uses_tool_then_answers():
    agent = make_agent(
        [
            LLMResponse(tool_calls=[ToolCall("t1", "calculator", {"expression": "6 * 7"})]),
            LLMResponse(text="The answer is 42."),
        ]
    )
    result = agent.run("What is 6 times 7?")
    assert result.answer == "The answer is 42."
    assert result.steps == 2
    assert result.tool_calls == [("calculator", {"expression": "6 * 7"})]


def test_tool_result_is_fed_back_to_model():
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall("t1", "calculator", {"expression": "2 + 2"})]),
            LLMResponse(text="done"),
        ]
    )
    seen = []
    original = llm.complete

    def spy(system, messages, tools):
        seen.append([dict(m) for m in messages])
        return original(system, messages, tools)

    llm.complete = spy
    Agent(llm, default_registry()).run("q")
    last_user = seen[1][-1]
    assert last_user["role"] == "user"
    block = last_user["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1"
    assert block["content"] == "4"
    assert block["is_error"] is False


def test_tool_error_is_reported_not_raised():
    agent = make_agent(
        [
            LLMResponse(tool_calls=[ToolCall("t1", "calculator", {"expression": "1 / 0"})]),
            LLMResponse(text="That is undefined."),
        ]
    )
    result = agent.run("q")
    assert result.finished
    assert result.answer == "That is undefined."


def test_unknown_tool_is_reported_not_raised():
    agent = make_agent(
        [
            LLMResponse(tool_calls=[ToolCall("t1", "does_not_exist", {})]),
            LLMResponse(text="ok"),
        ]
    )
    assert agent.run("q").finished


def test_step_limit_stops_runaway_loop():
    looping = [
        LLMResponse(tool_calls=[ToolCall(f"t{i}", "calculator", {"expression": "1 + 1"})])
        for i in range(10)
    ]
    result = make_agent(looping, max_steps=3).run("q")
    assert not result.finished
    assert result.steps == 3
    assert "step limit" in result.answer


def test_memory_keeps_first_message_and_drops_oldest_pairs():
    memory = Memory(max_messages=5)
    memory.add({"role": "user", "content": "question"})
    for i in range(6):
        memory.add({"role": "assistant", "content": f"a{i}"})
        memory.add({"role": "user", "content": f"u{i}"})
    assert len(memory) <= 5
    assert memory.messages[0]["content"] == "question"
    # roles still alternate correctly after trimming
    roles = [m["role"] for m in memory.messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
