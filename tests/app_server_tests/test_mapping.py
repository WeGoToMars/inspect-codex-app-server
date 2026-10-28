"""Transcript reconstruction from fixed Codex protocol events.

Exercise message order, tool results, timestamps, compaction, and accepted
delegation. These tests feed the mapper directly without starting Codex.
"""

from __future__ import annotations

import json

import pytest
from inspect_ai.event import InputEvent

from inspect_codex_app_server.mapping import EventStreamMapper

from .support import (
    command_item,
    completed_item,
    make_mapper,
    make_state,
    model_events,
    usage_params,
)


def test_native_tool_times_and_denied_review_are_preserved(captured_events, monkeypatch):
    """Native lifecycle time and guardian denial must retain their meaning."""
    from datetime import datetime, timezone

    from inspect_ai.event import ReviewEvent, ToolEvent

    from inspect_codex_app_server import mapping

    mapper = EventStreamMapper("gpt-5.4")
    mapper.root_thread_id = "root"
    mapper.on_item_completed("item/completed", completed_item("root", {
        "type": "subAgentActivity", "id": "spawn", "kind": "started",
        "agentThreadId": "child", "agentPath": "/root/worker",
    }))
    mapper.on_item_started("item/started", {
        "threadId": "child", "startedAtMs": 1700000000000,
        "item": {"id": "native-command", "type": "commandExecution"},
    })
    mapper.on_item_completed("item/completed", {
        "threadId": "child", "completedAtMs": 1700000004000,
        "item": {"id": "native-command", "type": "commandExecution", "command": "true",
                 "status": "completed", "durationMs": 4000, "exitCode": 0, "aggregatedOutput": "ok"},
    })
    mapper.on_review_event("item/autoApprovalReview/completed", {
        "threadId": "child", "reviewId": "review", "targetItemId": "native-command",
        "review": {"status": "denied", "rationale": "NATIVE_DENIAL_SENTINEL"},
        "action": {"type": "command", "command": "true"},
    })
    event = next(event for event in captured_events if isinstance(event, ToolEvent) and event.function == "bash")
    assert event.timestamp == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    assert (event.completed - event.timestamp).total_seconds() == 4
    review = next(event for event in captured_events if isinstance(event, ReviewEvent))
    assert review.decision == "terminate" and "NATIVE_DENIAL_SENTINEL" in review.explanation
    assert review.span_id == mapper.subagents["child"].agent_span_id

    # Inspect expects the caller to supply working time when completing a tool.
    # The fixed start and completion times must retain the four-second duration.
    finished = datetime(2023, 11, 14, 22, 13, 24, tzinfo=timezone.utc)

    class CompletionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return finished

    subagent_event = mapper.subagents["child"].tool_event
    subagent_event.timestamp = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    monkeypatch.setattr(mapping, "datetime", CompletionClock)
    mapper.on_thread_closed("thread/closed", {"threadId": "child"})
    assert subagent_event.completed == finished
    assert subagent_event.working_time == 4.0




def test_native_turn_seconds_and_item_milliseconds_share_clock(captured_events):
    """A misdated turn expands the viewer's agent duration back to 1970.

    The pinned native Turn schema uses Unix seconds; item lifecycle fields
    use milliseconds. Both represent the same fixed four-second interval.
    """
    from datetime import datetime, timezone

    from inspect_ai.event import InfoEvent, ToolEvent

    mapper = make_mapper()
    mapper.on_turn_started("turn/started", {
        "threadId": "root", "turn": {"id": "turn-1", "startedAt": 1700000000},
    })
    mapper.on_item_started("item/started", {
        "threadId": "root", "startedAtMs": 1700000000000,
        "item": {"id": "command", "type": "commandExecution"},
    })
    mapper.on_item_completed("item/completed", {
        "threadId": "root", "completedAtMs": 1700000004000,
        "item": command_item("command"),
    })

    mapper.on_turn_completed("turn/completed", {
        "threadId": "root", "turn": {"id": "turn-1", "startedAt": 1700000000, "completedAt": 1700000010},
    })
    mapper.on_goal_updated("thread/goal/updated", {"goal": {"objective": "o", "status": "complete"}})

    turn = next(e for e in captured_events if isinstance(e, InfoEvent) and e.source == "codex-turn")
    tool = next(e for e in captured_events if isinstance(e, ToolEvent))
    goal = next(e for e in captured_events if isinstance(e, InfoEvent) and e.source == "goal")
    assert turn.timestamp == tool.timestamp == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    assert (tool.completed - turn.timestamp).total_seconds() == 4
    # Events after a turn completes follow it, not its start.
    assert (goal.timestamp - turn.timestamp).total_seconds() == 10




def test_duplicate_item_delivery_is_deduped(captured_events):
    """Redelivered items contribute one message exchange and one model event."""
    mapper = make_mapper(make_state())
    item = completed_item("root", command_item("cmd1"))
    mapper.on_item_completed("", item)
    mapper.on_item_completed("", item)
    mapper.on_item_completed("", item)
    assert [m.role for m in mapper.state.messages] == ["user", "assistant", "tool"]
    assert mapper.state.messages[-1].text == "ok"
    assert len(model_events(captured_events)) == 1








def test_root_and_child_tools_share_names_arguments_and_results(captured_events):
    mapper = EventStreamMapper("codex/test", mcp_tool_aliases=(("remote", "lookup", "lookup_alias"),))
    mapper.root_thread_id = "root"
    mapper.on_thread_metadata("thread/metadata/read", {
        "thread": {"id": "child", "parentThreadId": "root", "model": "codex/test"}})
    for thread in ("root", "child"):
        mapper.on_item_completed("item/completed", completed_item(thread, {
            "id": "search", "type": "webSearch", "query": "arithmetic",
            "action": {"type": "search", "query": "arithmetic"},
        }))
        mapper.on_item_completed("item/completed", completed_item(thread, {
            "id": "lookup", "type": "mcpToolCall", "server": "remote", "tool": "lookup",
            "arguments": {"key": 42}, "status": "completed",
            "result": {"content": [{"type": "text", "text": "found"}]},
        }))
        calls = [call for message in mapper.thread_messages(thread) if message.role == "assistant"
                 for call in message.tool_calls or [] if call.function != "agent"]
        assert [(call.function, call.arguments) for call in calls] == [
            ("web_search", {"query": "arithmetic"}), ("lookup_alias", {"key": 42})]
        tool_messages = [m for m in mapper.thread_messages(thread) if m.role == "tool" and m.function != "agent"]
        assert [(m.function, m.text) for m in tool_messages] == [("web_search", "(completed)"), ("lookup_alias", "found")]


def test_dynamic_tool_failure_is_a_tool_error(captured_events):
    mapper = make_mapper(make_state())
    mapper.on_item_completed(
        "",
        completed_item(
            "root",
            {
                "type": "dynamicToolCall",
                "id": "d1",
                "tool": "lookup",
                "arguments": {"query": "example"},
                "status": "completed",
                "success": False,
                "contentItems": [{"type": "inputText", "text": "nope"}],
            },
        ),
    )
    tool_messages = [m for m in mapper.messages if m.role == "tool"]
    assert tool_messages[-1].error is not None


def test_compaction_tokens_before_is_unknown(captured_events):
    """Compaction preserves duration while leaving unreported context size unknown."""
    mapper = make_mapper(make_state())
    mapper.on_token_usage("", usage_params("root", 5000))
    mapper.on_item_completed("", completed_item("root", {"type": "contextCompaction", "id": "comp1"}))
    compaction = [e for e in captured_events if getattr(e, "event", None) == "compaction"]
    assert len(compaction) == 1
    assert compaction[0].tokens_before is None
    assert compaction[0].working_start == 0.0

    # Context compaction with explicit start and completion timestamps.
    mapper.on_item_started("", {"threadId": "root", "item": {"type": "contextCompaction", "id": "comp2"}, "startedAtMs": 10000})
    mapper.on_item_completed("", {"threadId": "root", "item": {"type": "contextCompaction", "id": "comp2"}, "completedAtMs": 25000})
    comp2 = [e for e in captured_events if getattr(e, "event", None) == "compaction" and (e.metadata or {}).get("compaction_id") == "comp2"][0]
    assert comp2.working_start == 15.0


def test_child_compaction_flushes_only_child_history(captured_events):
    state = make_state()
    state.messages[0].text = "ROOT_INPUT_SENTINEL"
    mapper = make_mapper(state)
    mapper.on_item_completed("", completed_item("root", {
        "type": "collabAgentToolCall", "id": "spawn", "tool": "spawnAgent",
        "senderThreadId": "root", "receiverThreadIds": ["child"],
        "status": "completed", "prompt": "CHILD_INPUT_SENTINEL",
    }))
    for number in range(2):
        mapper.on_item_completed("", completed_item("child", {
            "type": "reasoning", "id": f"reason-{number}", "content": ["CHILD_REASONING_SENTINEL"],
        }))
        mapper.on_item_completed("", completed_item("child", {
            "type": "contextCompaction", "id": f"compact-{number}",
        }))
    child = mapper.subagents["child"]
    events = [event for event in model_events(captured_events) if event.span_id == child.agent_span_id]
    assert len(events) == 2
    assert events[0].input[0].text == "CHILD_INPUT_SENTINEL"
    assert len({event.output.message.id for event in events}) == 2
    assert all("ROOT_INPUT_SENTINEL" not in message.text for event in events for message in event.input)
    assert "CHILD_REASONING_SENTINEL" in str(child.messages)
    assert "CHILD_REASONING_SENTINEL" not in str(state.messages)


def test_unknown_item_is_retained_as_raw_evidence(captured_events):
    mapper = make_mapper(make_state())
    mapper.on_item_completed(
        "item/completed", completed_item("root", {"type": "brandNewItem", "id": "u1", "payload": 42})
    )
    raw = [e for e in captured_events if getattr(e, "source", None) == "codex-raw-item"]
    assert len(raw) == 1
    assert raw[0].data["item"]["type"] == "brandNewItem"
    assert any(r["method"] == "item/completed" for r in mapper.raw_events)


def test_model_event_input_excludes_its_own_message(captured_events):
    """A model event's input snapshot excludes its own output message."""
    state = make_state()
    mapper = make_mapper(state)
    mapper.on_item_completed("", completed_item("root", command_item("a")))

    event = model_events(captured_events)[0]
    emitted_ids = {b for m in event.input for b in ([m.id] if getattr(m, "id", None) else [])}
    assistant_ids = {m.id for m in mapper.messages if m.role == "assistant"}
    assert not (emitted_ids & assistant_ids)


def test_pending_content_order_is_preserved(captured_events):
    """Reasoning and text retain their arrival order in the reconstructed message."""
    mapper = make_mapper(make_state())
    mapper.on_item_completed("", completed_item("root", {"type": "agentMessage", "id": "m1", "text": "text-first"}))
    mapper.on_item_completed("", completed_item("root", {"type": "reasoning", "id": "r1", "content": ["think"]}))
    mapper.on_item_completed("", completed_item("root", {"type": "agentMessage", "id": "m2", "text": "text-second"}))
    mapper.on_turn_completed("", {"threadId": "root", "turn": {"id": "t1"}})

    final = [m for m in mapper.messages if m.role == "assistant"][-1]
    kinds = [type(b).__name__ for b in final.content]
    assert kinds == ["ContentText", "ContentReasoning", "ContentText"]


def test_empty_submission_stays_empty_after_log_roundtrip(captured_events):
    """Unsubmitted agent text stays in the transcript, never in the scored output."""
    from inspect_ai.model import ModelOutput

    state = make_state()
    mapper = make_mapper(state)
    mapper.on_item_completed("item/completed", completed_item("root", {
        "type": "agentMessage", "id": "partial", "text": "a" * 64}))
    mapper.finalize_state(state, "")
    assert ModelOutput.model_validate_json(state.output.model_dump_json()).completion == ""
    assert any(m.text == "a" * 64 for m in state.messages)


def test_user_messages_preserve_new_input_without_echo_duplicates(captured_events):
    state = make_state()
    state.messages[0].text = "Initial prompt"
    mapper = make_mapper(state)
    for number, text in enumerate(("Initial prompt", "Next instruction", "Next instruction")):
        mapper.on_item_completed("item/completed", completed_item("root", {
            "type": "userMessage", "id": f"u{number}", "content": [{"type": "text", "text": text}],
        }))
    assert [m.text for m in state.messages] == ["Initial prompt", "Next instruction"]
    assert [e.input for e in captured_events if e.event == "input"] == ["Next instruction"]

    # Native prompt augmentation updates the existing message, retaining its identity.
    mapper.on_item_completed("item/completed", completed_item("root", {
        "type": "userMessage", "id": "augmented", "content": [
            {"type": "text", "text": "Context\n\nNext instruction"}],
    }))
    assert [m.text for m in state.messages] == ["Initial prompt", "Context\n\nNext instruction"]


def test_user_message_in_subagent_is_mapped(captured_events):
    mapper = make_mapper(make_state())
    mapper.on_item_completed(
        "",
        completed_item("root", {"type": "subAgentActivity", "id": "sa1", "kind": "started", "agentThreadId": "child", "agentPath": "root/worker"}),
    )
    mapper.on_item_completed(
        "",
        completed_item(
            "child",
            {
                "type": "userMessage",
                "id": "cu1",
                "content": [{"type": "text", "text": "child prompt"}],
            },
        ),
    )
    sub = mapper.subagents["child"]
    assert [m.text for m in sub.messages] == ["child prompt"]
    assert all(m.text != "child prompt" for m in mapper.state.messages)
    child_inputs = [e for e in captured_events if getattr(e, "event", None) == "input" and getattr(e, "span_id", None) == sub.agent_span_id]
    assert len(child_inputs) == 1
    assert child_inputs[0].input == "child prompt"

    # A mid-turn message separates the calls on either side of it, as on the root.
    for item in (command_item("ls"), {"type": "userMessage", "id": "steer", "content": [
            {"type": "text", "text": "steer"}]}, command_item("cat"), {"type": "plan", "id": "p", "text": "child plan"}):
        mapper.on_item_completed("item/completed", completed_item("child", item))
    assert [(m.role, [c.id for c in getattr(m, "tool_calls", None) or []]) for m in sub.messages[1:]] == [
        ("assistant", ["call_ls"]), ("tool", []), ("user", []), ("assistant", ["call_cat"]), ("tool", [])]
    plan, = [e for e in captured_events if getattr(e, "source", None) == "plan"]
    assert plan.span_id == sub.agent_span_id


def test_multi_tool_calls_grouped_into_single_model_event(captured_events):
    """Multiple tool calls in a turn step belong to a single ModelEvent."""
    from inspect_ai.event import ToolEvent

    mapper = make_mapper(make_state())
    # Model reasons, then executes two tools before new reasoning
    mapper.on_item_completed("", completed_item("root", {"type": "reasoning", "id": "r1", "content": ["thinking"]}))
    mapper.on_item_completed("", completed_item("root", command_item("c1", command="ls -la")))
    mapper.on_item_completed("", completed_item("root", {
        "type": "mcpToolCall",
        "id": "m1",
        "server": "inventory",
        "tool": "status",
        "arguments": {},
        "status": "completed",
        "result": {"content": [{"type": "text", "text": "time ok"}]},
    }))

    m_events = model_events(captured_events)
    # Verify that multiple tool calls within the same step update a single ModelEvent in place.
    assert len({e.uuid for e in m_events}) == 1
    asst_msg = m_events[-1].output.choices[0].message
    assert len(asst_msg.tool_calls) == 2
    assert asst_msg.tool_calls[0].function == "bash"
    assert asst_msg.tool_calls[1].function == "inventory_status"

    asst_messages = [m for m in mapper.messages if m.role == "assistant"]
    assert len(asst_messages) == 1
    assert len(asst_messages[0].tool_calls) == 2

    # Verify separate ToolEvents are emitted for each individual tool call.
    t_events = [e for e in captured_events if isinstance(e, ToolEvent)]
    assert len(t_events) == 2
    assert t_events[0].function == "bash"
    assert t_events[1].function == "inventory_status"


def test_file_change_renders_markdown_diff_view(captured_events):
    """fileChange items produce ToolCall and ToolEvent with syntax-highlighted diff markdown."""
    from inspect_ai.event import ToolEvent

    mapper = make_mapper(make_state())
    patch = "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new"
    mapper.on_item_completed("", completed_item("root", {
        "type": "fileChange",
        "id": "fc1",
        "changes": [{"path": "/app/test.py", "kind": {"type": "update"}, "diff": patch}],
        "status": "applied",
    }))

    m_events = model_events(captured_events)
    assert len(m_events) == 1
    tc = m_events[0].output.choices[0].message.tool_calls[0]
    assert tc.function == "text_editor"
    assert tc.view is not None
    assert tc.arguments["path"] == "/app/test.py"
    assert patch in tc.view.content

    te = next(e for e in captured_events if isinstance(e, ToolEvent) and e.function == "text_editor")
    assert te.view is not None
    assert patch in te.view.content


def test_turn_diff_updated_retained_in_evidence_not_transcript(captured_events):
    """turn/diff/updated retains raw notifications in evidence but does not emit InfoEvents to transcript."""
    from inspect_ai.event import InfoEvent

    mapper = make_mapper(make_state())
    diff_text = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py"
    mapper.on_diff_updated("turn/diff/updated", {"diff": diff_text})

    diff_events = [e for e in captured_events if isinstance(e, InfoEvent) and e.source == "diff"]
    assert len(diff_events) == 0
    # Confirm notification is retained in evidence raw_events rather than the transcript.
    assert any(ev.get("method") == "turn/diff/updated" for ev in mapper.raw_events)




@pytest.mark.parametrize("thread", ["root", "child"])
def test_native_call_after_a_bridged_call_follows_its_result(captured_events, thread):
    """A response's calls complete in order; each tool result follows its own assistant."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
    from inspect_ai.tool import ToolCall

    mapper = make_mapper(make_state())
    mapper.on_item_completed("", completed_item("root", {
        "type": "subAgentActivity", "id": "sa", "kind": "started", "agentThreadId": "child", "agentPath": "/root/w"}))
    bridged = ToolCall(id="x", function="lookup", arguments={})
    mapper.inspect_tool_calls.add((thread, "x"))
    mapper.inspect_exchanges[(thread, "x")] = (ChatMessageAssistant(content="", tool_calls=[bridged]),
                                               [ChatMessageTool(content="found", tool_call_id="x", function="lookup")])
    for item in (command_item("c1"), {"type": "dynamicToolCall", "id": "x", "tool": "lookup", "arguments": {},
                                      "status": "completed", "success": True}, command_item("c2")):
        mapper.on_item_completed("", completed_item(thread, item))

    messages = mapper.messages if thread == "root" else mapper.subagents["child"].messages
    assert [(m.role, [c.id for c in getattr(m, "tool_calls", None) or []] or getattr(m, "tool_call_id", None))
            for m in messages[-6:]] == [("assistant", ["call_c1"]), ("tool", "call_c1"), ("assistant", ["x"]),
                                   ("tool", "x"), ("assistant", ["call_c2"]), ("tool", "call_c2")]


def raw_call(mapper, parent, call_id, name, arguments):
    mapper.on_raw_item("rawResponseItem/completed", {
        "threadId": parent,
        "item": {"type": "function_call", "call_id": call_id,
                 "name": name, "arguments": json.dumps(arguments)},
    })


def activity(mapper, parent, call_id, child, path, kind):
    mapper.on_item_completed("item/completed", {
        "threadId": parent,
        "item": {"type": "subAgentActivity", "id": call_id,
                 "agentThreadId": child, "agentPath": path, "kind": kind},
    })


def test_rejected_delegation_does_not_rewrite_or_deliver_input(captured_events):
    mapper = EventStreamMapper("test")
    mapper.root_thread_id = "root"
    raw_call(mapper, "root", "spawn", "spawn_agent", {
        "task_name": "worker", "message": "ACCEPTED_TASK",
    })
    activity(mapper, "root", "spawn", "child", "/root/worker", "started")

    # Native registry.rs rejects a second spawn with the same occupied path.
    raw_call(mapper, "root", "duplicate", "spawn_agent", {
        "task_name": "worker", "message": "REJECTED_DUPLICATE",
    })
    # Native message_tool.rs rejects whitespace before sending anything.
    raw_call(mapper, "root", "empty", "followup_task", {
        "target": "worker", "message": "   ",
    })
    for call_id in ("duplicate", "empty"):
        mapper.on_raw_item("rawResponseItem/completed", {
            "threadId": "root",
            "item": {"type": "function_call_output", "call_id": call_id,
                     "output": "Rejected by native tool"},
        })
    sub = mapper.subagents["child"]
    assert sub.prompt == "ACCEPTED_TASK"
    assert [message.text for message in sub.messages] == ["ACCEPTED_TASK"]
    assert [event.input for event in captured_events if isinstance(event, InputEvent)] == ["ACCEPTED_TASK"]

    # Valid control: input appears once native delivery succeeds.
    raw_call(mapper, "root", "followup", "followup_task", {
        "target": "worker", "message": "ACCEPTED_FOLLOWUP",
    })
    assert [message.text for message in sub.messages] == ["ACCEPTED_TASK"]
    activity(mapper, "root", "followup", "child", "/root/worker", "interacted")
    assert [message.text for message in sub.messages] == ["ACCEPTED_TASK", "ACCEPTED_FOLLOWUP"]


def test_native_activity_owns_nested_and_root_message_recipient(captured_events):
    mapper = EventStreamMapper("test")
    mapper.root_thread_id = "root"
    raw_call(mapper, "root", "spawn-parent", "spawn_agent", {
        "task_name": "worker", "message": "PARENT_TASK",
    })
    activity(mapper, "root", "spawn-parent", "parent", "/root/worker", "started")
    raw_call(mapper, "parent", "spawn-child", "spawn_agent", {
        "task_name": "worker", "message": "NESTED_TASK",
    })
    activity(mapper, "parent", "spawn-child", "child", "/root/worker/worker", "started")

    # AgentPath.resolve permits root-relative paths with several segments.
    raw_call(mapper, "root", "nested", "followup_task", {
        "target": "worker/worker", "message": "NESTED_FOLLOWUP",
    })
    activity(mapper, "root", "nested", "child", "/root/worker/worker", "interacted")
    raw_call(mapper, "child", "report", "send_message", {
        "target": "/root", "message": "ROOT_MESSAGE",
    })
    activity(mapper, "child", "report", "root", "/root", "interacted")

    assert [message.text for message in mapper.subagents["parent"].messages
            if message.role == "user"] == ["PARENT_TASK"]
    assert [message.text for message in mapper.subagents["child"].messages] == ["NESTED_TASK", "NESTED_FOLLOWUP"]
    assert [message.text for message in mapper.messages if message.role == "user"] == ["ROOT_MESSAGE"]
    assert [event.input for event in captured_events if isinstance(event, InputEvent)] == [
        "PARENT_TASK", "NESTED_TASK", "NESTED_FOLLOWUP", "ROOT_MESSAGE",
    ]
