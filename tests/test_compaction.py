"""
Tests for compaction logic.
"""

from math import ceil
from pathlib import Path
from time import perf_counter
from hola.context import estimate_tokens, calculate_context_size
from hola.ollama.client import OllamaClient
from hola.session import Session
from hola.session.compaction import (
    check_context_threshold,
    get_messages_to_compact,
    autocompact,
    COMPACTION_FAILED_MESSAGE,
)
from hola.session.models import Message as DBMessage
from hola.session.storage import initialize
from hola.types import Message, Role
from hola.tools.types import ToolCall, ToolCallFunction
import asyncio
import json
import shutil
import unittest
import packify


def _clean_hola():
    hola_dir = Path.cwd() / '.hola'
    if hola_dir.exists():
        shutil.rmtree(hola_dir)


def load_fixture(session, path):
    """Load a compaction test fixture from a JSON file into a
        session.
    """
    with open(path) as f:
        data = json.load(f)
    session.config = data['config']
    for msg in data['messages']:
        session.add_message(
            msg['model'], msg['phase'],
            Message(role=Role[msg['role'].upper()], content=msg['content']),
        )


FIXTURE_DIR = Path(__file__).parent / 'fixtures'
COMPACTION_FIXTURE = FIXTURE_DIR / 'compaction_conversation.json'

ADAPTERS = [
    'tool_enabled_chat_adapter',
    'xml_workaround_chat_adapter',
    'generate_endpoint_adapter',
]


class TestEstimateTokens(unittest.TestCase):
    """Tests for estimate_tokens."""
    def test_empty_content_has_minimal_estimate(self):
        m = Message(role=Role.USER, content='')
        assert estimate_tokens(m) == 1  # 1 byte for packed None

    def test_content_scales_estimate_linearly(self):
        m = Message(role=Role.USER, content='a' * 10)
        assert estimate_tokens(m) == 3  # ceil((10 + 1) / 4)
        m2 = Message(role=Role.USER, content='a' * 20)
        assert estimate_tokens(m2) == 6  # ceil((20 + 1) / 4)

    def test_tool_calls_add_packed_size(self):
        tc = ToolCall(
            call_id='1',
            function=ToolCallFunction(name='test', arguments={}),
        )
        m = Message(role=Role.ASSISTANT, content='hi', tool_calls=[tc])
        result = estimate_tokens(m)
        content_chars = len('hi')
        packed_len = len(packify.pack([tc]))
        expected = ceil((content_chars + packed_len) / 4)
        assert result == expected, (result, expected)


class TestEstimateTokensDBMessage(unittest.TestCase):
    """Tests for estimate_tokens with DBMessage objects."""
    def setUp(self):
        initialize(True)

    def tearDown(self):
        _clean_hola()

    def test_dbmessage_uses_details_bytes(self):
        m = DBMessage({
            'session_id': 'test',
            'sequence': 1,
            'content': 'hi',
        })
        m.details = {'tool_calls': None}
        result = estimate_tokens(m)
        content_chars = len('hi')
        packed_len = len(m.data.get('details', b''))
        expected = ceil((content_chars + packed_len) / 4)
        assert result == expected, (result, expected)

    def test_dbmessage_without_details_falls_back_to_empty_bytes(self):
        m = DBMessage({
            'session_id': 'test',
            'sequence': 1,
            'content': 'hello',
        })
        result = estimate_tokens(m)
        expected = ceil(len('hello') / 4)
        assert result == expected, (result, expected)


class TestCalculateContextSize(unittest.TestCase):
    """Tests for calculate_context_size."""
    def test_empty_list_returns_zero(self):
        assert calculate_context_size([]) == 0

    def test_fallback_to_estimate_tokens(self):
        m = Message(role=Role.USER, content='hello world')
        result = calculate_context_size([m])
        assert result == ceil(len('hello world') / 4)

    def test_estimated_tokens_field_used_when_set(self):
        m = Message(role=Role.USER, content='ignored', estimated_tokens=42)
        assert calculate_context_size([m]) == 42

    def test_tokens_field_takes_highest_priority(self):
        m = Message(
            role=Role.USER, content='ignored',
            tokens=100, estimated_tokens=42,
        )
        assert calculate_context_size([m]) == 100

    def test_mixed_list_sums_correctly(self):
        m1 = Message(role=Role.USER, content='a', tokens=5)
        m2 = Message(role=Role.USER, content='hello world', estimated_tokens=8)
        m3 = Message(role=Role.USER, content='test')
        total = calculate_context_size([m1, m2, m3])
        assert total == 15, total


class SessionTestCase(unittest.TestCase):
    """Base class for tests that need a real portable session."""
    def setUp(self):
        initialize(True)
        self.session = Session('compaction_test', portable=True)

    def tearDown(self):
        _clean_hola()


class TestCheckContextThreshold(SessionTestCase):
    """Tests for check_context_threshold."""
    def test_below_threshold_returns_false(self):
        self.session.config = {'context_threshold': 10000}
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='small')
        )
        assert check_context_threshold(self.session, 'TEST') is False

    def test_above_threshold_returns_true(self):
        self.session.config = {'context_threshold': 1}
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='enough')
        )
        assert check_context_threshold(self.session, 'TEST') is True

    def test_empty_history_returns_false(self):
        self.session.config = {'context_threshold': 1}
        assert check_context_threshold(self.session, 'TEST') is False

    def test_custom_config_overrides_defaults(self):
        self.session.config = {
            'context_threshold_ratio': 0.1,
            'max_context': {'TEST': 100},
        }
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='x' * 200)
        )
        assert check_context_threshold(self.session, 'TEST') is True


class TestGetMessagesToCompact(SessionTestCase):
    """Tests for get_messages_to_compact."""
    def test_accumulates_oldest_messages_up_to_target(self):
        self.session.config = {'context_compaction_target': {'TEST': 1}}
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='first')
        )
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='second')
        )
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='third')
        )
        result = get_messages_to_compact(self.session, 'TEST')
        assert len(result) == 1
        assert result[0].content == 'first'

    def test_all_messages_returned_when_target_exceeds_total(self):
        self.session.config = {'context_compaction_target': {'TEST': 100000}}
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='first')
        )
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='second')
        )
        result = get_messages_to_compact(self.session, 'TEST')
        assert len(result) == 2

    def test_empty_history_returns_empty_list(self):
        result = get_messages_to_compact(self.session, 'TEST')
        assert result == []

    def test_custom_compaction_ratio(self):
        self.session.config = {
            'context_compaction_ratio': 0.5,
            'max_context': {'TEST': 10},
            'context_compaction_target': {'TEST': 5},
        }
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='first')
        )
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='second')
        )
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='third')
        )
        result = get_messages_to_compact(self.session, 'TEST')
        assert len(result) == 1
        assert result[0].content == 'first'


class TestAutocompact(unittest.IsolatedAsyncioTestCase):
    """Real autocompact tests using a fixture and live Ollama."""
    def setUp(self):
        initialize(True)
        self.session = Session('autocompact_test', portable=True)
        self.client = OllamaClient()

    async def asyncTearDown(self):
        await self.client.close()
        _clean_hola()

    async def test_below_threshold_returns_none(self):
        self.session.add_message(
            'granite4.1:3b', 'TEST', Message(role=Role.USER, content='tiny')
        )
        result = await autocompact(
            self.session, None, 'TEST', debug=True, request="Test"
        )
        assert result is None

    async def _run_autocompact(self, adapter_name, *,
            request="Outline a new software system.",
            debug=False, fail=False,
        ) -> tuple[Session, Message | None]:
        initialize(True)
        session = Session('autocompact_test', portable=True)
        load_fixture(session, COMPACTION_FIXTURE)
        config = session.config
        if fail:
            config['models'] = {'DEFAULT': 'nonexistent-model'}
            config['adapters'] = {'nonexistent-model': adapter_name}
        else:
            config['adapters'] = {'granite4.1:3b': adapter_name}
        session.config = config
        result = await autocompact(
            session, self.client, 'EXECUTING', debug=debug, request=request,
        )
        return session, result

    async def test_autocompact_basic(self):
        for name in ADAPTERS:
            with self.subTest(adapter=name):
                try:
                    start = perf_counter()
                    session, result = await self._run_autocompact(name, debug=True)
                    end = perf_counter()
                    assert result is not None, f'{name} returned None'
                    assert result.role == Role.SYSTEM, result.role
                    assert 'Compaction failed' not in result.content, result
                    print(f"\nCompaction result ({name}: {end-start:.2f}s):")
                    print(result.content)

                    history = session.get_history(
                        'EXECUTING', only_since_last_compaction=True
                    )
                    assert len(history) >= 1
                    compact_msg = history[0]
                    assert compact_msg.content == result.content
                    assert compact_msg.is_compaction is True

                    all_history = session.get_history(
                        'EXECUTING', only_since_last_compaction=False
                    )
                    compacted = {
                        m.compacted_by for m in all_history if m.compacted_by
                    }
                    assert compact_msg.id in compacted
                finally:
                    _clean_hola()

    async def test_autocompact_fallback(self):
        for name in ADAPTERS:
            with self.subTest(adapter=name):
                session, result = await self._run_autocompact(
                    name, fail=True,
                )
                try:
                    assert result is not None, f'{name} returned None'
                    assert result.role == Role.SYSTEM, result.role
                    assert COMPACTION_FAILED_MESSAGE in result.content

                    history = session.get_history(
                        'EXECUTING', only_since_last_compaction=True
                    )
                    assert len(history) >= 1
                    assert history[0].is_compaction is True
                    assert COMPACTION_FAILED_MESSAGE in history[0].content
                finally:
                    _clean_hola()


if __name__ == "__main__":
    unittest.main()
