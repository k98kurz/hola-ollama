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
    split_oversized_last_message,
    split_message,
    autocompact,
    COMPACTION_FAILED_MESSAGE,
    CONTINUED_MESSAGE_HEADER,
    DIGEST_MAX_TOTAL_CHARS,
    DIGEST_OMITTED_MARKER,
    TOOL_CALLS_HEADER,
    _format_tool_calls_digest,
)
from hola.session.models import Message as DBMessage
from hola.session.storage import initialize
from hola.tools.types import ToolCall, ToolCallFunction
from hola.types import Message, Role
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
        session. Message models and model config keys are overridden
        with `MODEL` so the fixture stays model-agnostic.
    """
    with open(path) as f:
        data = json.load(f)
    config = data['config']
    config['models'] = {'DEFAULT': MODEL}
    config['adapters'] = {MODEL: 'tool_enabled_chat_adapter'}
    session.config = config
    for msg in data['messages']:
        session.add_message(
            MODEL, msg['phase'],
            Message(role=Role[msg['role'].upper()], content=msg['content']),
        )


FIXTURE_DIR = Path(__file__).parent / 'fixtures'
COMPACTION_FIXTURE = FIXTURE_DIR / 'compaction_conversation.json'


def _tail_content_of(tail_section: str) -> str:
    """Extracts the retained content from a tail section, stripping
        the continuation header and any tool call digest.
    """
    assert tail_section.startswith(
        CONTINUED_MESSAGE_HEADER + '\n'
    ), tail_section[:80]
    rest = tail_section[len(CONTINUED_MESSAGE_HEADER) + 1:]
    marker = f'\n\n{TOOL_CALLS_HEADER}\n'
    if marker in rest:
        rest = rest.split(marker)[0]
    return rest

ADAPTERS = [
    'tool_enabled_chat_adapter',
    'xml_workaround_chat_adapter',
    'generate_endpoint_adapter',
]

MODEL = 'granite4.1:3b'


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


class TestSplitMessage(unittest.TestCase):
    """Tests for split_message."""
    def test_split_at_line_boundary(self):
        content = '\n'.join(
            f'line {i:03d}: some content for the fixture'
            for i in range(100)
        )
        m = Message(role=Role.USER, content=content)
        head, tail = split_message(m, 500)
        assert head.content + _tail_content_of(tail) == content
        assert head.content.endswith('\n'), head.content[-40:]
        assert len(tail) <= 500, len(tail)
        assert head.sequence == m.sequence
        assert head.role == Role.USER
        assert head.thinking is None

    def test_split_falls_back_to_space_boundary(self):
        content = ' '.join(['supercalifragilistic' * 3] * 100)
        m = Message(role=Role.USER, content=content)
        head, tail = split_message(m, 400)
        assert head.content + _tail_content_of(tail) == content
        assert head.content.endswith(' '), head.content[-40:]

    def test_split_exact_index_without_boundaries(self):
        content = 'a' * 1000
        m = Message(role=Role.USER, content=content)
        head, tail = split_message(m, 400)
        expected_tail = 400 - (len(CONTINUED_MESSAGE_HEADER) + 1)
        assert head.content == 'a' * (1000 - expected_tail)
        assert _tail_content_of(tail) == 'a' * expected_tail

    def test_tail_respects_budget_when_snap_overshoots(self):
        content = 'a' * 100 + '\n' + 'b' * 500
        m = Message(role=Role.USER, content=content)
        retain = len(CONTINUED_MESSAGE_HEADER) + 1 + 50
        head, tail = split_message(m, retain)
        body = _tail_content_of(tail)
        assert len(body) == 50, len(body)
        assert head.content + body == content

    def test_no_content_retained_when_budget_exhausted(self):
        m = Message(
            role=Role.USER, content='line one\nline two\nline three\nxx'
        )
        head, tail = split_message(m, len(CONTINUED_MESSAGE_HEADER) + 1)
        assert head.content == m.content
        assert _tail_content_of(tail) == ''

    def test_budget_exceeding_content_retains_all(self):
        content = 'hello world\nsecond line'
        m = Message(role=Role.USER, content=content)
        head, tail = split_message(m, 10**9)
        assert head.content == ''
        assert _tail_content_of(tail) == content

    def test_head_gets_fresh_token_estimate(self):
        content = '\n'.join(f'line {i}' for i in range(200))
        m = Message(
            role=Role.USER, content=content, estimated_tokens=99999,
        )
        head, tail = split_message(m, 500)
        assert head.estimated_tokens == 0
        assert head.tokens == 0
        assert head.id is None

    def test_no_digest_without_tool_calls(self):
        m = Message(role=Role.USER, content='abc\ndef')
        head, tail = split_message(m, 10**9)
        assert TOOL_CALLS_HEADER not in tail
        assert head.tool_calls is None

    def test_digest_lists_all_tool_calls(self):
        tc1 = ToolCall(
            call_id='c1',
            function=ToolCallFunction(
                name='search', arguments={'query': 'cats'},
            ),
        )
        tc2 = ToolCall(
            call_id='c2',
            function=ToolCallFunction(
                name='sum', arguments={'numbers': [1, 2, 3]},
            ),
        )
        content = 'part one\npart two\npart three\npart four'
        m = Message(
            role=Role.ASSISTANT, content=content, tool_calls=[tc1, tc2],
        )
        head, tail = split_message(m, 10**9)
        assert head.tool_calls is m.tool_calls
        assert len(head.tool_calls) == 2
        assert '- [search] [query: cats]' in tail
        assert '- [sum] [numbers: [1, 2, 3]]' in tail
        assert _tail_content_of(tail) == content

    def test_digest_renders_nested_tool_call_as_placeholder(self):
        inner = ToolCall(function=ToolCallFunction(
            name='get_number', arguments={},
        ))
        outer = ToolCall(function=ToolCallFunction(
            name='process', arguments={'data': inner},
        ))
        m = Message(
            role=Role.ASSISTANT, content='x', tool_calls=[outer],
        )
        head, tail = split_message(m, 10**9)
        assert '- [process] [data: <nested: get_number>]' in tail
        assert 'ToolCallFunction' not in tail
        assert head.tool_calls == [outer]

    def test_digest_truncates_long_argument_values(self):
        tc = ToolCall(function=ToolCallFunction(
            name='write', arguments={'content': 'y' * 500},
        ))
        m = Message(role=Role.ASSISTANT, content='x', tool_calls=[tc])
        head, tail = split_message(m, 10**9)
        digest_line = [
            line for line in tail.splitlines()
            if line.startswith('- [write]')
        ][0]
        assert len(digest_line) < 200, len(digest_line)
        assert digest_line.endswith('...]')
        assert head.tool_calls == [tc]

    def test_digest_caps_many_tool_calls(self):
        calls = [
            ToolCall(
                call_id=f'c{i}',
                function=ToolCallFunction(
                    name=f'op{i}', arguments={'x': i},
                ),
            )
            for i in range(200)
        ]
        digest = _format_tool_calls_digest(calls)
        assert len(digest) <= DIGEST_MAX_TOTAL_CHARS + 64, len(digest)
        omitted = len(calls) - digest.count('\n- [')
        assert omitted > 0
        assert digest.endswith(DIGEST_OMITTED_MARKER.format(omitted))

    def test_digest_hard_truncates_oversized_first_call(self):
        tc = ToolCall(
            call_id='c1',
            function=ToolCallFunction(
                name='big',
                arguments={f'p{i}': 'y' * 100 for i in range(20)},
            ),
        )
        digest = _format_tool_calls_digest([tc])
        assert len(digest) <= DIGEST_MAX_TOTAL_CHARS, len(digest)
        assert digest.startswith(TOOL_CALLS_HEADER + '\n- [big]')
        assert digest.endswith('...')

    def test_digest_unchanged_when_under_cap(self):
        tc = ToolCall(
            call_id='c1',
            function=ToolCallFunction(
                name='search', arguments={'query': 'cats'},
            ),
        )
        digest = _format_tool_calls_digest([tc])
        assert digest == f'{TOOL_CALLS_HEADER}\n- [search] [query: cats]'


class TestSplitOversizedLastMessage(SessionTestCase):
    """Tests for split_oversized_last_message."""
    def _huge_content(self, lines: int = 50) -> str:
        return '\n'.join(
            f'line {i:03d}: ' + ('x' * 70) for i in range(lines)
        )

    def _add_history(self):
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='hello world')
        )
        self.session.add_message(
            'm', 'TEST',
            Message(role=Role.USER, content=self._huge_content()),
        )

    def test_splits_oversized_last_message(self):
        self.session.config = {
            'context_threshold': 800,
            'context_compaction_target': {'TEST': 500},
        }
        self._add_history()
        history = self.session.get_history('TEST')
        first, last = history
        assert len(last.content) > 4000, len(last.content)
        to_compact = get_messages_to_compact(self.session, 'TEST')
        result, tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        assert tail is not None
        assert [m.sequence for m in result] == [
            first.sequence, last.sequence
        ]
        head = result[-1]
        assert head.sequence == last.sequence
        assert head.content != last.content
        assert head.content + _tail_content_of(tail) == last.content
        assert head.content.endswith('\n')
        assert not head.is_compaction
        assert _tail_content_of(tail).startswith('line 0')

    def test_no_split_when_last_message_fits(self):
        self.session.config = {
            'context_threshold': 100000,
            'context_compaction_target': {'TEST': 500},
        }
        self._add_history()
        to_compact = get_messages_to_compact(self.session, 'TEST')
        result, tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        assert tail is None
        assert result == to_compact

    def test_reserve_clamped_to_half_threshold(self):
        self.session.config = {
            'context_threshold': 100,
            'context_compaction_target': {'TEST': 1000},
            'context_compaction_reserve': 1000,
        }
        self._add_history()
        to_compact = get_messages_to_compact(self.session, 'TEST')
        result, tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        # Reserve clamps to 50, so budget = 100 - 50 - 0 = 50 tokens.
        assert tail is not None
        assert len(_tail_content_of(tail)) <= 50 * 4

    def test_no_split_when_other_retained_exceeds_budget(self):
        self.session.config = {
            'context_threshold': 100,
            'context_compaction_target': {'TEST': 1},
        }
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='hello world')
        )
        self.session.add_message(
            'm', 'TEST',
            Message(role=Role.USER, content=self._huge_content()),
        )
        self.session.add_message(
            'm', 'TEST', Message(role=Role.USER, content='latest news')
        )
        to_compact = get_messages_to_compact(self.session, 'TEST')
        # Target 1 selects only 'hello world'; the huge middle message
        # is retained unselected and alone keeps the budget nonpositive.
        assert [m.content for m in to_compact] == ['hello world']
        result, tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        assert tail is None
        assert result == to_compact

    def _set_last_tokens(self, tokens: int) -> None:
        seq = self.session.get_history('TEST')[-1].sequence
        self.session.dbsession.messages().query().equal(
            'sequence', seq
        ).update({'tokens': tokens})

    def test_measured_tokens_shrink_retained_tail(self):
        self.session.config = {
            'context_threshold': 800,
            'context_compaction_target': {'TEST': 500},
        }
        self._add_history()
        to_compact = get_messages_to_compact(self.session, 'TEST')
        _, plain_tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        self._set_last_tokens(3000)
        to_compact = get_messages_to_compact(self.session, 'TEST')
        _, dense_tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        assert plain_tail is not None
        assert dense_tail is not None
        assert len(dense_tail) < len(plain_tail), (
            len(dense_tail), len(plain_tail)
        )

    def test_reserve_override_changes_retained_size(self):
        config = {
            'context_threshold': 800,
            'context_compaction_target': {'TEST': 500},
            'context_compaction_reserve': 50,
        }
        self.session.config = config
        self._add_history()
        to_compact = get_messages_to_compact(self.session, 'TEST')
        _, small_reserve_tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        # The config setter merges keys, so this only bumps the reserve.
        self.session.config = {'context_compaction_reserve': 600}
        to_compact = get_messages_to_compact(self.session, 'TEST')
        _, big_reserve_tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        assert small_reserve_tail is not None
        assert big_reserve_tail is not None
        assert len(small_reserve_tail) > len(big_reserve_tail), (
            len(small_reserve_tail), len(big_reserve_tail)
        )

    def test_tool_calls_dominated_message_splits_to_digest(self):
        self.session.config = {
            'context_threshold': 400,
            'context_compaction_target': {'TEST': 200},
        }
        tc = ToolCall(
            call_id='c1',
            function=ToolCallFunction(
                name='bigwrite', arguments={'content': 'y' * 2000},
            ),
        )
        self.session.add_message(
            'm', 'TEST',
            Message(role=Role.ASSISTANT, content='x', tool_calls=[tc]),
        )
        to_compact = get_messages_to_compact(self.session, 'TEST')
        result, tail = split_oversized_last_message(
            self.session, 'TEST', to_compact
        )
        assert tail is not None
        head = result[-1]
        assert head.sequence == to_compact[0].sequence
        assert head.content == ''
        assert head.tool_calls is not None
        assert head.tool_calls[0].function.name == 'bigwrite'
        assert '- [bigwrite]' in tail
        assert len(_tail_content_of(tail)) < 300


class TestAutocompactAdapterResolution(unittest.TestCase):
    """Autocompact adapter lookup resolves via the ADAPTERS registry
        and rejects unknown names. No Ollama calls.
    """
    def setUp(self):
        initialize(True)
        self.session = Session('adapter_test', portable=True)

    def tearDown(self):
        _clean_hola()

    def test_unknown_adapter_raises_value_error(self):
        self.session.add_message(
            MODEL, 'TEST', Message(role=Role.USER, content='compact me')
        )
        self.session.config = {
            'models': {'DEFAULT': MODEL},
            'adapters': {MODEL: 'not_a_real_adapter'},
        }
        with self.assertRaises(ValueError):
            asyncio.run(autocompact(
                self.session, None, 'TEST', force=True, request="Test"
            ))


class TestAutocompactRetries(unittest.TestCase):
    """Autocompact retry and backoff behavior, exercised offline with a
        `None` client so no Ollama calls are made. The `None` client
        raises `AttributeError` on every attempt.
    """
    def setUp(self):
        initialize(True)
        self.session = Session('retry_test', portable=True)

    def tearDown(self):
        _clean_hola()

    def _configure(self, **overrides):
        config = {'models': {'DEFAULT': MODEL}}
        config.update(overrides)
        self.session.config = config
        self.session.add_message(
            MODEL, 'TEST', Message(role=Role.USER, content='compact me')
        )

    def test_retries_then_falls_back(self):
        self._configure(
            compaction_retries=1, compaction_retry_delay=0,
            compaction_retry_delay_mult=2.0,
        )
        with self.assertLogs(
            'hola.session.compaction', level='WARNING'
        ) as cm:
            result = asyncio.run(autocompact(
                self.session, None, 'TEST', force=True, request="Test"
            ))
        assert result is not None
        assert COMPACTION_FAILED_MESSAGE in result.content, result.content
        warnings = [
            r.getMessage() for r in cm.records if r.levelname == 'WARNING'
        ]
        assert len(warnings) == 2, warnings
        assert 'attempt 1/2' in warnings[0], warnings
        assert 'attempt 2/2' in warnings[1], warnings
        assert all(
            w.split('): ', 1)[-1].strip() for w in warnings
        ), warnings
        history = self.session.get_history(
            'TEST', only_since_last_compaction=False
        )
        assert any(m.is_compaction for m in history), history

    def test_backoff_respects_delay_and_multiplier(self):
        self._configure(
            compaction_retries=3, compaction_retry_delay=0.05,
            compaction_retry_delay_mult=3.0,
        )
        start = perf_counter()
        with self.assertLogs('hola.session.compaction', level='WARNING'):
            asyncio.run(autocompact(
                self.session, None, 'TEST', force=True, request="Test"
            ))
        elapsed = perf_counter() - start
        # two sleeps: 0.05 + 0.15; a multiplier of 1 would total 0.10
        assert elapsed >= 0.18, elapsed

    def test_debug_reraises_without_retry(self):
        self._configure(compaction_retries=3, compaction_retry_delay=0.05)
        with self.assertLogs(
            'hola.session.compaction', level='WARNING'
        ) as cm:
            with self.assertRaises(AttributeError):
                asyncio.run(autocompact(
                    self.session, None, 'TEST', force=True, debug=True,
                    request="Test",
                ))
        warnings = [r for r in cm.records if r.levelname == 'WARNING']
        assert len(warnings) == 1, [r.getMessage() for r in cm.records]


class TestAutocompact(unittest.IsolatedAsyncioTestCase):
    """Real autocompact tests using a fixture and live Ollama."""
    def setUp(self):
        initialize(True)
        self.session = Session('autocompact_test', portable=True)
        self.client = OllamaClient(timeout=600.0)

    async def asyncTearDown(self):
        await self.client.close()
        _clean_hola()

    @classmethod
    def tearDownClass(cls):
        async def _unload():
            client = OllamaClient()
            try:
                await client.unload_model(MODEL)
            finally:
                await client.close()
        asyncio.run(_unload())

    async def test_below_threshold_returns_none(self):
        self.session.add_message(
            MODEL, 'TEST', Message(role=Role.USER, content='tiny')
        )
        result = await autocompact(
            self.session, None, 'TEST', debug=True, request="Test"
        )
        assert result is None

    async def test_force_compacts_below_threshold(self):
        self.session.config = {'models': {'DEFAULT': MODEL}}
        self.session.add_message(
            MODEL, 'TEST', Message(role=Role.USER, content='tiny')
        )
        result = await autocompact(
            self.session, self.client, 'TEST', debug=True, force=True,
            request="Test",
        )
        assert result is not None
        assert result.role == Role.SYSTEM, result.role
        assert 'Compaction failed' not in result.content, result.content
        history = self.session.get_history(
            'TEST', only_since_last_compaction=True
        )
        assert history[0].is_compaction is True
        assert history[0].content == result.content
        all_history = self.session.get_history(
            'TEST', only_since_last_compaction=False
        )
        compacted_by = {m.compacted_by for m in all_history if m.compacted_by}
        assert history[0].id in compacted_by

    async def test_force_respects_compaction_target(self):
        self.session.config = {
            'models': {'DEFAULT': MODEL},
            'context_compaction_target': {'TEST': 1},
        }
        for i in range(3):
            self.session.add_message(
                MODEL, 'TEST', Message(role=Role.USER, content=f'number {i}')
            )
        result = await autocompact(
            self.session, self.client, 'TEST', debug=True, force=True,
            request="Test",
        )
        assert result is not None
        all_history = self.session.get_history(
            'TEST', only_since_last_compaction=False
        )
        compacted = [m for m in all_history if m.compacted_by]
        unmarked = [
            m for m in all_history
            if not m.compacted_by and not m.is_compaction
        ]
        assert len(compacted) == 1, [m.content for m in compacted]
        assert compacted[0].content == 'number 0', compacted[0].content
        assert len(unmarked) == 2, [m.content for m in unmarked]

    async def test_force_with_empty_history_returns_none(self):
        result = await autocompact(
            self.session, self.client, 'TEST', debug=True, force=True,
            request="Test",
        )
        assert result is None

    async def test_force_returns_existing_compaction_when_only_message(self):
        self.session.config = {'models': {'DEFAULT': MODEL}}
        self.session.add_message(
            MODEL, 'TEST', Message(role=Role.USER, content='original')
        )
        seqs = [
            m.sequence for m in self.session.get_history(
                'TEST', only_since_last_compaction=False
            )
        ]
        self.session.insert_compaction(
            MODEL, 'TEST', 'existing summary', seqs
        )
        result = await autocompact(
            self.session, None, 'TEST', debug=True, force=True,
            request="Test",
        )
        assert result is not None
        assert result.is_compaction is True
        assert result.content == 'existing summary'
        all_history = self.session.get_history(
            'TEST', only_since_last_compaction=False
        )
        compactions = [m for m in all_history if m.is_compaction]
        assert len(compactions) == 1, len(compactions)

    def _add_oversized_history(self) -> str:
        self.session.add_message(
            MODEL, 'TEST',
            Message(role=Role.USER, content='Plan the work carefully.'),
        )
        huge = '\n'.join([
            f'chapter {i:03d}: ' + ('detail ' * 11) for i in range(240)
        ])
        self.session.add_message(
            MODEL, 'TEST', Message(role=Role.USER, content=huge),
        )
        return huge

    def _tail_section_of(self, content: str) -> str:
        return content.split(
            f'\n\n{CONTINUED_MESSAGE_HEADER}\n', 1
        )[1]

    async def test_autocompact_splits_oversized_last_message(self):
        self.session.config = {'models': {'DEFAULT': MODEL}}
        self._add_oversized_history()
        result = await autocompact(
            self.session, self.client, 'TEST', debug=True, force=True,
            request="Test",
        )
        assert result is not None
        assert result.role == Role.SYSTEM, result.role
        assert 'Compaction failed' not in result.content, result.content
        assert CONTINUED_MESSAGE_HEADER in result.content
        tail_section = self._tail_section_of(result.content)
        assert 'chapter 239:' in tail_section
        assert 'chapter 000:' not in tail_section
        assert 'chapter 150:' in tail_section
        history = self.session.get_history('TEST')
        assert len(history) == 1
        assert history[0].is_compaction is True
        assert history[0].content == result.content
        all_history = self.session.get_history(
            'TEST', only_since_last_compaction=False
        )
        compacted = {
            m.compacted_by for m in all_history if m.compacted_by
        }
        assert history[0].id in compacted
        assert len(compacted) == 1, len(compacted)
        post_size = calculate_context_size(history)
        assert post_size < 4096, post_size

    async def test_autocompact_fallback_retains_tail(self):
        self.session.config = {
            'models': {'DEFAULT': 'nonexistent-model'},
            'adapters': {'nonexistent-model': 'tool_enabled_chat_adapter'},
        }
        self._add_oversized_history()
        result = await autocompact(
            self.session, self.client, 'TEST', debug=False, force=True,
            request="Test",
        )
        assert result is not None
        assert result.role == Role.SYSTEM, result.role
        assert COMPACTION_FAILED_MESSAGE in result.content
        tail_section = self._tail_section_of(result.content)
        assert 'chapter 239:' in tail_section
        assert 'chapter 000:' not in tail_section
        history = self.session.get_history('TEST')
        assert len(history) == 1
        assert history[0].is_compaction is True
        assert history[0].content == result.content
        all_history = self.session.get_history(
            'TEST', only_since_last_compaction=False
        )
        compacted = {
            m.compacted_by for m in all_history if m.compacted_by
        }
        assert history[0].id in compacted

    async def _run_autocompact(
            self, adapter_name, *,
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
            config['compaction_retry_delay'] = 0
        else:
            config['adapters'] = {MODEL: adapter_name}
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
                with self.assertLogs(
                    'hola.session.compaction', level='WARNING'
                ) as cm:
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

                    messages = [r.getMessage() for r in cm.records]
                    assert any('compaction' in m.lower() for m in messages), (
                        messages)
                    warnings = [
                        m for m in messages if 'failed (attempt' in m
                    ]
                    assert len(warnings) >= 2, messages
                    assert all(
                        m.split('): ', 1)[-1].strip() for m in warnings
                    ), warnings
                finally:
                    _clean_hola()


if __name__ == "__main__":
    unittest.main()
