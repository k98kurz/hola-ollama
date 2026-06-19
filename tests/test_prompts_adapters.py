"""
Test adapter functions for transforming Message lists.
"""

from hola.prompts.adapters import (
    tool_enabled_chat_adapter,
    xml_workaround_chat_adapter,
    generate_endpoint_adapter,
)
from hola.types import ChatRequest, GenerateRequest, Message, Role
from hola.tools.types import ToolDefinition
import unittest


class TestAdapters(unittest.TestCase):
    """Test adapter functions."""
    def setUp(self):
        self.phase_instructions = "Respond helpfully."
        self.user_msg = Message(role=Role.USER, content="Hello")
        self.asst_msg = Message(
            role=Role.ASSISTANT, content="Hi there!",
            thinking="be polite", tool_calls=None,
        )
        self.sys_msg = Message(role=Role.SYSTEM, content="Be concise.")
        self.tool_msg = Message(role=Role.TOOL, content="42")
        self.tool_def = ToolDefinition(
            name='test_tool', description='A test tool',
            parameters={'type': 'object', 'properties': {
                'input': {'type': 'string'},
            }},
        )

    # --- tool_enabled_chat_adapter ---
    def test_tool_enabled_basic(self):
        result = tool_enabled_chat_adapter(
            self.phase_instructions, [self.user_msg, self.tool_msg], [],
        )
        assert isinstance(result, ChatRequest), type(result)
        assert result.model == 'granite4.1:8b', result.model
        assert result.messages[0].role == Role.SYSTEM, result.messages[0].role
        assert self.phase_instructions in result.messages[0].content
        assert result.messages[1].role == Role.USER
        assert result.messages[1].content == "Hello"
        assert result.messages[2].role == Role.TOOL
        assert result.messages[2].content == "42"
        assert result.tools is None

    def test_tool_enabled_combines_system(self):
        extra_sys = Message(role=Role.SYSTEM, content="Extra rule.")
        messages = [self.sys_msg, self.user_msg, extra_sys]
        result = tool_enabled_chat_adapter(
            self.phase_instructions, messages, [],
        )
        combined = result.messages[0].content
        assert "Be concise." in combined, combined
        assert "Extra rule." in combined, combined
        assert "Respond helpfully." in combined, combined
        # Only one system message in the output list
        sys_count = sum(
            1 for m in result.messages if m.role == Role.SYSTEM
        )
        assert sys_count == 1, sys_count

    def test_tool_enabled_preserves_order(self):
        messages = [self.sys_msg, self.user_msg, self.asst_msg, self.sys_msg]
        result = tool_enabled_chat_adapter(
            self.phase_instructions, messages, [],
        )
        assert len(result.messages) == 3, len(result.messages)
        # Index 0 is system, 1 is user, 2 is assistant
        assert result.messages[1].content == "Hello"
        assert result.messages[2].content == "Hi there!"

    def test_tool_enabled_tools(self):
        result = tool_enabled_chat_adapter(
            self.phase_instructions, [self.user_msg],
            [self.tool_def],
        )
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0].name == 'test_tool'

        # Empty tools list produces None
        result = tool_enabled_chat_adapter(
            self.phase_instructions, [self.user_msg], [],
        )
        assert result.tools is None

    def test_tool_enabled_rag_cag_model(self):
        result = tool_enabled_chat_adapter(
            self.phase_instructions, [self.user_msg], [],
            model_name='custom-model', rag_cag='<rag>context</rag>',
        )
        assert result.model == 'custom-model', result.model
        assert '<rag>context</rag>' in result.messages[0].content

    def test_tool_enabled_preserves_details(self):
        result = tool_enabled_chat_adapter(
            self.phase_instructions,
            [self.user_msg, self.asst_msg], [],
        )
        assert result.messages[2].thinking == 'be polite'
        assert result.messages[2].tool_calls is None

    # --- xml_workaround_chat_adapter ---
    def test_xml_workaround_basic(self):
        result = xml_workaround_chat_adapter(
            self.phase_instructions, [self.user_msg, self.tool_msg], [],
        )
        assert isinstance(result, ChatRequest), type(result)
        assert result.model == 'granite4.1:8b', result.model
        assert result.tools is None
        assert self.phase_instructions in result.messages[0].content
        assert result.messages[1].role == Role.USER
        assert result.messages[1].content == "Hello"
        assert result.messages[2].role == Role.TOOL
        assert result.messages[2].content == "42"

    def test_xml_workaround_combines_system(self):
        extra_sys = Message(role=Role.SYSTEM, content="Extra rule.")
        messages = [self.sys_msg, self.user_msg, extra_sys]
        result = xml_workaround_chat_adapter(
            self.phase_instructions, messages, [],
        )
        combined = result.messages[0].content
        assert "Be concise." in combined
        assert "Extra rule." in combined
        sys_count = sum(
            1 for m in result.messages if m.role == Role.SYSTEM
        )
        assert sys_count == 1, sys_count

    def test_xml_workaround_with_tools(self):
        messages = [self.sys_msg, self.user_msg]
        result = xml_workaround_chat_adapter(
            self.phase_instructions, messages,
            [self.tool_def],
        )
        assert result.tools is None
        assert '<tools>' in result.messages[0].content
        assert 'test_tool' in result.messages[0].content
        assert 'can call tools' in result.messages[0].content.lower()

    def test_xml_workaround_no_tools(self):
        messages = [self.sys_msg, self.user_msg]
        result = xml_workaround_chat_adapter(
            self.phase_instructions, messages, [],
        )
        assert '<tools>' not in result.messages[0].content
        assert 'can call tools' not in result.messages[0].content.lower()

    def test_xml_workaround_rag_cag_order(self):
        messages = [self.sys_msg, self.user_msg, self.asst_msg]
        result = xml_workaround_chat_adapter(
            self.phase_instructions, messages, [],
            rag_cag='<rag>context</rag>',
        )
        combined = result.messages[0].content
        assert '<rag>context</rag>' in combined
        # Rag appears before combined system messages
        idx_rag = combined.index('<rag>context</rag>')
        idx_sys = combined.index('Be concise.')
        assert idx_rag < idx_sys, f'rag at {idx_rag}, sys at {idx_sys}'
        # Non-system messages in correct order after index 0
        assert len(result.messages) == 3, len(result.messages)

    # --- generate_endpoint_adapter ---
    def test_generate_endpoint_basic(self):
        messages = [self.sys_msg, self.user_msg, self.asst_msg, self.tool_msg]
        result = generate_endpoint_adapter(
            self.phase_instructions, messages, [],
        )
        assert isinstance(result, GenerateRequest), type(result)
        assert result.model == 'granite4.1:8b', result.model
        assert self.phase_instructions in (result.system or '')
        assert '<|im_start|>user' in result.prompt
        assert '<|im_start|>assistant' in result.prompt
        assert '<|im_start|>tool' in result.prompt
        assert '42' in result.prompt
        assert '<|im_end|>' in result.prompt

    def test_generate_endpoint_combines_system(self):
        extra_sys = Message(role=Role.SYSTEM, content="Extra rule.")
        messages = [self.sys_msg, self.user_msg, extra_sys]
        result = generate_endpoint_adapter(
            self.phase_instructions, messages, [],
        )
        assert 'Be concise.' in (result.system or '')
        assert 'Extra rule.' in (result.system or '')
        # Non-system messages in prompt
        assert 'Hello' in result.prompt

    def test_generate_endpoint_with_tools(self):
        messages = [self.sys_msg, self.user_msg]
        result = generate_endpoint_adapter(
            self.phase_instructions, messages,
            [self.tool_def],
        )
        assert '<tools>' in (result.system or '')
        # Prompt should not contain tool XML
        assert '<tools>' not in result.prompt

    def test_generate_endpoint_rag_cag(self):
        messages = [self.sys_msg, self.user_msg, self.asst_msg]
        result = generate_endpoint_adapter(
            self.phase_instructions, messages, [],
            rag_cag='<rag>context</rag>',
        )
        assert '<rag>context</rag>' in (result.system or '')
        idx_rag = result.system.index('<rag>context</rag>')
        idx_sys = result.system.index('Be concise.')
        assert idx_rag < idx_sys, f'rag at {idx_rag}, sys at {idx_sys}'
        assert '<|im_start|>user' in result.prompt
        assert '<|im_start|>assistant' in result.prompt

    def test_generate_endpoint_system_none(self):
        result = generate_endpoint_adapter(
            '', [self.user_msg], [],
        )
        assert result.system == '', repr(result.system)

    def test_generate_endpoint_no_non_system(self):
        messages = [self.sys_msg, Message(role=Role.SYSTEM, content="More.")]
        result = generate_endpoint_adapter(
            self.phase_instructions, messages, [],
        )
        assert result.prompt == '', repr(result.prompt)
        assert 'Be concise.' in (result.system or '')
        assert 'More.' in (result.system or '')


if __name__ == '__main__':
    unittest.main()
