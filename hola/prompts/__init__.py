"""
Prompt generation and adaptation system for LLM interactions.
"""

from hola.prompts.adapters import (
    generate_endpoint_adapter,
    tool_enabled_chat_adapter,
    xml_workaround_chat_adapter,
)
from hola.prompts.generators import (
    prompt_generator,
    generate_prompt,
    generate_compaction_prompt,
    generate_done_prompt,
    generate_executing_prompt,
    generate_initial_planning_prompt,
    generate_planning_finalization_prompt,
    generate_planning_task_prompt,
    generate_planning_task_remediation_prompt,
    generate_reviewing_final_prompt,
    generate_reviewing_task_prompt,
    interpolate_template,
)
from hola.types import Adapter, PromptGenerator
