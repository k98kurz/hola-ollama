"""
Prompt generator functions for different workflow phases.
"""

from string import Template


PLANNING_PROMPT_PREFIX = """You are a workflow planning agent preparing a plan \
to fulfill the request.
Break it down into clear, actionable steps using the task tools. The number of \
tasks should be commensurate with the complexity of the request. Any details \
for a task can be accessed using `memorize` and `recall` tools with the task \
memory key. The phase responsible for completion of most tasks is the EXECUTING \
phase, so all non-planning tasks should be assigned to the EXECUTING phase. In \
rare instances, specific tasks may be useful for the REVIEWING phase, in which \
case you may add tasks for that phase, but only do so if you foresee a very \
specific potential issue that must be subjected to a dedicated review task.
"""

INITIAL_PLANNING_PROMPT_TEMPLATE = f"""{PLANNING_PROMPT_PREFIX}
If creating a plan will be best done as a multi-step process, e.g. to create \
related tasks in batches, create tasks for the PLANNING phase.
Once the tasks are created and no PLANNING phase tasks remain incomplete, the
workflow will move to the EXECUTING phase, which .

<original_request>
$request
</original_request>"""

PLANNING_TASK_PROMPT_TEMPLATE = f"""{PLANNING_PROMPT_PREFIX}
You have been assigned a task to further the plan creation process.

<original_request>
$request
</original_request>

<task>
$task
</task>"""

PLANNING_FINALIZATION_PROMPT_TEMPALTE = f"""{PLANNING_PROMPT_PREFIX}
All PLANNING phase tasks have been completed. Finalize the plan so the EXECUTING \
phase can begin working. If no further changes are necessary to the existing \
plan, halt so the workflow can transition to the next phase.

<original_request>
$request
</original_request>"""


EXECUTING_PROMPT_TEMPLTAE = """You are a workflow execution agent. You have been \
assigned a specific task in furtherance of fulfilling a request. You must focus \
on completing your assigned task and nothing else.
Any details for the task can be accessed using `memorize` and `recall` tools with \
the task memory key.

IMPORTANT: when you complete the task, call the `complete_task` tool.

<original_request>
$request
</original_request>

<task>
$task
</task>"""


REVIEWING_TASK_PROMPT_TEMPLATE = """You are a workflow review agent. You must \
review the work done by the prior workflow execution agent toward a specific task.
If the task was completed successfully, use the `accept_task` tool.
If the task was not completed successfully and requires remediation, use the \
`reject_task` tool with actionable feedback.
Any details for the task can be accessed using the `recall` tool with the task \
memory key.

IMPORTANT: you must call exactly one of `accept_task` or `reject_task`.
Call the `accept_task` only if the task was completed successfully.
Call the `reject_task` only if the task was failed or is substantially incomplete.

<request>
$request
</request>

<task>
$task
</task>"""


PLANNING_TASK_REMEDIATION_PROMPT_TEMPLATE = """You are a workflow planning agent.
The prior executing agent's work was rejected by a review agent.
Your goal is to plan a course of remediation and insert that task in the correct
place in the planned tasks. Include relevant information as notes for the \
remediation task.

IMPORTANT: you must use the `view_tasks` tool to see the full plan, `add_task` \
to create the remediation task, and `move_task` to place it after the failed task.

<original_request>
$request
</original_request>

<task>
$task
</task>

<review_feedback>
$reject_reason
</review_feedback>"""


REVIEWING_FINAL_PROMPT_TEMPLATE = """You are a workflow review agent. You must \
review the work done toward fulfilling the original request. Examine the completed \
tasks and memories to ensure the original request was fulfilled. If the original \
request was fulfilled, call the `accept` tool. If it was not, call the `reject` \
tool with actionable feedback.

IMPORTANT: you must call exactly one of `accept` or `reject`.

<original_request>
$request
</original_request>"""


PLANNING_RECOVER_FINAL_REJECTION_TEMPLATE = f"""{PLANNING_PROMPT_PREFIX}
The final review agent rejected the work done to fulfill the original request.
You must consider the rejection feedback and create a remediation plan to fix \
any defficiencies and ensure the original request is properly fulfilled.

<original_request>
$request
</original_request>

<review_feedback>
$review
</review_feedback>"""


COMPACTION_PROMPT_TEMPLATE = """You are a context compaction agent.
Your task is to read and compress the messages that follow into a summary of \
what was done, what was learned, and the most relevant details.
Your output should be significantly shorter than the messages you summarize, \
and it should focus on high-level information useful outside the context of \
any particular task.

If there is a current task indicated below, include relevant details for it in \
your summary.

Mention relevant memory keys, files, and important details about previous tasks.

<original_request>
$request
</original_request>

<current_task>
$task
</current_task>"""


DONE_PROMPT_TEMPLATE = """You are the final summary agent for a workflow.
Your task is to produce a final message to convey the most important information \
for fulfilling the original request. Examine memories and files as necessary to \
collect the information you need to respond meaningfully to the original request.

<original_request>
$request
</original_request>"""


def interpolate_template(template: str, **values) -> str:
    """Interpolate the template prompt, replacing placeholders with
        appropriate values.
    """
    return Template(template).substitute(values)


def generate_initial_planning_prompt(**kwargs) -> str:
    """Generate initial prompt for PLANNING phase."""
    template = kwargs.get(
        'initial_planning_prompt_template', INITIAL_PLANNING_PROMPT_TEMPLATE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    addendum = kwargs.get('initial_planning_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_planning_task_prompt(**kwargs) -> str:
    """Generate task prompt for PLANNING phase."""
    template = kwargs.get(
        'planning_task_prompt_template', PLANNING_TASK_PROMPT_TEMPLATE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    assert "task" in kwargs, "cannot generate prompt without task"
    addendum = kwargs.get('planning_task_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_planning_finalization_prompt(**kwargs) -> str:
    """Generate initial prompt for PLANNING phase."""
    template = kwargs.get(
        'planning_finalization_prompt_template',
        PLANNING_FINALIZATION_PROMPT_TEMPALTE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    addendum = kwargs.get('planning_finalization_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_executing_prompt(**kwargs) -> str:
    """Generate prompt for EXECUTING phase."""
    tempalte = kwargs.get(
        'executing_prompt_template', EXECUTING_PROMPT_TEMPLTAE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    assert "task" in kwargs, "cannot generate prompt without task"
    addendum = kwargs.get('executing_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_reviewing_task_prompt(**kwargs) -> str:
    """Generate prompt for REVIEWING phase for a task."""
    tempalte = kwargs.get(
        'reviewing_task_prompt_template', REVIEWING_TASK_PROMPT_TEMPLATE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    assert "task" in kwargs, "cannot generate prompt without task"
    addendum = kwargs.get('_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_planning_task_remediation_prompt(**kwargs) -> str:
    """Generate prompt for PLANNING phase to remediate a rejected task."""
    tempalte = kwargs.get(
        'planning_task_remediation_prompt_template',
        PLANNING_TASK_REMEDIATION_PROMPT_TEMPLATE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    assert "task" in kwargs, "cannot generate prompt without task"
    assert "reject_reason" in kwargs, "cannot generate prompt without reject_reason"
    addendum = kwargs.get('_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_reviewing_final_prompt(**kwargs) -> str:
    """Generate prompt for phase."""
    tempalte = kwargs.get(
        'reviewing_final_prompt_template',
        REVIEWING_FINAL_PROMPT_TEMPLATE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    addendum = kwargs.get('_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_compaction_prompt(**kwargs) -> str:
    """Generate prompt for compaction."""
    tempalte = kwargs.get(
        'compaction_prompt_template',
        COMPACTION_PROMPT_TEMPLATE
    )
    assert "request" in kwargs, "cannot generate prompt without request"
    if "task" not in kwargs:
        kwargs["task"] = "N/A"
    addendum = kwargs.get('_prompt_addendum', None)
    if addendum:
        return interpolate_template(template, **kwargs) + f'\n\n{addendum}'
    return interpolate_template(template, **kwargs)


def generate_done_prompt(**kwargs) -> str:
    """Generate prompt for phase."""
    tempalte = kwargs.get('done_prompt_template', DONE_PROMPT_TEMPLATE)
    assert "request" in kwargs, "cannot generate prompt without request"
    return interpolate_template(template, **kwargs)
