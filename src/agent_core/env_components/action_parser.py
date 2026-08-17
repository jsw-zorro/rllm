"""Action parser for RLLM terminal bench environment."""

import logging
import re
from typing import List, Tuple, Any

import yaml

from src.agent_core.env_components.action_factory import ActionFactory
from src.agent_core.env_components.actions import Action, BashAction, ContextCollapseAction, FinishAction



class ActionParserXMLYaml:
    """Handles parsing of agent responses into structured actions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.action_factory = ActionFactory()
    
    def get_action_list_from_agent_response(self, response: str) -> Tuple[List[Action], List[str], bool]:
        """Parse agent response text into structured actions.
        
        Returns:
            Tuple of (actions, errors, found_action_attempt) where:
            - actions: List of parsed Action objects
            - errors: List of parsing error messages
            - found_action_attempt: True if any action XML tags (non-think) were found in the response
        """
        xml_pattern = r'<(\w+)>([\s\S]*?)</\1>'
        matches = re.findall(xml_pattern, response)
        
        actions: List[Action] = []
        errors: List[str] = []
        
        # Tags that should be ignored (not parsed as actions)
        ignored_tags = {
            'think'  # Default tag for thinking for many models, including Qwen
        }
        
        found_action_attempt = any(tag_name.lower() not in ignored_tags for tag_name, _ in matches) 
        
        for tag_name, content in matches:
            if tag_name.lower() in ignored_tags:
                self.logger.debug(f"Skipping {tag_name} tag (not an action)")
                continue
            try:
                yaml_data = yaml.safe_load(content.strip())
                
                # Check if yaml_data is a dictionary
                if not isinstance(yaml_data, dict):
                    error_msg = self._format_parsing_error(
                        tag_name, content, yaml_data, 
                        "Invalid action format. Expected YAML."
                    )
                    errors.append(error_msg)
                    self.logger.warning(f"Invalid YAML format in {tag_name}: got {type(yaml_data).__name__} instead of dict")
                    continue
                
                if tag_name == 'bash':
                    actions.append(BashAction(
                        cmd=yaml_data.get('cmd', ''),
                        block=yaml_data.get('block', True),
                        timeout_secs=yaml_data.get('timeout_secs', 30),
                        end_after_cmd_success=yaml_data.get('end_after_cmd_success', False)
                    ))
                elif tag_name == 'collapse_context':
                    actions.append(ContextCollapseAction(
                        summary=yaml_data.get('summary', ''),
                        msg_indexes_to_collapse=yaml_data.get('msg_indexes_to_collapse', [])
                    ))
                elif tag_name == 'todo':
                    try:
                        action = self.action_factory.create_todo_action(yaml_data)
                        if action:
                            actions.append(action)
                        else:
                            error_msg = self._format_parsing_error(
                                tag_name, content, yaml_data,
                                f"Unknown todo action: {yaml_data.get('action', 'none')}"
                            )
                            errors.append(error_msg)
                    except ValueError as e:
                        error_msg = self._format_parsing_error(
                            tag_name, content, yaml_data,
                            str(e)
                        )
                        errors.append(error_msg)
                elif tag_name == 'scratchpad':
                    action = self.action_factory.create_scratchpad_action(yaml_data)
                    if action:
                        actions.append(action)
                    else:
                        error_msg = self._format_parsing_error(
                            tag_name, content, yaml_data,
                            f"Unknown scratchpad action: {yaml_data.get('action', 'none')}"
                        )
                        errors.append(error_msg)
                elif tag_name == 'file':
                    try:
                        action = self.action_factory.create_file_action(yaml_data)
                        if action:
                            actions.append(action)
                        else:
                            error_msg = self._format_parsing_error(
                                tag_name, content, yaml_data,
                                f"Unknown file action: {yaml_data.get('action', 'none')}"
                            )
                            errors.append(error_msg)
                    except ValueError as e:
                        error_msg = self._format_parsing_error(
                            tag_name, content, yaml_data,
                            str(e)
                        )
                        errors.append(error_msg)
                elif tag_name == 'search':
                    try:
                        action = self.action_factory.create_search_action(yaml_data)
                        if action:
                            actions.append(action)
                        else:
                            error_msg = self._format_parsing_error(
                                tag_name, content, yaml_data,
                                f"Unknown search action: {yaml_data.get('action', 'none')}"
                            )
                            errors.append(error_msg)
                    except ValueError as e:
                        error_msg = self._format_parsing_error(
                            tag_name, content, yaml_data,
                            str(e)
                        )
                        errors.append(error_msg)
                elif tag_name == 'finish':
                    message = yaml_data.get('message', 'Task completed')
                    actions.append(FinishAction(message=message))
                    
            except yaml.YAMLError as e:
                # Try to extract line number from YAML error
                error_desc = str(e)
                problem_mark = getattr(e, 'problem_mark', None)
                if problem_mark:
                    line_num = problem_mark.line + 1
                    col_num = problem_mark.column + 1
                    problem = getattr(e, 'problem', None) or str(e)
                    error_desc = f"YAML syntax error at line {line_num}, column {col_num}: {problem}"
                else:
                    error_desc = f"YAML parsing error: {e}"
                
                error_msg = self._format_parsing_error(
                    tag_name, content, None, error_desc
                )
                errors.append(error_msg)
                self.logger.debug(f"Failed to parse YAML in {tag_name} tag: {e}")
                continue
        
        return actions, errors, found_action_attempt
    
    def _format_parsing_error(self, tag_name: str, content: str, parsed_value: Any, error_desc: str) -> str:
        """Format a parsing error message with helpful context."""
        # Truncate content if too long
        display_content = content if len(content) <= 100 else content[:100] + "..."
        
        # Enhanced error messages for common mistakes
        if tag_name == 'todo' and parsed_value:
            # Check for common single-action mistake
            if 'action' in parsed_value and 'operations' not in parsed_value:
                error_desc += "\n\nError: Todo actions must use the batch format with 'operations' list."
            # Check if they tried to use a string instead of list for operations
            elif 'operations' in parsed_value and not isinstance(parsed_value.get('operations'), list):
                error_desc += f"\n\nError: 'operations' must be a list, but got {type(parsed_value.get('operations')).__name__}"
        
        # Add more specific type information
        if parsed_value is not None and "Expected YAML" in error_desc:
            error_desc += f"\nReceived type: {type(parsed_value).__name__}"
            if isinstance(parsed_value, (str, int, float, bool)):
                error_desc += f" with value: {repr(parsed_value)}"
        
        # Get the expected format for each tag type
        example_formats = {
            'bash': '''<bash>
cmd: "your command here"
block: true  # optional, defaults to true
timeout_secs: 30  # optional, defaults to 30
end_after_cmd_success: false  # optional, defaults to false
</bash>''',
            'collapse_context': '''<collapse_context>
msg_indexes_to_collapse: [1, 2, 3, 4]  # sequential message IDs to remove
summary: |
  Summary of collapsed messages...
</collapse_context>''',
            'todo': '''<todo>
operations:
  - action: add
    content: "Task description"
  - action: complete
    task_id: 1
  - action: delete
    task_id: 2
  - action: view_all
view_all: true  # optional: show todo list after all operations
</todo>

# Common mistakes:
# - Missing 'operations' list
# - Invalid action types''',
            'scratchpad': '''<scratchpad>
action: add_note  # or view_all_notes
content: "Note content"  # required for add_note action
</scratchpad>''',
            'file': '''<file>
action: read  # or write, edit, multi_edit, metadata
file_path: "/path/to/file.txt"
# For read action:
offset: 100  # optional: start from line 100
limit: 50    # optional: read only 50 lines

# For write action:
content: "File content goes here\nwith new lines in the string"
  
# For edit action:
old_string: "text to replace"
new_string: "replacement text"
replace_all: false  # optional: replace all occurrences

# For multi_edit action:
edits:
  - old_string: "first text"
    new_string: "first replacement"
  - old_string: "second text"
    new_string: "second replacement"
    replace_all: true

# For metadata action:
file_paths:  # required: list of file paths (max 10)
  - "/path/to/file1.txt"
  - "/path/to/file2.py"
</file>''',
            'search': '''<search>
action: grep  # or glob, ls
# For grep action:
pattern: "regex pattern"  # required
path: "/directory/to/search"  # optional, defaults to current
include: "*.py"  # optional file pattern filter

# For glob action:
pattern: "*.js"  # required glob pattern
path: "/directory/to/search"  # optional, defaults to current

# For ls action:
path: "/absolute/path"  # required absolute path
ignore:  # optional list of patterns to exclude
  - "*.pyc"
  - "__pycache__"
</search>''',
            'finish': '''<finish>
message: "Brief summary of what was accomplished"
</finish>'''
        }
        
        error_content = f"""[ERROR] Failed to parse <{tag_name}> action: {error_desc}
Content: "{display_content}"
"""
        
        if tag_name in example_formats:
            error_content += f"Expected format:\n{example_formats[tag_name]}"
        
        # Format in XML
        error_msg = f"<parse_error>\n{error_content}\n</parse_error>"
        
        return error_msg