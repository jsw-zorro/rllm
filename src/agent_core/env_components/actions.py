"""Action definitions for RLLM terminal bench environment."""

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Action(ABC):
    pass


@dataclass
class BashAction(Action):
    cmd: str
    block: bool = True
    timeout_secs: int = 30
    end_after_cmd_success: bool = False


@dataclass
class ContextCollapseAction(Action):
    summary: str
    msg_indexes_to_collapse: List[int]


# Base classes for Todo and Scratchpad
@dataclass
class TodoAction(Action):
    """Base class for all todo-related actions"""
    pass


@dataclass
class ScratchpadAction(Action):
    """Base class for all scratchpad-related actions"""
    pass


# Specific Todo actions
@dataclass
class AddTodoAction(TodoAction):
    content: str
    view_all: bool = False


@dataclass
class CompleteTodoAction(TodoAction):
    task_id: int
    view_all: bool = False


@dataclass
class DeleteTodoAction(TodoAction):
    task_id: int
    view_all: bool = False


@dataclass
class ViewAllTodosAction(TodoAction):
    pass


@dataclass
class TodoOperation:
    """Represents a single todo operation within a batch"""
    action: str
    content: str = None
    task_id: int = None


@dataclass
class BatchTodoAction(TodoAction):
    """Action for handling multiple todo operations in a single call"""
    operations: List[TodoOperation]
    view_all: bool = False


# Specific Scratchpad actions
@dataclass
class AddNoteAction(ScratchpadAction):
    content: str


@dataclass
class ViewAllNotesAction(ScratchpadAction):
    pass


# File operation actions
@dataclass
class FileAction(Action):
    """Base class for file operations"""
    pass


@dataclass
class ReadAction(FileAction):
    file_path: str
    offset: Optional[int] = None
    limit: Optional[int] = None


@dataclass
class WriteAction(FileAction):
    file_path: str
    content: str


@dataclass
class EditAction(FileAction):
    file_path: str
    old_string: str
    new_string: str
    replace_all: bool = False


@dataclass
class EditOperation:
    old_string: str
    new_string: str
    replace_all: bool = False


@dataclass
class MultiEditAction(FileAction):
    file_path: str
    edits: List[EditOperation]


# Search operation actions
@dataclass
class SearchAction(Action):
    """Base class for search operations"""
    pass


@dataclass
class GrepAction(SearchAction):
    """Search file contents using regex patterns"""
    pattern: str
    path: Optional[str] = None  # Directory to search in (defaults to current)
    include: Optional[str] = None  # File pattern filter (e.g., "*.py")


@dataclass
class GlobAction(SearchAction):
    """Find files by name pattern"""
    pattern: str
    path: Optional[str] = None  # Directory to search in (defaults to current)


@dataclass
class LSAction(SearchAction):
    """List directory contents"""
    path: str  # Absolute directory path
    ignore: Optional[List[str]] = None  # Glob patterns to exclude


@dataclass
class FileMetadataAction(FileAction):
    """Get metadata for multiple files (max 10)"""
    file_paths: List[str]  # List of absolute file paths


@dataclass
class FinishAction(Action):
    """Signal that the task has been completed"""
    message: str  # Brief summary of what was accomplished
