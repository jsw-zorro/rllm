"""Action factory for creating actions from parsed data."""

from typing import Dict, Optional

from src.agent_core.env_components.actions import (
    TodoAction, BatchTodoAction, TodoOperation,
    ScratchpadAction, AddNoteAction, ViewAllNotesAction,
    FileAction, ReadAction, WriteAction, EditAction,
    MultiEditAction, EditOperation, FileMetadataAction,
    SearchAction, GrepAction, GlobAction, LSAction
)



class ActionFactory:
    """Factory for creating actions from parsed data"""
    
    def __init__(self):
        self._scratchpad_factories = {
            'add_note': self._create_add_note,
            'view_all_notes': self._create_view_all_notes
        }
        self._file_factories = {
            'read': self._create_read_action,
            'write': self._create_write_action,
            'edit': self._create_edit_action,
            'multi_edit': self._create_multi_edit_action,
            'metadata': self._create_metadata_action
        }
        self._search_factories = {
            'grep': self._create_grep_action,
            'glob': self._create_glob_action,
            'ls': self._create_ls_action
        }
    
    def create_todo_action(self, yaml_data: Dict) -> Optional[TodoAction]:
        """Create a todo action from parsed YAML data"""
        # Only support batch operations
        try:
            return self._create_batch_todo(yaml_data)
        except ValueError:
            # Re-raise to let the parser handle the error formatting
            raise
    
    def create_scratchpad_action(self, yaml_data: Dict) -> Optional[ScratchpadAction]:
        """Create a scratchpad action from parsed YAML data"""
        action_type = yaml_data.get('action')
        factory = self._scratchpad_factories.get(action_type)
        if factory:
            return factory(yaml_data)
        return None
    
    def create_file_action(self, yaml_data: Dict) -> Optional[FileAction]:
        """Create a file action from parsed YAML data"""
        action_type = yaml_data.get('action')
        if not action_type:
            raise ValueError("File action requires 'action' field")
        
        factory = self._file_factories.get(action_type)
        if factory:
            return factory(yaml_data)
        return None
    
    def create_search_action(self, yaml_data: Dict) -> Optional[SearchAction]:
        """Create a search action from parsed YAML data"""
        action_type = yaml_data.get('action')
        if not action_type:
            raise ValueError("Search action requires 'action' field")
        
        factory = self._search_factories.get(action_type)
        if factory:
            return factory(yaml_data)
        return None
    
    def _create_batch_todo(self, data: Dict) -> BatchTodoAction:
        """Create a batch todo action from parsed YAML data with comprehensive validation"""
        operations_data = data.get('operations', [])
        
        # Validate operations exists and is non-empty
        if not operations_data:
            raise ValueError("Todo batch must contain at least one operation")
        
        if not isinstance(operations_data, list):
            raise ValueError(f"Operations must be a list, got {type(operations_data).__name__}")
        
        # Track task_ids to prevent duplicates within same batch
        complete_ids = set()
        delete_ids = set()
        valid_actions = {'add', 'complete', 'delete', 'view_all'}
        
        operations = []
        for i, op_data in enumerate(operations_data):
            if not isinstance(op_data, dict):
                raise ValueError(f"Operation {i+1} must be a dictionary, got {type(op_data).__name__}")
            
            action = op_data.get('action', '').strip()
            if not action:
                raise ValueError(f"Operation {i+1} missing required 'action' field")
            
            if action not in valid_actions:
                raise ValueError(
                    f"Operation {i+1} has invalid action '{action}'. "
                    f"Valid actions: {', '.join(sorted(valid_actions))}"
                )
            
            # Action-specific validation
            if action == 'add':
                content = op_data.get('content', '')
                if not isinstance(content, str):
                    raise ValueError(f"Operation {i+1}: 'content' must be a string")
                if not content.strip():
                    raise ValueError(f"Operation {i+1}: 'add' action requires non-empty 'content'")
                
            elif action in ('complete', 'delete'):
                task_id = op_data.get('task_id')
                if task_id is None:
                    raise ValueError(f"Operation {i+1}: '{action}' action requires 'task_id'")
                if not isinstance(task_id, int) or task_id < 1:
                    raise ValueError(f"Operation {i+1}: task_id must be a positive integer, got {task_id}")
                
                # Check for duplicates within this batch
                if action == 'complete':
                    if task_id in complete_ids:
                        raise ValueError(f"Operation {i+1}: duplicate complete for task_id {task_id}")
                    complete_ids.add(task_id)
                else:  # delete
                    if task_id in delete_ids:
                        raise ValueError(f"Operation {i+1}: duplicate delete for task_id {task_id}")
                    delete_ids.add(task_id)
            
            # Create the operation
            operation = TodoOperation(
                action=action,
                content=op_data.get('content'),
                task_id=op_data.get('task_id')
            )
            operations.append(operation)
        
        return BatchTodoAction(
            operations=operations,
            view_all=data.get('view_all', False)
        )
    
    # Individual factory methods for Scratchpad actions
    def _create_add_note(self, data: Dict) -> AddNoteAction:
        return AddNoteAction(content=data.get('content', ''))
    
    def _create_view_all_notes(self, data: Dict) -> ViewAllNotesAction:
        return ViewAllNotesAction()
    
    # Individual factory methods for File actions
    def _create_read_action(self, data: Dict) -> ReadAction:
        file_path = data.get('file_path')
        if not file_path:
            raise ValueError("Read action requires 'file_path'")
        
        return ReadAction(
            file_path=file_path,
            offset=data.get('offset'),
            limit=data.get('limit')
        )
    
    def _create_write_action(self, data: Dict) -> WriteAction:
        file_path = data.get('file_path')
        if not file_path:
            raise ValueError("Write action requires 'file_path'")
        
        content = data.get('content', '')
        if not isinstance(content, str):
            raise ValueError("Write action 'content' must be a string")
        
        return WriteAction(
            file_path=file_path,
            content=content
        )
    
    def _create_edit_action(self, data: Dict) -> EditAction:
        file_path = data.get('file_path')
        if not file_path:
            raise ValueError("Edit action requires 'file_path'")
        
        old_string = data.get('old_string')
        if old_string is None:
            raise ValueError("Edit action requires 'old_string'")
        
        new_string = data.get('new_string')
        if new_string is None:
            raise ValueError("Edit action requires 'new_string'")
        
        return EditAction(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=data.get('replace_all', False)
        )
    
    def _create_multi_edit_action(self, data: Dict) -> MultiEditAction:
        file_path = data.get('file_path')
        if not file_path:
            raise ValueError("Multi-edit action requires 'file_path'")
        
        edits_data = data.get('edits', [])
        if not edits_data:
            raise ValueError("Multi-edit action requires at least one edit in 'edits' list")
        
        if not isinstance(edits_data, list):
            raise ValueError(f"'edits' must be a list, got {type(edits_data).__name__}")
        
        edits = []
        for i, edit_data in enumerate(edits_data):
            if not isinstance(edit_data, dict):
                raise ValueError(f"Edit {i+1} must be a dictionary, got {type(edit_data).__name__}")
            
            old_string = edit_data.get('old_string')
            if old_string is None:
                raise ValueError(f"Edit {i+1} requires 'old_string'")
            
            new_string = edit_data.get('new_string')
            if new_string is None:
                raise ValueError(f"Edit {i+1} requires 'new_string'")
            
            edits.append(EditOperation(
                old_string=old_string,
                new_string=new_string,
                replace_all=edit_data.get('replace_all', False)
            ))
        
        return MultiEditAction(
            file_path=file_path,
            edits=edits
        )
    
    # Individual factory methods for Search actions
    def _create_grep_action(self, data: Dict) -> GrepAction:
        pattern = data.get('pattern')
        if not pattern:
            raise ValueError("Grep action requires 'pattern'")
        
        return GrepAction(
            pattern=pattern,
            path=data.get('path'),
            include=data.get('include')
        )
    
    def _create_glob_action(self, data: Dict) -> GlobAction:
        pattern = data.get('pattern')
        if not pattern:
            raise ValueError("Glob action requires 'pattern'")
        
        return GlobAction(
            pattern=pattern,
            path=data.get('path')
        )
    
    def _create_ls_action(self, data: Dict) -> LSAction:
        path = data.get('path')
        if not path:
            raise ValueError("LS action requires 'path'")
        
        ignore = data.get('ignore', [])
        if ignore and not isinstance(ignore, list):
            raise ValueError(f"'ignore' must be a list, got {type(ignore).__name__}")
        
        return LSAction(
            path=path,
            ignore=ignore
        )
    
    def _create_metadata_action(self, data: Dict) -> FileMetadataAction:
        file_paths = data.get('file_paths', [])
        if not file_paths:
            raise ValueError("Metadata action requires 'file_paths' list")
        
        if not isinstance(file_paths, list):
            raise ValueError(f"'file_paths' must be a list, got {type(file_paths).__name__}")
        
        if len(file_paths) > 10:
            raise ValueError(f"Metadata action supports maximum 10 files, got {len(file_paths)}")
        
        for i, path in enumerate(file_paths):
            if not isinstance(path, str):
                raise ValueError(f"File path {i+1} must be a string, got {type(path).__name__}")
        
        return FileMetadataAction(file_paths=file_paths)