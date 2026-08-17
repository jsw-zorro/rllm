"""Shared step execution logic for both RL training and evaluation agents."""

import logging
from typing import List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.agent_core.env_components.actions import Action, BashAction, FinishAction
from src.agent_core.env_components.action_parser import ActionParserXMLYaml
from src.agent_core.env_components.action_handler import ActionHandler

logger = logging.getLogger(__name__)


class StepResult(Enum):
    """Result of executing a step."""
    CONTINUE = "continue"
    FINISH = "finish"
    ERROR = "error"
    NO_ACTION = "no_action"
    PARSE_ERROR_LIMIT = "parse_error_limit"


@dataclass
class StepExecutionResult:
    """Result of executing a single step."""
    result: StepResult
    responses: List[str]
    has_error: bool
    finish_message: Optional[str] = None
    actions_executed: int = 0


class StepExecutor:
    """Executes a single step of agent interaction with callbacks for customization."""
    
    def __init__(
        self,
        action_parser: ActionParserXMLYaml,
        action_handler: ActionHandler,
        max_consecutive_parse_errors: int = 3,
        on_no_action: Optional[Callable[[], None]] = None,
        on_parse_error_limit: Optional[Callable[[], None]] = None,
        on_finish_action: Optional[Callable[[FinishAction], None]] = None,
        on_bash_success: Optional[Callable[[BashAction], None]] = None,
        on_action_executed: Optional[Callable[[Action, str, bool], None]] = None,
    ):
        """Initialize the step executor.
        
        Args:
            action_parser: Parser for extracting actions from agent responses
            action_handler: Handler for executing actions
            max_consecutive_parse_errors: Maximum consecutive parsing errors before stopping
            on_no_action: Callback when no action is attempted
            on_parse_error_limit: Callback when parse error limit is reached
            on_finish_action: Callback when finish action is executed
            on_bash_success: Callback when bash action succeeds with end_after_cmd_success
            on_action_executed: Callback after each action execution (action, output, is_error)
        """
        self.action_parser = action_parser
        self.action_handler = action_handler
        self.max_consecutive_parse_errors = max_consecutive_parse_errors
        self.consecutive_parse_errors = 0
        
        # Callbacks
        self.on_no_action = on_no_action
        self.on_parse_error_limit = on_parse_error_limit
        self.on_finish_action = on_finish_action
        self.on_bash_success = on_bash_success
        self.on_action_executed = on_action_executed
    
    async def execute_step(self, agent_response: str) -> StepExecutionResult:
        # Parse actions from response
        actions, parsing_errors, found_action_attempt = self.action_parser.get_action_list_from_agent_response(
            agent_response
        )
        
        # Check if no action attempt was made
        if not found_action_attempt:
            if self.on_no_action:
                self.on_no_action()
            logger.info("No actions attempted in response")
            return StepExecutionResult(
                result=StepResult.NO_ACTION,
                responses=["No actions were attempted."],
                has_error=True
            )
        
        # Reset counter if we got valid actions
        if actions:
            logger.debug(f"Parsed {len(actions)} actions")
            self.consecutive_parse_errors = 0
        
        responses = []
        has_error = False
        
        # Handle parsing errors
        if parsing_errors:
            responses.extend(parsing_errors)
            has_error = True
            
            # Only count as consecutive if NO valid actions were parsed
            if not actions:
                self.consecutive_parse_errors += 1
                logger.warning(f"Consecutive parsing errors: {self.consecutive_parse_errors}")
                
                if self.consecutive_parse_errors >= self.max_consecutive_parse_errors:
                    if self.on_parse_error_limit:
                        self.on_parse_error_limit()
                    responses.append(f"\n<error>Trajectory ended due to {self.max_consecutive_parse_errors} consecutive parsing errors.</error>")
                    return StepExecutionResult(
                        result=StepResult.PARSE_ERROR_LIMIT,
                        responses=responses,
                        has_error=True
                    )
        
        # Execute each action
        finish_message = None
        step_result = StepResult.CONTINUE
        actions_executed = 0
        
        for action in actions:
            try:
                # Execute the action
                output, is_error = await self.action_handler.handle_action(action)
                responses.append(output)
                actions_executed += 1
                
                if is_error:
                    has_error = True
                
                # Callback after execution
                if self.on_action_executed:
                    self.on_action_executed(action, output, is_error)
                
                # Check for completion conditions
                if isinstance(action, BashAction):
                    if action.end_after_cmd_success and not is_error:
                        if self.on_bash_success:
                            self.on_bash_success(action)
                        logger.info("Bash action succeeded with end_after_cmd_success")
                        step_result = StepResult.FINISH
                        finish_message = "Success command executed"
                        break
                elif isinstance(action, FinishAction):
                    if self.on_finish_action:
                        self.on_finish_action(action)
                    logger.info(f"Finish action called: {action.message}")
                    step_result = StepResult.FINISH
                    finish_message = action.message
                    break
                    
            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                responses.append(f"<error>Action execution failed: {str(e)}</error>")
                has_error = True
                step_result = StepResult.ERROR
        
        return StepExecutionResult(
            result=step_result,
            responses=responses,
            has_error=has_error,
            finish_message=finish_message,
            actions_executed=actions_executed
        )
    
    def reset_error_counters(self):
        """Reset error counters for a new trajectory."""
        self.consecutive_parse_errors = 0