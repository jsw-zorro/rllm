"""Command execution abstraction for both Docker and Tmux environments."""

import asyncio
from abc import ABC, abstractmethod
from typing import Tuple


class CommandExecutor(ABC):
    """Abstract base class for command execution in different environments."""
    
    @abstractmethod
    async def execute(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Execute a command and return (output, return_code)."""
    
    @abstractmethod
    async def execute_background(self, cmd: str) -> None:
        """Execute a command in background."""


class DockerExecutor(CommandExecutor):
    """Execute commands using docker exec."""
    
    def __init__(self, container_name: str):
        self.container_name = container_name
    
    async def execute(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Execute a command in the Docker container and return (output, return_code)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'exec', self.container_name, 'bash', '-c', cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), 
                timeout=timeout
            )
            output = stdout.decode('utf-8', errors='replace')
            exit_code = proc.returncode or 0
            
            return output, exit_code
            
        except asyncio.TimeoutError:
            if proc:
                proc.kill()
            return f"Command timed out after {timeout} seconds", 124  # 124 is the standard timeout exit code
        except Exception as e:
            return f"Error executing command: {str(e)}", 1
    
    async def execute_background(self, cmd: str) -> None:
        """Execute a command in background in the Docker container."""
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'exec', '-d', self.container_name, 'bash', '-c', cmd
            )
            await proc.wait()
        except Exception:
            # Background execution failures are silently ignored
            pass


class TmuxExecutor(CommandExecutor):
    """Execute commands using tmux session."""
    
    def __init__(self, session):
        """Initialize with a TmuxSession object."""
        self.session = session
    
    async def execute(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Execute a command in the tmux session and return (output, return_code)."""
        try:
            self.session.send_keys([cmd, "Enter"], block=True, max_timeout_sec=timeout)
            
            output = self.session.capture_pane()
            
            # We can't reliably determine exit code in tmux without modifying the command
            # Return 0 for now - the agent logic will need to handle errors differently
            return output, 0
            
        except TimeoutError:
            return f"Command timed out after {timeout} seconds", 124
        except Exception as e:
            return f"Error executing command: {str(e)}", 1
    
    async def execute_background(self, cmd: str) -> None:
        """Execute a command in background in the tmux session."""
        try:
            # For background commands, add '&' if not present
            if not cmd.strip().endswith('&'):
                cmd = f"{cmd} &"
            
            self.session.send_keys([cmd, "Enter"], block=False, min_timeout_sec=0.1)
        except Exception:
            pass