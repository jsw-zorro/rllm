from pathlib import Path

def load_sys_msg() -> str:
    """Returns the system message content as a string."""
    system_msg_path = Path(__file__).parent / "system_prompt.md"
    
    if system_msg_path.exists():
        return system_msg_path.read_text(encoding='utf-8').strip()
    
    raise FileNotFoundError(f"System message file not found: {system_msg_path}")
