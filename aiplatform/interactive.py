"""
Interactive CLI Mode for AIPlatform

This module provides an interactive chat interface for the AIPlatform CLI,
allowing users to have conversational interactions with AI models.
"""

import sys
import os
import json
import readline
import atexit
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class InteractiveChat:
    """Interactive chat interface for AIPlatform."""
    
    def __init__(self, model: str = "gigachat3-702b", language: str = "en"):
        self.model = model
        self.language = language
        self.history: List[Dict[str, Any]] = []
        self.running = True
        self.context = {}
        
        # Setup readline for better input handling
        self._setup_readline()
        
        # Load history if exists
        self._load_history()
    
    def _setup_readline(self):
        """Setup readline for better input handling."""
        try:
            # Enable tab completion
            readline.parse_and_bind('tab: complete')
            
            # Setup history file
            history_file = os.path.expanduser('~/.aiplatform_history')
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass
            
            atexit.register(lambda: readline.write_history_file(history_file))
            
        except ImportError:
            # readline not available on Windows
            pass
    
    def _load_history(self):
        """Load chat history from file."""
        history_file = 'chat_history.json'
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"Loaded {len(self.history)} messages from history")
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
    
    def _save_history(self):
        """Save chat history to file."""
        history_file = 'chat_history.json'
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")
    
    def _get_prompt(self) -> str:
        """Get interactive prompt."""
        return f"[{self.model}]> "
    
    def _show_help(self):
        """Show help information."""
        help_text = """
Interactive AIPlatform Chat Commands:
  /help, /h, /?     - Show this help
  /history, /hist   - Show conversation history
  /clear, /c        - Clear conversation history
  /model <name>     - Switch AI model
  /save <file>      - Save conversation to file
  /load <file>      - Load conversation from file
  /context          - Show current context
  /stats            - Show session statistics
  /exit, /quit, /q  - Exit interactive mode
  
Available Models:
  - gigachat3-702b (default)
  - gpt-4 (if OpenAI configured)
  - claude-3-opus (if Claude configured)
  
Examples:
  /model gpt-4
  /save my_conversation.json
  What is quantum computing?
        """
        print(help_text)
    
    def _show_history(self):
        """Show conversation history."""
        if not self.history:
            print("No conversation history")
            return
        
        print("\n=== Conversation History ===")
        for i, msg in enumerate(self.history[-10:], 1):  # Show last 10 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            print(f"{i}. [{role.upper()}] {timestamp}")
            print(f"   {content[:100]}{'...' if len(content) > 100 else ''}")
            print()
    
    def _clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        print("Conversation history cleared")
    
    def _switch_model(self, model_name: str):
        """Switch AI model."""
        available_models = [
            "gigachat3-702b", "gpt-4", "gpt-3.5-turbo", 
            "claude-3-opus", "claude-3-sonnet"
        ]
        
        if model_name in available_models:
            self.model = model_name
            print(f"Switched to model: {model_name}")
        else:
            print(f"Unknown model: {model_name}")
            print(f"Available models: {', '.join(available_models)}")
    
    def _save_conversation(self, filename: str):
        """Save conversation to file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            print(f"Conversation saved to: {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def _load_conversation(self, filename: str):
        """Load conversation from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            print(f"Conversation loaded from: {filename}")
        except Exception as e:
            print(f"Error loading conversation: {e}")
    
    def _show_context(self):
        """Show current context."""
        print("\n=== Current Context ===")
        print(f"Model: {self.model}")
        print(f"Language: {self.language}")
        print(f"History length: {len(self.history)}")
        print(f"Context keys: {list(self.context.keys())}")
    
    def _show_stats(self):
        """Show session statistics."""
        user_messages = [m for m in self.history if m.get('role') == 'user']
        assistant_messages = [m for m in self.history if m.get('role') == 'assistant']
        
        total_chars = sum(len(m.get('content', '')) for m in self.history)
        
        print("\n=== Session Statistics ===")
        print(f"Total messages: {len(self.history)}")
        print(f"User messages: {len(user_messages)}")
        print(f"Assistant messages: {len(assistant_messages)}")
        print(f"Total characters: {total_chars}")
        print(f"Average message length: {total_chars / len(self.history) if self.history else 0:.1f}")
        
        if self.history:
            first_msg = self.history[0].get('timestamp', '')
            last_msg = self.history[-1].get('timestamp', '')
            print(f"Session start: {first_msg}")
            print(f"Last message: {last_msg}")
    
    def _process_command(self, user_input: str) -> bool:
        """Process special commands."""
        if not user_input.startswith('/'):
            return False
        
        parts = user_input[1:].split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in ['help', 'h', '?']:
            self._show_help()
        elif command in ['history', 'hist']:
            self._show_history()
        elif command in ['clear', 'c']:
            self._clear_history()
        elif command == 'model' and args:
            self._switch_model(args[0])
        elif command == 'save' and args:
            self._save_conversation(args[0])
        elif command == 'load' and args:
            self._load_conversation(args[0])
        elif command == 'context':
            self._show_context()
        elif command == 'stats':
            self._show_stats()
        elif command in ['exit', 'quit', 'q']:
            self.running = False
            print("Goodbye!")
        else:
            print(f"Unknown command: {user_input}")
            print("Type /help for available commands")
        
        return True
    
    def _generate_response(self, user_message: str) -> str:
        """Generate AI response (simulated)."""
        try:
            # Try to use genai module
            import importlib.util
            genai_path = os.path.join(os.path.dirname(__file__), 'genai.py')
            
            if os.path.exists(genai_path):
                spec = importlib.util.spec_from_file_location("genai", genai_path)
                genai = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(genai)
                
                # Create model and generate response
                model = genai.create_genai_model(self.model, language=self.language)
                response = model.generate_text(user_message, max_tokens=200)
                return response
            else:
                # Fallback to simulated response
                return f"[Simulated {self.model}] I understand you said: '{user_message}'. This is a simulated response since the actual AI model is not available."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _add_to_history(self, role: str, content: str):
        """Add message to history."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'model': self.model
        }
        self.history.append(message)
    
    def run(self):
        """Run interactive chat session."""
        print("=== AIPlatform Interactive Chat ===")
        print(f"Model: {self.model}")
        print("Type /help for commands or just start chatting!")
        print("Press Ctrl+C or type /exit to quit")
        print()
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input(self._get_prompt()).strip()
                    
                    if not user_input:
                        continue
                    
                    # Process commands
                    if self._process_command(user_input):
                        continue
                    
                    # Add user message to history
                    self._add_to_history('user', user_input)
                    
                    # Generate and display response
                    print("Thinking...")
                    response = self._generate_response(user_input)
                    
                    print(f"\n{response}\n")
                    
                    # Add assistant response to history
                    self._add_to_history('assistant', response)
                    
                except KeyboardInterrupt:
                    print("\nUse /exit to quit")
                    continue
                except EOFError:
                    print("\nGoodbye!")
                    break
                    
        finally:
            # Save history on exit
            self._save_history()


def start_interactive_mode(model: str = "gigachat3-702b", language: str = "en"):
    """Start interactive chat mode."""
    chat = InteractiveChat(model, language)
    chat.run()


if __name__ == "__main__":
    # Test interactive mode
    start_interactive_mode()
