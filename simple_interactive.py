#!/usr/bin/env python3
"""
Simple Interactive Mode Test for AIPlatform

This is a simplified version that works on Windows without readline.
"""

import json
import os
from typing import List, Dict, Any

class SimpleInteractiveChat:
    def __init__(self, model: str = "gigachat3-702b"):
        self.model = model
        self.history = []
        self.running = True
    
    def show_help(self):
        print("""
Simple Interactive Chat Commands:
  /help - Show this help
  /history - Show conversation history  
  /clear - Clear history
  /exit - Exit chat
  
Just type your message and press Enter!
        """)
    
    def show_history(self):
        if not self.history:
            print("No conversation history")
            return
        
        print("\n=== Conversation History ===")
        for i, msg in enumerate(self.history[-5:], 1):  # Show last 5
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            print(f"{i}. [{role.upper()}] {content[:50]}...")
    
    def clear_history(self):
        self.history.clear()
        print("History cleared")
    
    def generate_response(self, user_message: str) -> str:
        """Generate simulated response."""
        return f"[{self.model}] I understand you said: '{user_message}'. This is a simulated response."
    
    def add_to_history(self, role: str, content: str):
        self.history.append({
            'role': role,
            'content': content,
            'model': self.model
        })
    
    def run(self):
        print("=== Simple AIPlatform Chat ===")
        print(f"Model: {self.model}")
        print("Type /help for commands")
        print("Type /exit to quit")
        print()
        
        while self.running:
            try:
                user_input = input(f"[{self.model}]> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'help':
                        self.show_help()
                    elif command == 'history':
                        self.show_history()
                    elif command == 'clear':
                        self.clear_history()
                    elif command in ['exit', 'quit']:
                        print("Goodbye!")
                        self.running = False
                    else:
                        print("Unknown command. Type /help")
                    continue
                
                # Add user message
                self.add_to_history('user', user_input)
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"\n{response}\n")
                
                # Add assistant response
                self.add_to_history('assistant', response)
                
            except KeyboardInterrupt:
                print("\nUse /exit to quit")
            except EOFError:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    chat = SimpleInteractiveChat()
    chat.run()
