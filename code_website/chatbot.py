"""Expose the canonical chatbot implementation to the Flask entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code_chatbot.chatbot import MedicalChatBot
