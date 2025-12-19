import json
from typing import List, Dict, Any
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import content_types
from collections.abc import Iterable

from config import config
from tools.search_tool import SearchTool

class CodeAgent:
    def __init__(self):
        self.search_tool = SearchTool()
        self.provider = config.chat_provider
        
        config.validate_chat_config()
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported chat provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model_name = config.chat_model
        self.tools = [self.search_tool.get_tool_definition()]
        self.messages = [
            {"role": "system", "content": "You are an expert AI software engineer. You have access to a codebase and can search it to answer questions. Always verify your assumptions by searching the code. When answering, reference specific files and lines if possible."}
        ]
    
    def _init_gemini(self):
        """Initialize Gemini client."""
        genai.configure(api_key=config.gemini_api_key)
        
        # Map the function directly for Gemini
        self.tools = [self.search_tool.search_codebase]
        
        system_instruction = "You are an expert AI software engineer. You have access to a codebase and can search it to answer questions. Always verify your assumptions by searching the code. When answering, reference specific files and lines if possible."
        
        self.model = genai.GenerativeModel(
            model_name=config.chat_model,
            tools=self.tools,
            system_instruction=system_instruction
        )
        self.chat_session = self.model.start_chat(enable_automatic_function_calling=True)
    
    def _init_ollama(self):
        """Initialize Ollama client using OpenAI-compatible API."""
        from openai import OpenAI as OllamaClient
        
        self.client = OllamaClient(
            base_url=f"{config.ollama_base_url}/v1",
            api_key="ollama"  # Ollama doesn't require real API key
        )
        self.model_name = config.chat_model
        self.tools = [self.search_tool.get_tool_definition()]
        self.messages = [
            {"role": "system", "content": "You are an expert AI software engineer. You have access to a codebase and can search it to answer questions. Always verify your assumptions by searching the code. When answering, reference specific files and lines if possible."}
        ]

    def chat(self, user_input: str) -> str:
        if self.provider == "openai":
            return self._chat_openai(user_input)
        elif self.provider == "gemini":
            return self._chat_gemini(user_input)
        elif self.provider == "ollama":
            return self._chat_ollama(user_input)

    def _chat_openai(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            self.messages.append(response_message)

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "search_codebase":
                        tool_output = self.search_tool.search_codebase(**function_args)
                        
                        self.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_output,
                        })
            else:
                return response_message.content

    def _chat_gemini(self, user_input: str) -> str:
        try:
            response = self.chat_session.send_message(user_input)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"
    
    def _chat_ollama(self, user_input: str) -> str:
        """Chat using Ollama (same as OpenAI since Ollama uses OpenAI-compatible API)."""
        return self._chat_openai(user_input)

    def reset(self):
        if self.provider in ["openai", "ollama"]:
            self.messages = [
                {"role": "system", "content": "You are an expert AI software engineer. You have access to a codebase and can search it to answer questions. Always verify your assumptions by searching the code. When answering, reference specific files and lines if possible."}
            ]
        elif self.provider == "gemini":
            self.chat_session = self.model.start_chat(enable_automatic_function_calling=True)

