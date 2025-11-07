# core/agent.py
from dotenv import load_dotenv
from google.genai import types
from core.utils import _setup_client, _format_tools, _convert_to_content, _format_prompt

load_dotenv()


class Agent:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        prompt: str = "",
        model: str = "",
        tools: list = [],
        temperature: float = 0.0,
        max_tokens: int = 3000,
    ) -> None:
        self.name = name
        self.description = description
        self.prompt = _format_prompt(prompt, name, description)
        self.model = model
        self.tools = _format_tools(tools)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = _setup_client()

    def execute(self, entry: str, memory: list = []) -> str:
        try:
            memory_contents = _convert_to_content(
                history_list=memory, max_tokens=self.max_tokens, client=self.client
            )
            new_user_content = types.Content(
                role="user", parts=[types.Part(text=entry)]
            )
            memory_contents.append(new_user_content)

            response = self.client.models.generate_content(
                model=self.model,
                contents=memory_contents,
                config={
                    "system_instruction": self.prompt,
                    "tools": self.tools,
                    "temperature": self.temperature,
                },
            )
            if not response.candidates:
                return "Error: The model did not return a valid candidate."
            candidate = response.candidates[0]
            final_text = ""
            for part in candidate.content.parts:
                if part.text:
                    final_text += part.text
            return final_text
        except Exception as e:
            print(f"Error executing agent: {e}")
            raise
