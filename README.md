# gemini_abstraction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Google GenAI](https://img.shields.io/badge/Google-GenAI-yellow)](https://cloud.google.com/genai)
[![dotenv](https://img.shields.io/badge/dotenv-5.0%2B-green)](https://pypi.org/project/python-dotenv/)

## Overview

`gemini_abstraction` is a Python-based abstraction layer for interacting with Google's GenAI models. It simplifies the process of setting up clients, managing token limits, and integrating tools for enhanced conversational AI experiences.

## Features

- **Client Setup**: Automatically configures the GenAI client for seamless integration.
- **Token Management**: Handles token limits and provides summarization for long-term memory.
- **Tool Integration**: Supports custom tools for extending agent capabilities.
- **Customizable Agents**: Easily configure prompts, models, and other parameters.

## Code Highlights

### Utilities

The `core/utils.py` module provides essential functions:
- `_setup_client`: Initializes the GenAI client.
- `_convert_to_content`: Converts conversation history into GenAI-compatible content, with summarization for token management.
- `_format_tools`: Formats tools for agent integration.

### Agent

The `core/agent.py` module defines the `Agent` class:
- Configurable with prompts, models, tools, and other parameters.
- Executes user inputs and maintains conversational memory.

### Example Tool

A sample tool, `obter_data_e_hora_atual`, provides the current date and time for specific time zones.

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Set up your environment variables using a `.env` file.

3. Run the main script:
    ```bash
    python main.py
    ```

4. Interact with the agent to get responses based on your inputs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Google GenAI](https://cloud.google.com/genai) for the powerful AI models.
- [Python dotenv](https://pypi.org/project/python-dotenv/) for environment variable management.

