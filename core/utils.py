# core/utils.py
from typing import List, Dict, Tuple
from google import genai
from google.genai import types


def _setup_client() -> genai.Client:
    """
    Sets up and returns the GenAI client.
    """
    try:
        client = genai.Client()
        return client
    except Exception as e:
        print(f"Error setting up GenAI client: {e}")
        raise


def _convert_to_content(
    history_list: List[Dict[str, str]], max_tokens: int, client: genai.Client
) -> List[types.Content]:
    """
    Converts the entire history into Contents.
    If it exceeds the token limit, summarizes EVERYTHING at once.
    """

    all_contents: List[types.Content] = []

    for message in history_list:
        role = "user" if message["role"] == "user" else "model"
        text_part = types.Part(text=message["content"])
        content_item = types.Content(role=role, parts=[text_part])
        all_contents.append(content_item)

    try:
        token_count_response = client.models.count_tokens(
            model="gemini-2.5-flash", contents=all_contents
        )
        total_tokens = token_count_response.total_tokens
        print(f"📊 Current tokens: {total_tokens}/{max_tokens}")
    except Exception as e:
        print(f"Error counting tokens: {e}. Returning contents as-is.")
        return all_contents

    SUMMARIZATION_BUFFER = max(50, int(max_tokens * 0.2))
    if total_tokens <= max_tokens - SUMMARIZATION_BUFFER:
        return all_contents

    print(
        "🚨 Max tokens exceeded! Starting history summarization for long-term memory..."
    )

    previous_summary = None
    messages_to_summarize = all_contents

    if all_contents and all_contents[0].role == "user":
        first_text = all_contents[0].parts[0].text
        if "[MEMORY CONTEXT FROM PREVIOUS CONVERSATION]" in first_text:
            try:
                previous_summary = (
                    first_text.split("[MEMORY CONTEXT FROM PREVIOUS CONVERSATION]")[1]
                    .split("[END OF MEMORY CONTEXT")[0]
                    .strip()
                )
                messages_to_summarize = all_contents[1:]
                print(f"📚 Previous summary found. Will accumulate new information...")
            except:
                print("⚠️ Error parsing previous summary. Starting fresh.")
                previous_summary = None

    history_text = "\n".join(
        [
            f"{c.role}: {c.parts[0].text}"
            for c in messages_to_summarize
            if c.parts and hasattr(c.parts[0], "text")
        ]
    )

    if not history_text.strip():
        print("⚠️ No content to summarize. Returning as-is.")
        return all_contents

    if previous_summary:
        summarization_prompt = f"""You are updating your internal memory of this ongoing conversation. Create a NEW, COMPLETE narrative that includes ALL previous context PLUS the new interactions.

CRITICAL REQUIREMENTS:
- Write in 1-5 cohesive paragraphs (no more than 5)
- MERGE the previous memory with new information into ONE continuous story
- Include EVERY detail from both: previous context + new messages
- Write in past tense, maintaining chronological flow
- Be specific and concrete - don't lose ANY information
- This is a CUMULATIVE memory - it should grow with each update
- DO NOT restart the story - CONTINUE and EXPAND it

YOUR TASK:
Take the PREVIOUS MEMORY below and SEAMLESSLY ADD the NEW INTERACTIONS to create ONE COMPLETE, UPDATED narrative. Every detail from both sections must be preserved.

Write in English. Do not use bullet points, headers, or lists - only flowing paragraphs.

═══════════════════════════════════════
PREVIOUS MEMORY (MUST BE FULLY PRESERVED AND EXPANDED):
{previous_summary}
═══════════════════════════════════════

NEW INTERACTIONS SINCE LAST SUMMARY (ADD THIS TO THE STORY):
{history_text}
═══════════════════════════════════════

Now write your COMPLETE ACCUMULATED memory (previous + new merged together):"""
    else:
        summarization_prompt = f"""You are creating an internal memory recap of this conversation. Write a natural, flowing narrative summary that captures the complete story of what happened, as if you're telling yourself what you remember about this interaction.

CRITICAL REQUIREMENTS:
- Write in 1-5 cohesive paragraphs (no more than 5)
- Use a natural, narrative storytelling style
- Include EVERY important detail: names, facts, numbers, decisions, context
- Write in past tense, as if recounting what occurred
- Be specific and concrete - don't generalize or omit details
- Maintain chronological flow where it matters
- This is your ONLY memory - capture everything essential

NARRATIVE STRUCTURE:
Start with who the user is and what they wanted. Then describe what was discussed, what information was shared, what decisions were made, and how things progressed. Include all relevant details naturally within the narrative flow. End with the current state or any pending matters.

Write in English. Do not use bullet points, headers, or lists - only flowing paragraphs.

CONVERSATION HISTORY TO SUMMARIZE:
{history_text}

Now write your complete narrative memory recap:"""

    try:
        summary_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text=summarization_prompt)]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.8,
            ),
        )
        summary = summary_response.text
        print(
            "✅ Summarization completed. The new memory will be used in the next turn."
        )
        print(f"📝 Summary:\n{summary}\n")

    except Exception as e:
        print(f"Failed to generate the summary: {e}. Keeping only recent messages.")
        keep_last = min(5, len(all_contents))
        return all_contents[-keep_last:]

    summary_content = types.Content(
        role="user",
        parts=[
            types.Part(
                text=f"""[MEMORY CONTEXT FROM PREVIOUS CONVERSATION]

{summary}

[END OF MEMORY CONTEXT - Continue the conversation naturally using this background information]"""
            )
        ],
    )

    return [summary_content]


def _format_tools(tools: List):
    """
    Formats the provided tools for use with the agent.
    If a tool is an Agent instance, creates a wrapper function that executes it.
    """
    from core.agent import Agent

    formatted_tools = []

    for tool in tools:
        if isinstance(tool, Agent):
            # Create a wrapper function that executes the agent
            def create_agent_wrapper(agent_instance):
                """Factory function to properly capture the agent in closure"""

                def agent_tool(query: str) -> str:
                    f"""
                        {agent_instance.description}
                        
                        This tool executes the {agent_instance.name} agent.
                        
                        Args:
                            query: A query or instruction to send to the agent
                            
                        Returns:
                            The agent's response as a string
                    """.strip()

                    try:
                        response = agent_instance.execute(entry=query)
                        return response
                    except Exception as e:
                        return f"Error executing {agent_instance.name}: {str(e)}"

                # Set function metadata using agent's name and description
                agent_tool.__name__ = agent_instance.name
                agent_tool.__doc__ = f"""
                    {agent_instance.description}

                    This tool executes the {agent_instance.name} agent.

                    Args:
                        query (str): A query or instruction to send to the agent

                    Returns:
                        str: The agent's response
                """.strip()

                return agent_tool

            # Create and add the wrapper function
            wrapped_function = create_agent_wrapper(tool)
            formatted_tools.append(wrapped_function)
        else:
            # Keep regular functions as-is
            formatted_tools.append(tool)

    return formatted_tools


def _format_prompt(name: str, description: str, prompt: str) -> str:
    """
    Formats the base prompt with additional instructions to reinforce behaviors
    and improve efficiency using Chain of Thought (CoT).

    Args:
        name: The agent's name
        description: The agent's description
        prompt: The base prompt to be enhanced

    Returns:
        Enhanced prompt in markdown format (English)
    """
    formatted_prompt = f"""
        # System Role
        ## Identity
        You are **{name}**, a highly capable and efficient assistant. 
        {description}

        # System Instructions
        {prompt}

        ---

        ## 🎯 CRITICAL EXECUTION PROTOCOL

        ### ⚠️ GOLDEN RULE: EXECUTE, DON'T JUST DESCRIBE!

        Follow this workflow for ALL requests:

        #### Step 1: Internal Analysis (Chain of Thought)
        - Analyze the user's request internally (DO NOT write this to the user).
        - Determine the required information and the tools/agents needed.
        - Plan the execution order of the tools.

        #### Step 2: IMMEDIATE Tool Execution
        - ✓ Execute the tools/agents IMMEDIATELY.
        - ✓ DO NOT ask for confirmation before using tools.
        - ✓ DO NOT describe what you're going to do—JUST DO IT.

        #### Step 3: Respond to the User
        - Present the results clearly and concisely.
        - Synthesize information if multiple tools were used.
        - Be direct, helpful, and actionable.

        ---

        ## ❌ PROHIBITED BEHAVIORS

        Avoid the following:
        - ❌ Describing actions instead of executing them.
        - ❌ Asking for confirmation before using tools.
        - ❌ Saying "I will consult" or "Let me check that for you."
        - ❌ Providing vague or incomplete responses.

        ---

        ## ✅ CORRECT BEHAVIORS

        Always:
        - ✓ Identify the necessary tools/agents.
        - ✓ Execute the tools IMMEDIATELY without hesitation.
        - ✓ Provide clear, direct, and actionable responses.
        - ✓ Use multiple tools in sequence when required.

        ---

        ## 📋 EXECUTION EXAMPLES

        ### ❌ WRONG:
        **User:** "What's the price of PETR4?"  
        **Agent:** "To get the price of PETR4, I will consult the agent_cotacoes."

        ### ✅ CORRECT:
        **User:** "What's the price of PETR4?"  
        **Agent:** [CALLS agent_cotacoes IMMEDIATELY]  
        **Agent:** "PETR4 is trading at R$ 35.40."

        ---

        ### ❌ WRONG:
        **User:** "Should I buy MGLU3?"  
        **Agent:** "I will consult the specialists to help you..."

        ### ✅ CORRECT:
        **User:** "Should I buy MGLU3?"  
        **Agent:** [CALLS agent_cotacoes, agent_fundamentos, agent_recomendacoes]  
        **Agent:** "MGLU3 is trading at R$ 2.15. Fundamentals show P/E of 15.2 and DY of 3.5%. The recommendation is Hold, with a target price of R$ 2.30."

        ---

        ## 🔥 REMEMBER

        You are an EXECUTOR, not a planner who merely describes actions.

        **IMMEDIATE ACTION > Description of intentions**

        When tools are available: **USE THEM NOW!**
    """.strip()

    return formatted_prompt
