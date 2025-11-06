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
    """
    formatted_tools = []
    for tool in tools:
        formatted_tools.append(tool)
    return formatted_tools
