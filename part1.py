from openai import OpenAI
from duckduckgo_search import DDGS
import re

client = OpenAI(api_key="Empty", base_url="http://127.0.0.1:11434/v1")


def search(query, max_results=10):
    results = DDGS().text(query, max_results=max_results)
    return str("\n".join(str(results[i]) for i in range(len(results))))


def extract_action_and_input(text):
    action_pattern = r"Action: (.+?)\n"
    input_pattern = r"Action Input: \"(.+?)\""
    action = re.findall(action_pattern, text)
    action_input = re.findall(input_pattern, text)
    return action, action_input


System_prompt = """
Answer the following questions as best you can. Use web search only if necessary.

You have access to the following tool:

Search: useful for when you need to answer questions about current events or when you need specific information that you don't know. However, you should only use it if the user's query cannot be answered with general knowledge or a straightforward response.

You will receive a message from the human, then you should start a loop and do one of the following:

Option 1: Respond to the human directly
- If the user's question is simple, conversational, or can be answered with your existing knowledge, respond directly.
- If the user did not ask for a search or the question does not need updated or specific information, avoid using the search tool.

Use the following format when responding to the human:
Thought: Explain why a direct response is sufficient or why no search is needed.
Action: Response To Human
Action Input: "your response to the human, summarizing what you know or concluding the conversation"

Option 2: Use the search tool to answer the question.
- If you need more information to answer the question, or if the question is about a current event or specific knowledge that may not be in your training data, use the search tool.
- After receiving search results, decide whether you have enough information to answer the question or if another search is necessary.

Use the following format when using the search tool:
Thought: Explain why a search is needed or why additional searches are needed.
Action: Search
Action Input: "the input to the action, to be sent to the tool"

Remember to always decide carefully whether to search or respond directly. If you can answer the question without searching, always prefer responding directly.
"""


def run_agent(prompt, system_prompt):
    # Prepare the initial message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    previous_searches = set()  # Track previous search queries to avoid repetition

    while True:
        response = client.chat.completions.create(
            model="llama3.1",
            messages=messages,
        )
        response_text = response.choices[0].message.content
        print(response_text)

        action, action_input = extract_action_and_input(response_text)
        if not action or not action_input:
            return "Failed to parse action or action input."

        if action[-1] == "Search":
            search_query = action_input[-1].strip()

            # Check if the search query has been performed before
            if search_query in previous_searches:
                print("Repeated search detected. Stopping to prevent infinite loop.")
                break

            print(
                f"================ Performing search for: {search_query} ================"
            )
            observation = search(search_query)
            print("================ Search completed !! ================")
            previous_searches.add(search_query)  # Add to the set of performed searches

            messages.extend(
                [
                    {"role": "system", "content": response_text},
                    {"role": "user", "content": f"Observation: {observation}"},
                ]
            )

        elif action[-1] == "Response To Human":
            print(f"Response: {action_input[-1]}")
            break

        # Prevent infinite looping
        if len(previous_searches) > 3:
            print("Too many searches performed. Ending to prevent infinite loop.")
            break

    return action_input[-1]


response = run_agent(prompt="Hi", system_prompt=System_prompt)
