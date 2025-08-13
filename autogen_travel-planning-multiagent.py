import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
import requests
from autogen_core.tools import FunctionTool

model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key="gemini-api-key", # Replace with your actual Google Gemini API key
)

google_search_key = "google-search API key" # Replace with your actual Google Search API key
search_engine_ID = "google-search-engine-key" # Replace with your actual Google Custom Search Engine ID

def web_search(query: str) -> str:
    """
    Perform a web search using Google Custom Search API to fetch information about travel costs.
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,               # The search query
        'key': google_search_key,    # Your Google API key
        'cx': search_engine_ID,             # Custom Search Engine ID
        'num': 10                  # Number of search results to return (default is 10)
    }
    
    try:
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            search_results = response.json()  # Parse the JSON response
            results = []
            
            # Extract relevant information (title, link, snippet) from the response
            for item in search_results.get('items', []):
                result = {
                    'title': item.get('title'),
                    'link': item.get('link'),
                    'snippet': item.get('snippet')
                }
                results.append(result)
            print(results)
            return results
        else:
            print(f"Error: Received status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

google_search_tool = FunctionTool(
    web_search, 
    description="Search google for information, return results with a title, link, snippet"
)

local_agent = AssistantAgent(
    "local_agent",
    model_client=model_client,
    description="A local assistant that can provide all required assumptions for a trip plan.",
    system_message="YOU PROVIDE ALL ASSUMPTIONS FOR TRIP PLAN to next agent",
)

search_agent = AssistantAgent(
    "search_agent",
    model_client=model_client,
    tools = [google_search_tool],
    description="Search google for information, returns results with a snippet, title, link.",
    system_message="Your an helpfull assistent, " \
    "YOU ACCESS YOUR TOOLS, SEARCH INFORMATION USING YOUR TOOL, " \
    "YOU CAN ACCESS REAL-TIME INFORMATION FROM YOUR TOOLS, " \
    "YOU ONLY PROVIDE DETAILED SERACH RESULTS TO PLANNER AGENT, " \
    "YOU DONT CREATE PLAN, PROVIDE INFORMATION LINKS FROM SEARCH RESULTS AS REFERENCES",
    reflect_on_tool_use=True,
    tool_call_summary_format="{results}",
)

planner_agent = AssistantAgent(
    "planner_agent",
    model_client=model_client,
    description="Generate output only based on search results",
    system_message="YOU are an helpfull assistent, You get Information from Search Agent, " \
    "If information not enough repeat the chat with Search Agent, " \
    "Your a research analyst and a planner, You can analyse the search results and suggest a " \
    "travel plan for a user based on their request.",
)

language_agent = AssistantAgent(
    "language_agent",
    model_client=model_client,
    description="A helpful assistant that can provide language tips for a given destination.",
    system_message="You are a helpful assistant that can review travel plans, " \
    "providing feedback on important/critical tips about how best to address " \
    "language or communication challenges for the given destination. " \
    "If the plan already includes language tips, you can mention that the plan is satisfactory, " \
    "with rationale.",
)

travel_summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=model_client,
    description="A helpful assistant that can summarize the travel plan.",
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and PROVIDE DETAILED FINAL PLAN. " \
    "You must ensure that the final plan is integrated and complete. " \
    "YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. " \
    "When the plan is complete and all perspectives are integrated, First Provide Complete Plan Summer Details and you can respond with TERMINATE.",
)

termination = TextMentionTermination("TERMINATE")
group_chat = RoundRobinGroupChat(
    [local_agent, search_agent, planner_agent, language_agent, travel_summary_agent], max_turns = 8 )

async def travelplan_run() -> None:
    stream = group_chat.run_stream(task="Plan a 3 day trip to Nepal with detailed Budget Plan.")
    await Console(stream)

asyncio.run(travelplan_run())    