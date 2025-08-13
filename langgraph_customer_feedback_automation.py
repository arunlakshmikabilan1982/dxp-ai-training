import os
import streamlit as st
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_mongodb.agent_toolkit import MONGODB_AGENT_SYSTEM_PROMPT
from langchain_salesforce import SalesforceTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START
import re
import json
from langchain_openai import ChatOpenAI
from pymongo import MongoClient

# === ENV CONFIG ===
os.environ["GOOGLE_API_KEY"] = "gemini-api-key" # Replace with your actual Google Gemini API key
os.environ["OPENAI_API_KEY"] = "sk-proj-..."  # Replace with your actual OpenAI key


# # === LLM Setup ===
# llm = ChatOpenAI(temperature=0)  # Used for tool reasoning, if needed

# === LLM Setup ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# MongoDB Configuration
uri = "mongodb://localhost:27017/"
db_name = "Customer_Feedback"
collection_name = "feedback_collection"

# MongoDB Setup - works with OPENAI
# db = MongoDBDatabase.from_connection_string("mongodb://localhost:27017/")
# mongo_toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)

# Create Agent for MongoDB - works with OPENAI
# system_message = MONGODB_AGENT_SYSTEM_PROMPT.format(top_k=5)
# mongo_agent_executor = create_react_agent(
#     tools=mongo_toolkit.get_tools(),
#     llm=llm,
#     system_message=system_message
# )

# Create MongoClient instance
client = MongoClient(uri)
# Connect to database and collection
db = client["Customer_Feedback"]
collection = db["feedback_collection"]

# Salesforce Setup
sf_tool = SalesforceTool(
    username="salesforce-username", # Replace with your Salesforce username
    password="salesforce-password", # Replace with your Salesforce password
    security_token="salesforce-security-token" # Replace with your Salesforce security token
)

# Robust JSON extractor
def extract_json_from_llm(result: str, fallback: dict) -> dict:
    # Remove markdown/code fences
    cleaned = re.sub(r"```[a-zA-Z]*", "", result).strip("` \n")

    # Extract JSON block using regex
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        print("❌ No JSON block found in LLM output.")
        return fallback

    try:
        json_str = json_match.group()
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)  # Remove trailing commas
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("❌ JSON decoding error:", e)
        return fallback

# === STATE ===
class FeedbackState(TypedDict, total=False):
    feedback: str
    sentiment: str
    crm_result: str
    mongo_result: str

# === AGENTS ===
def classify_feedback(state: FeedbackState):
    prompt = PromptTemplate(
        input_variables=["feedback"],
        template="""
        Classify the sentiment of this customer feedback as Positive, Neutral, or Negative.
        Feedback: "{feedback}"
        Respond with only one word.
        """
    )
    chain = prompt | llm
    sentiment = chain.invoke({"feedback": state["feedback"]}).content.strip()
    return {"sentiment": sentiment}

def route_sentiment(state: FeedbackState):
    sentiment = state.get("sentiment", "").lower()
    return "support" if sentiment == "negative" else "store"

def crm_action(state: FeedbackState):
    combined_prompt = PromptTemplate(
        input_variables=["feedback", "sentiment"],
        template="""
        You are a support automation assistant.

        Given the following customer feedback and its sentiment, generate a structured support case in **strict JSON format**. Follow these rules:
        - DO NOT add any explanation or formatting (no markdown, no triple quotes).
        - Output ONLY the JSON object — no headers or comments.
        - Use proper JSON syntax — double quotes for keys and string values.
        - *** Respond with ONLY a raw JSON object. Do NOT use ```json or any markdown formatting.

        ### Feedback:
        "{feedback}"

        ### Sentiment:
        {sentiment}

        ### Respond with JSON exactly in this format:
        {{
        "subject": "<short, professional subject line>",
        "description": "<detailed and formal support description> and Include the Department that should handle it: "Technical Support" | "Billing" | "Shipping" | "General" in the Description ",
        "priority": "High" | "Medium" | "Low"
        }}
        """
    )

    chain = combined_prompt | llm
    result = chain.invoke({
        "feedback": state["feedback"],
        "sentiment": state["sentiment"]
    }).content.strip()
    print("🔍 Raw LLM Output:", result)
    fallback_data = {
        "subject": "Customer Feedback Case",
        "description": state["feedback"],
        "priority": "Medium"
    }
  
    parsed = extract_json_from_llm(result, fallback_data)
    print("✅ Parsed JSON:", parsed)
    
    request = {
        "operation": "create",
        "object_name": "Case",
        "record_data": {
            "Subject": parsed["subject"],
            "Description": parsed["description"],
            "Priority": parsed["priority"],
            "Origin": "Feedback App"
        }
    }

    res = sf_tool.run(request)
    return {"crm_result": f"Salesforce Case Created"}

def store_feedback(state: FeedbackState):
    # query = f"Insert the following feedback into the '{collection_name}' collection in database '{db_name}': {state['feedback']}"
    # mongo_agent_executor.invoke({"messages": [("user", query)]})

    result = collection.insert_one({"feedback": state['feedback']})
    return {"mongo_result": f"✅ Feedback inserted into MongoDB"}

# === GRAPH ===
workflow = StateGraph(FeedbackState)

workflow.add_node("classify", classify_feedback)
workflow.add_node("crm_action", crm_action)
workflow.add_node("store_feedback", store_feedback)
workflow.set_entry_point("classify")

workflow.add_conditional_edges("classify", route_sentiment, {
    "support": "crm_action",
    "store": "store_feedback"
})
workflow.add_edge("crm_action", END)
workflow.add_edge("store_feedback", END)

graph = workflow.compile()

# === STREAMLIT UI ===
st.set_page_config(page_title="Feedback Router", layout="centered")
st.title("🔁 Customer Feedback Processor")

user_feedback = st.text_area("Enter customer feedback:", height=150)

if st.button("Submit Feedback"):
    if user_feedback.strip():
        with st.spinner("Processing feedback..."):
            result = graph.invoke({"feedback": user_feedback})
        st.success("Feedback processed!")

        st.subheader("🔍 Feedback Flow Result")
        for k, v in result.items():
            st.write(f"**{k.upper()}**: {v}")
    else:
        st.warning("Please enter some feedback before submitting.")
