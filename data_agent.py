from tavily import TavilyClient
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
import os

# Tavily API Client
tavily_client = TavilyClient(api_key="your_api_key")
llm = Ollama(model="llama3.2:3b")

# Define the sources to extract data from
DATA_SOURCES = {
    "OneMonthPlan": "https://www.straighttalk.com/support/one-month-plans/",
    "InternationalPlan": "https://www.straighttalk.com/support/international-plans/",
    "TermsConditions": "https://www.straighttalk.com/support/terms-conditions/",
    "PrivacyPolicy": "https://www.straighttalk.com/privacy-policy/"
}

class UserQuery(BaseModel):
    question: str = Field(..., title="User Question", description="The question the user asks the agent.")

class DataExtractorAgent:
    def __init__(self):
        self.client = tavily_client
        self.llm = llm

    def extract_data(self, url):
        try:
            print(f"Extracting data from: {url}")
            response = self.client.extract(url)
            
            if response and "results" in response and response["results"]:
                return response["results"][0]["raw_content"]
            else:
                print(f"No valid data extracted from {url}")
                return ""
        except Exception as e:
            print(f"Error extracting data from {url}: {e}")
            return ""

    def summarize_data(self, content):
        try:
            system_prompt = "You are an AI assistant. Summarize the following text into a concise and clear format, retaining all essential information. Do not add extra details."
            prompt = f"{system_prompt}\n\nContent:\n{content}"
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error summarizing data: {e}")
            return content

    def extract_and_save_data(self):
        for key, url in DATA_SOURCES.items():
            raw_content = self.extract_data(url)
            if raw_content:
                summarized_content = self.summarize_data(raw_content)
                file_name = f"{key}_Extracted.txt"
                
                with open(file_name, "w", encoding="utf-8") as file:
                    file.write(summarized_content)
                
                print(f"Summarized data successfully saved to {file_name}")

class ContextResponseAgent:
    def __init__(self):
        self.llm = Ollama(model="llama3.2:3b")

    def respond_to_query(self, user_query: UserQuery):
        try:
            extracted_files = ["OneMonthPlan_Extracted.txt", "InternationalPlan_Extracted.txt", "TermsConditions_Extracted.txt", "PrivacyPolicy_Extracted.txt"]
            context = ""
            for file_name in extracted_files:
                if os.path.exists(file_name):
                    with open(file_name, "r", encoding="utf-8") as file:
                        context += file.read() + "\n\n"
            
            system_prompt = "You are an AI assistant. Use the provided context to answer the user's question without adding extra details."
            prompt = f"{system_prompt}\n\nUser Question: {user_query.question}\n\nContext:\n{context}"
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"

if __name__ == "__main__":
    extractor_agent = DataExtractorAgent()
    extractor_agent.extract_and_save_data()
    
    # Example usage of the query agent
    query_agent = ContextResponseAgent()
    user_question = "What are the details of the one-month plans?"
    response = query_agent.respond_to_query(UserQuery(question=user_question))
    print(response)
