import streamlit as st
import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import pandas as pd
from io import StringIO, BytesIO
import streamlit as st
import base64
from docx import Document
import toml
import os

# Everything is accessible via the st.secrets dict:
st.write("OPENAI_API_KEY:", st.secrets["OPENAI_API_KEY"])
st.write("SERP_API_KEY:", st.secrets["SERP_API_KEY"])
st.write("BROWSERLESS_API_KEY:", st.secrets["BROWSERLESS_API_KEY"])

st.write(
    "Has environment variables been set:",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
     os.environ["SERP_API_KEY"] == st.secrets["SERP_API_KEY"],
     os.environ["BROWSERLESS_API_KEY"] == st.secrets["BROWSERLESS_API_KEY"],
)

# Now you can use these variables throughout your script

def chunk_text(text, chunk_size=5000):
    """Break text into chunks of a specific size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def process_in_chunks(text, agent):
    """Process text in chunks and combine results."""
    chunks = chunk_text(text)
    results = []
    for chunk in chunks:
        result = agent({"input": chunk})
        results.append(result['output'])
    return ' '.join(results)

def read_docx(file):
    doc = Document(file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx"])

docx_text = None  # To store the text of the uploaded docx
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="450"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        docx_text = read_docx(uploaded_file)
        st.write(docx_text)



        
load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def append_to_txt(content, filename="research_results.txt"):
    """Append content to a txt file and return the path."""
    with open(filename, 'a') as f:
        f.write(content + '\n')
    return filename

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# search("list youtube videos to learn for AZ104")

# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


# scrape_website("what sould i study for AZ-104", "https://learn.microsoft.com/en-us/certifications/exams/az-104/")

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# 4. Use streamlit to create a web app
def main():

 st.header("AI research agent :bird:")
 query = st.text_input("Research goal", key='initial_query')


   # Initialize session state for accumulated results
 if "accumulated_results" not in st.session_state:
     st.session_state.accumulated_results = ""


 # Ensure both a document is uploaded and a query is entered
 combined_text = str(docx_text) + " " + query
 st.write("Doing research for ", combined_text)
 result_text = process_in_chunks(combined_text, agent)
 st.session_state.accumulated_results += result_text

 st.info(result_text)
 st.sidebar.markdown(str(result_text))
 # New input for the user to continue research
 follow_up_query = st.text_input(
    "Continue your research with a new question", key='follow_up_query')


 combined_follow_up_text = result_text+ " " + follow_up_query
 st.write("Continuing research for ", combined_follow_up_text)

 follow_up_result_text = process_in_chunks(combined_follow_up_text, agent)
 st.session_state.accumulated_results += follow_up_result_text
 st.info(follow_up_result_text)

    # Append follow-up result to a text file
 st.sidebar.markdown(follow_up_result_text)




def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1.x\n"  # noqa: E501
            "2. y📄\n"
            "3. z💬\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )

        st.session_state["OPENAI_API_KEY"] = api_key_input

        st.markdown("---")
       
        st.markdown("Made by [BlackFox](https://twitter.com/mm_sasmitha)")
        st.markdown("---")
 
sidebar()




if __name__ == '__main__':
    main()


