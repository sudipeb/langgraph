import os

import certifi
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if not ssl_cert_file or not os.path.exists(ssl_cert_file):
    os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

@tool()
def triple(num: float) -> float:
    """
    param num:a number to triple
    returns: the triple of the input
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)