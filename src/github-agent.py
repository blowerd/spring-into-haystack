from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo
from haystack.utils import Secret
from haystack.tools import tool
from typing import Annotated
from pydantic import BaseModel
from spellchecker import SpellChecker
from dotenv import load_dotenv
import re

load_dotenv()

ALLOW_LIST = {"haystack", "github", "mcp", "json", "readme", "markdown", "api"}
class SpellcheckParams(BaseModel):
    text: str

def clean_markdown(text: str) -> str:
    # Remove code blocks (```...```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code (`...`)
    text = re.sub(r"`.*?`", "", text)
    # Remove markdown links and images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    # Remove markdown formatting (*, _, etc.)
    text = re.sub(r"[*_~#>`-]", "", text)
    return text

def extract_words(text: str) -> list:
    # Extract only alphabetic words (ignore numbers, symbols)
    return re.findall(r"\b[a-zA-Z]{2,}\b", text)

@tool
def spellcheck_text(
    text: Annotated[str, "The raw markdown text to spellcheck"]
) -> dict:
    """
    Check for misspelled words in a markdown document, ignoring code blocks and formatting.
    """
    cleaned_text = clean_markdown(text)
    words = extract_words(cleaned_text)

    spell = SpellChecker()
    spell.word_frequency.load_words(ALLOW_LIST)

    misspelled = spell.unknown(words)
    suggestions = {word: spell.correction(word) for word in misspelled}

    return {
        "misspelled": list(misspelled),
        "suggestions": suggestions
    }


github_mcp_server = StdioServerInfo(
        ## TODO: Add correct params for the Github MCP Server (official or legacy)
        command = "docker",
        args = ["run","-i","--rm","-e","GITHUB_PERSONAL_ACCESS_TOKEN","ghcr.io/github/github-mcp-server"],
        env={
            "GITHUB_PERSONAL_ACCESS_TOKEN": Secret.from_env_var("GITHUB_PERSONAL_ACCESS_TOKEN").resolve_value()
        }
    )

print("MCP server is created")

## TODO: Create your tools here:
#tool_find_files = MCPTool(name="search_code", server_info=github_mcp_server)
tool_read_file = MCPTool(name="get_file_contents", server_info=github_mcp_server)
tool_create_issue = MCPTool(name="create_issue", server_info=github_mcp_server)

tools = [tool_read_file, tool_create_issue, spellcheck_text]

print("MCP tools are created")

## TODO: Create your Agent here:
agent = Agent(
    tools = tools,
    chat_generator = OpenAIChatGenerator(model = "o4-mini")
)

print("Agent created")

## Example query to test your agent
user_input = """Please check for spelling mistakes in the `README.md` file of the repository blowerd/spring-into-haystack. 
First, use the `tool_read_file` tool to retrieve the file. Then, extract the plain text (excluding code blocks and markdown formatting) and pass it to the `spellcheck_text` tool.
Only suggest corrections for words that are clearly misspelled. 
If any issues are found, use the `tool_create_issue` tool to open a GitHub issue in the same repository with a title like "Spelling Errors Found in README" and include the list of misspelled words and suggested corrections in the body."""

## (OPTIONAL) Feel free to add other example queries that can be resolved with this Agent

response = agent.run(messages=[ChatMessage.from_user(text=user_input)])

## Print the agent thinking process
print(response)
## Print the final response
print(response["messages"][-1].text)