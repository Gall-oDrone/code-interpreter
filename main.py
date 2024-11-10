from typing import Any, Dict
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_csv_agent
from langchain_experimental import Tool

load_dotenv()

class BaseAgent:
    def __init__(self, instructions: str, model: str = "gpt-4", temperature: float = 0):
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        self.tools = [PythonREPLTool()]
        self.prompt = self._create_prompt()
        self.agent_executor = None

    def _create_prompt(self):
        base_prompt = hub.pull("langchain-ai/react-agent-template")
        return base_prompt.partial(instructions=self.instructions)
    
    def create_agent_executor(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def invoke(self, input_data: str) -> Dict[str, Any]:
        if not self.agent_executor:
            self.create_agent_executor()
        return self.agent_executor.invoke({"input": input_data})

class PythonQRCodeAgent(BaseAgent):
    def create_agent_executor(self):
        agent = create_react_agent(
            prompt=self.prompt,
            llm=ChatOpenAI(temperature=self.temperature,model="gpt-4-turbo"),
            tools=self.tools,
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

class PythonCSVAgent(BaseAgent):
    def __init__(self, instructions: str, csv_path: str):
        super().__init__(instructions)
        self.csv_path = csv_path

    def create_agent_executor(self):
        self.agent_executor = create_csv_agent(
            llm=ChatOpenAI(temperature=self.temperature,model=self.model),
            path=self.csv_path,
            verbose=True,
            allow_dangerous_code=True,
        )
    
class PythonRouterGrandAgent(BaseAgent):
    def __init__(self, instructions: str, python_agent: PythonQRCodeAgent, csv_agent:PythonCSVAgent):
        super().__init__(instructions)
        self.python_agent_executor=python_agent.agent_executor
        self.csv_agent_executor = csv_agent.agent_executor

    def create_agent_executor(self):
        tools = [
            Tool(
                name="Python Agent",
                func=self.python_agent_executor.invoke,
                description="""Useful when you need to transform natural language to python and execute the python code. 
                               DOES NOT ACCEPT CODE AS INPUT."""
            ),
            Tool(
                name="CSV Agent",
                func=self.csv_agent_executor.invoke,
                description="""Useful when you need to answer questions over 2024-05-02_full_record.csv file, 
                               takes an input the entire question and returns the answer after running pandas calculations."""
            ),
        ]
        self.prompt = self._create_prompt().partial(instructions="")
        grand_agent = create_react_agent(
            prompt=self.prompt,
            llm=ChatOpenAI(temperature=self.temperature, model="gpt-4-turbo"),
            tools=tools,
        )
        self.agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

def main():
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code to answer the question.
    Only use the output of your code to answer the question.
    You might know the answer running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer."""

    csv_path = "/Users/diegogallovalenzuela/genAI/LangChainCourse/code-interpreter/csv/2024-05-02_full_record.csv"
    
    # Instantiate agents
    python_qrcode_agent = PythonQRCodeAgent(instructions)
    python_csv_agent = PythonCSVAgent(instructions, csv_path)
    
    # Run CSV agent
    csv_response = python_csv_agent.invoke(
        "print in descending order the result of calculating the unique books for the 'book' column which had the highest mean value for the column 'close' in file 2024-05-02_full_record.csv"
    )
    print(csv_response)
    
    # Run QR code agent
    qr_response = python_qrcode_agent.invoke(
        "generate and save in current working directory 15 QRcodes that point to www.udemy.com/course/langchain, you have qrcode package installed already"
    )
    print(qr_response)
    
    # Router Grand Agent
    python_router_grand_agent = PythonRouterGrandAgent("", python_qrcode_agent, python_csv_agent)
    router_response = python_router_grand_agent.invoke(
        "generate and save in current working directory 15 QRcodes that point to www.udemy.com/course/langchain, you have qrcode package installed already"
    )
    print(router_response)


if __name__ == '__main__':
    main()