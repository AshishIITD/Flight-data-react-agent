import os
import io
import time
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.google_genai import GoogleGenAI
from google.api_core.exceptions import ResourceExhausted
from utils import emplate
from plot_tool import generate_plot, clear_plots_directory, generate_subplots, generate_aggregated_plot
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CLEANED_CSV_FILE_PATH = "cleaned_output.csv"
qna_df = None

llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    max_retries=3,
)

def run_pandas_code_qna(code: str) -> str:
    global qna_df
    try:
        local_vars = {"df": qna_df, "pd": pd}
        exec(code, {"pd": pd}, local_vars)
        if "df" in local_vars:
            qna_df = local_vars["df"]
        return "✅ Code executed successfully. Output:\n" + str(local_vars.get("result", "No `result` variable found."))
    except Exception as e:
        return f"❌ Error running code: {e}"

pandas_exec_tool_qna = FunctionTool.from_defaults(
    fn=run_pandas_code_qna,
    name="run_pandas_code_qna",
    description="Run pandas code on the dataframe. Code must assign result to a variable named 'result'."
)

def create_csv_context_prompt(csv_path: str, enum_threshold: int = 20) -> str:
    try:
        df_temp = pd.read_csv(csv_path)
        filename = os.path.basename(csv_path)
        columns = df_temp.columns.tolist()
        dtypes = df_temp.dtypes.to_string()
        head = df_temp.head().to_string()

        buffer = io.StringIO()
        df_temp.info(buf=buffer)
        df_info = buffer.getvalue()

        description = df_temp.describe(include='all').to_string()

        column_enums = []
        for col in df_temp.columns:
            if df_temp[col].nunique() < enum_threshold:
                unique_values = df_temp[col].unique().tolist()
                column_enums.append(f"- `{col}`: {unique_values}")

        enum_details_prompt = ""
        if column_enums:
            enums_string = "\n".join(column_enums)
            enum_details_prompt = f"""
---
**Column Value Examples (Enums)**
Columns with fewer than {enum_threshold} unique values:
{enums_string}
"""

        return f"""

**CSV File**: {filename}

**Columns**: {columns}

airlie_id,airline_name
1,American Airline
2,Delta Air Lines
3,United Airlines
4,Southwest Airlines
5,JetBlue Airways
6,Alaska Airlines
7,Spirit Airlines
8,Frontier Airlines
9,Hawaiian Airlines
10,Allegiant Air
11,SkyWest Airlines
12,Air Canada
13,British Airways
14,Lufthansa
15,Air France
16,KLM Royal Dutch Airlines
17,Qantas
18,Emirates
19,Singapore Airlines
20,Cathay Pacific

**Data Types**:
{dtypes}

**First 5 Rows**:
{head}

**Info Summary**:
{df_info}

{enum_details_prompt}
""".strip()

    except Exception as e:
        return f"❌ Failed to read CSV: {e}"


qna_df = pd.read_csv(CLEANED_CSV_FILE_PATH)

template=emplate.replace("{df_context}",create_csv_context_prompt(csv_path='cleaned_output.csv'))

from llama_index.core.agent.react.formatter import ReActChatFormatter
from plot_tool import generate_plot, clear_plots_directory, generate_subplots, generate_aggregated_plot

plot_tool = FunctionTool.from_defaults(
    fn=generate_plot,
    name="generate_plot",
    description="Generates and saves a single plot from the dataframe. Arguments: df (the dataframe, always pass the global qna_df), x_col (column for x-axis), y_col (optional, column for y-axis), plot_type (e.g., 'scatter', 'line', 'bar', 'hist', 'box', 'kde'), title (plot title), filename (e.g., 'my_plot.png'). Always ensure filename is unique for each plot."
)

aggregated_plot_tool = FunctionTool.from_defaults(
    fn=generate_aggregated_plot,
    name="generate_aggregated_plot",
    description="Generates and saves a plot from pre-aggregated data (lists). Arguments: x_data (list of x-values), y_data (list of y-values), x_label (label for x-axis), y_label (label for y-axis), plot_type (e.g., 'bar', 'line', 'scatter'), title (plot title), filename (e.g., 'my_aggregated_plot.png'). Use this tool when you have aggregated data and want to plot it directly."
)

subplots_tool = FunctionTool.from_defaults(
    fn=generate_subplots,
    name="generate_subplots",
    description="Generates and saves a figure with multiple subplots from the dataframe. Arguments: df (the dataframe, always pass the global qna_df), plot_specs (a list of dictionaries, each specifying a subplot with 'x_col', 'y_col' (optional), 'plot_type', 'title'), filename (e.g., 'my_subplots.png'). Use this tool when multiple related plots are required. Always ensure filename is unique for each subplot figure."
)

agent = ReActAgent.from_tools(
    tools=[pandas_exec_tool_qna, plot_tool, aggregated_plot_tool, subplots_tool],
    llm=llm,
    verbose=True,
    max_iterations=30,
    react_chat_formatter=ReActChatFormatter(system_header=template)
)
print("Chat with the Q&A Agent (type 'exit' to quit):")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = agent.chat(user_input)
        print(f"Agent: {response}")
    except ResourceExhausted:
        print("ResourceExhausted: You have exceeded your API quota. Please wait and try again.")
        time.sleep(60) # Wait for a minute before retrying
    except Exception as e:
        print(f"An error occurred: {e}")
