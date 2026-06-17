
import pandas as pd
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
import os
import time
from google.api_core.exceptions import ResourceExhausted

CSV_FILE_PATH = "Flight Bookings.csv"
df = pd.read_csv(CSV_FILE_PATH)


def inspect_dataframe() -> str:
    try:
        info = []
        info.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        info.append(f"Columns: {list(df.columns)}")
        info.append("Data types:\n" + df.dtypes.to_string())
        info.append("Null counts:\n" + df.isnull().sum().to_string())
        info.append("Preview (first 5 rows):\n" + df.head(5).to_string(index=False))
        return "\n\n".join(info)
    except Exception as e:
        return f"Error inspecting dataframe: {e}"

inspect_tool = FunctionTool.from_defaults(
    fn=inspect_dataframe,
    name="inspect_dataframe",
    description="Returns info about dataframe: shape, columns, dtypes, nulls, preview."
)


def run_pandas_code(code: str) -> str:
    global df
    try:
        local_vars = {"df": df, "pd": pd}
        exec(code, {"pd": pd}, local_vars)
        if "df" in local_vars:
            df = local_vars["df"]
        return "✅ Code executed successfully."
    except Exception as e:
        return f"❌ Error running code: {e}"

pandas_exec_tool = FunctionTool.from_defaults(
    fn=run_pandas_code,
    name="run_pandas_code",
    description="Run pandas code on the dataframe to clean or enhance it."
)


def save_cleaned_csv(filename: str = "cleaned_output.csv") -> str:
    try:
        df.to_csv(filename, index=False)
        return f"✅ Cleaned DataFrame saved to '{filename}'."
    except Exception as e:
        return f"❌ Failed to save CSV: {e}"

save_tool = FunctionTool.from_defaults(
    fn=save_cleaned_csv,
    name="save_cleaned_csv",
    description="Save the modified DataFrame to a new CSV file."
)


import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    max_retries=3,
)
agent = ReActAgent.from_tools(
    tools=[inspect_tool, pandas_exec_tool, save_tool],
    llm=llm,
    verbose=True,
    max_iterations=100
)
print(agent.get_prompts())

task = """
Inspect the dataframe first. Then clean and enhance it by:
- Reviewing data types, unique values, and basic statistics to understand each column
- Removing duplicate rows
- Handling missing values by either filling with appropriate defaults or dropping them based on context
- Standardizing column names: convert to lowercase and replace spaces/special characters with underscores
- Converting date columns to proper datetime format after detecting them
- Removing redundant information in cells if any
- Fixing common data inconsistencies such as:
  - Text casing differences (e.g., 'Yes', 'yes', 'YES')
  - Trailing or leading spaces
  - Misformatted values in categorical fields
- Creating new columns if needed (e.g., extracting year from date, calculating age from birthdate)
- Verifying derived columns (if any exist) to ensure their values logically match the source data
- Renaming columns to clean, business-friendly names for better readability and easier querying

Note: Understand each column’s role before applying transformations. Use available tools to guide decisions.
Try to find as many anomalies as possible and fix them using the available tools and if not able to fix mention those aswell.
Finally, save the cleaned dataframe to 'cleaned_output.csv' and provide a concise summary of all cleaning and improvments steps performed.
"""

max_retries = 5
initial_delay = 20  # seconds

for i in range(max_retries):
    try:
        response = agent.chat(task)
        print("\n=== FINAL AGENT RESPONSE ===\n")
        print(response)
        break  # If successful, break the loop
    except ResourceExhausted as e:
        if i < max_retries - 1:
            delay = initial_delay * (2 ** i)
            print(f"Quota exceeded. Retrying in {delay} seconds... (Attempt {i+1}/{max_retries})")
            time.sleep(delay)
        else:
            print(f"Max retries reached. Could not complete the task due to quota limitations: {e}")
            raise # Re-raise the exception if max retries are exhausted
