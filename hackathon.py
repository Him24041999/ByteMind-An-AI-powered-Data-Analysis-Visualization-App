import streamlit as st
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def custom_show(*args, **kwargs):
    # Use a unique filename if you prefer unique names:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = os.path.join(os.getcwd(), f"plot_{timestamp}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close("all")
    return filename

# Uncomment the next line if you wish to override plt.show for all LLM-generated code.
plt.show = custom_show

model_choice = st.sidebar.radio("Select Language Model", ["ChatGPT", "Gemini"])

# Depending on the user's choice, import and configure the corresponding LLM.
if model_choice == "ChatGPT":
    # Use ChatGPT (via ChatOpenAI) configuration.
    from langchain.chat_models import ChatOpenAI
    OPENAI_API_KEY = "Please Enter you OpenAPI Key here"
    analysis_llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )
    viz_llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )
else:
    # Use Gemini (via ChatGoogleGenerativeAI) configuration.
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_API_KEY = "AIzaSyA4PBiyWaShXPXemFZv196fhKslrsMjnBA"
    MODEL_NAME = "gemini-1.5-pro"
    analysis_llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=MODEL_NAME,
        temperature=0.7,
    )
    viz_llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=MODEL_NAME,
        temperature=0.7
    )

# Import common LangChain components.
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

st.title("GenBI - Data Analysis & Visualization App")

# File uploader for CSV.
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    # Load the CSV data.
    df = pd.read_csv(uploaded_file)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.write("Data loaded successfully!")
    st.dataframe(df.head())

    # ---------------------------
    # 1) Analysis Agent (pandas-based)
    # ---------------------------
    analysis_agent = create_pandas_dataframe_agent(
        llm=analysis_llm,
        df=df,
        verbose=False,
        allow_dangerous_code=True
    )
    analysis_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=analysis_agent.agent,
        tools=analysis_agent.tools,
        verbose=True
    )

    # ---------------------------
    # 2) Visualization Agent
    # ---------------------------
    viz_system_message = SystemMessage(content="""
You are a visualization expert with access to a DataFrame named 'df'.
You can create ANY kind of chart (histogram, scatter, line, boxplot, etc.) using matplotlib or seaborn.
You must save exactly ONE plot to 'my_plot.png' and do not generate multiple figures.
Return the final result or file path to the saved image.
""")
    viz_agent = create_pandas_dataframe_agent(
        llm=viz_llm,
        df=df,
        verbose=False,
        allow_dangerous_code=True,
        agent_kwargs={"system_message": viz_system_message}
    )
    viz_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=viz_agent.agent,
        tools=viz_agent.tools,
        verbose=True
    )

    # ---------------------------
    # 3) Manager / Router with Conversation Memory
    # ---------------------------
    manager_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def manager_route_query(query: str) -> str:
        lower_q = query.lower()
        if any(kw in lower_q for kw in ["chart", "plot", "visual", "bar chart", "line chart", "diagram", "histogram"]):
            st.write("[Manager] Routing to Visualization Agent")
            return viz_agent_executor.run({"input": query, "chat_history": manager_memory.chat_memory.messages})
        else:
            st.write("[Manager] Routing to Analysis Agent")
            return analysis_agent_executor.run({"input": query, "chat_history": manager_memory.chat_memory.messages})

    # ---------------------------
    # 4) User Query Interface
    # ---------------------------
    query = st.text_input("Enter your query about the dataset:")
    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing query..."):
                result = manager_route_query(query)
            st.subheader("Response:")
            st.write(result)
            # If the result indicates a file path (for visualization), display the image.
            if isinstance(result, str) and result.strip().endswith(".png"):
                file_path = result.strip()
                if os.path.exists(file_path):
                    st.image(file_path, caption="Generated Plot", use_column_width=True)
                else:
                    st.error("Image file not found: " + file_path)
        else:
            st.error("Please enter a query.")
else:
    st.info("Please upload a CSV file to get started.")
