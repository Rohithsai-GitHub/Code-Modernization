import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API Key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["cpp_code"],
    template="You are an expert code converter. Convert the following C++ code to its equivalent Python code. Maintain the logic and functionality as much as possible, using idiomatic Python. Provide only the Python code, without any extra explanations or markdown comments outside the code block.\n\nC++ Code:\n```cpp\n{cpp_code}\n```\n\nPython Code:\n"
)

# Create an LLMChain
code_conversion_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit UI
st.set_page_config(page_title="C++ to Python Converter", layout="wide")
st.title("C++ to Python Code Converter")

st.write("Enter your C++ code below, and I'll convert it to Python!")

# Input text area for C++ code
cpp_code_input = st.text_area("C++ Code", height=300, help="Paste your C++ code here.")

if st.button("Convert to Python"):
    if cpp_code_input:
        with st.spinner("Converting... This might take a moment."):
            try:
                # Invoke the chain to get the conversion
                python_code_output = code_conversion_chain.run(cpp_code=cpp_code_input)
                st.subheader("Converted Python Code:")
                st.code(python_code_output, language="python")
            except Exception as e:
                st.error(f"An error occurred during conversion: {e}")
                st.info("Please ensure your Google API Key is valid and try again.")
    else:
        st.warning("Please enter some C++ code to convert.")

st.markdown("""
---
**How it works:**
This application uses Google's Gemini LLM via LangChain to perform the code conversion.
It sends your C++ code to the LLM with a specific prompt, asking it to translate the code into Python.
""")

st.subheader("Example C++ Code:")
st.code("""
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    long long sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
""", language="cpp")

st.markdown("Developed with ❤️ using Streamlit, LangChain, and Google Gemini LLM.")
