import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Ensure GOOGLE_API_KEY is set
if "GOOGLE_API_KEY" not in os.environ:
    st.error("`GOOGLE_API_KEY` not found in environment variables. Please set it in a `.env` file or directly in your environment.")
    st.stop()

# Initialize the Gemini LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM. Check your `GOOGLE_API_KEY` and internet connection. Error: {e}")
    st.stop()

# Trending languages for dropdowns
LANGUAGES = {
    "C++": "cpp",
    "Python": "python",
    "Java": "java",
    "JavaScript": "javascript",
    "Go": "go",
    "Rust": "rust",
    "TypeScript": "typescript",
    "C#": "csharp",
    "PHP": "php",
    "Ruby": "ruby"
}

# --- LangChain Prompts and Chains ---

# Prompt for code conversion
conversion_template = PromptTemplate(
    input_variables=["input_language", "output_language", "code"],
    template="You are an expert code converter. Convert the following {input_language} code to {output_language} code. Focus on maintaining logic and functionality, and use idiomatic {output_language} where appropriate. Provide only the converted code, without any extra explanations, comments outside the code, or markdown comments like 'Here is the converted code:'.\n\n{input_language} Code:\n```\n{code}\n```\n\n{output_language} Code:"
)
conversion_chain = LLMChain(llm=llm, prompt=conversion_template)

# Prompt for code readability improvement
readability_template = PromptTemplate(
    input_variables=["language", "code"],
    template="You are an expert code improver. Improve the readability, clarity, and adherence to standard best practices of the following {language} code. This includes better variable names, consistent formatting, comments where necessary, and breaking down complex parts. Provide only the improved code, without any extra explanations, comments outside the code, or markdown comments like 'Here is the improved code:'.\n\n{language} Code:\n```\n{code}\n```\n\nImproved {language} Code:"
)
readability_chain = LLMChain(llm=llm, prompt=readability_template)

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Universal Code Converter & Enhancer")
st.title("🚀 Universal Code Converter & Enhancer")
st.write("Convert code between different programming languages or improve readability of existing code.")

# Layout with two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.header("Input Code")
    input_language_name = st.selectbox("Select Input Language", list(LANGUAGES.keys()), index=1) # Default to Python
    input_language_slug = LANGUAGES[input_language_name]
    input_code = st.text_area("Paste your code here...", height=400, key="input_code")

with col2:
    st.header("Output Code")
    output_language_name = st.selectbox("Select Output Language", list(LANGUAGES.keys()), index=1) # Default to Python
    output_language_slug = LANGUAGES[output_language_name]
    output_code_display = st.empty() # Placeholder for displaying output code

# Conversion/Improvement Button
st.markdown("---")
if st.button("✨ Process Code", use_container_width=True, help="Convert code or improve readability"):
    if not input_code:
        st.warning("Please enter some code to process.")
    else:
        with st.spinner("Processing your code... This might take a moment."):
            try:
                if input_language_slug == output_language_slug:
                    # Same language selected: Improve readability
                    processed_code = readability_chain.run(language=input_language_name, code=input_code)
                    output_code_display.code(processed_code, language=output_language_slug)
                    st.success(f"Code readability improved for {input_language_name}!")
                else:
                    # Different languages selected: Convert
                    processed_code = conversion_chain.run(
                        input_language=input_language_name,
                        output_language=output_language_name,
                        code=input_code
                    )
                    output_code_display.code(processed_code, language=output_language_slug)
                    st.success(f"Code converted from {input_language_name} to {output_language_name}!")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.info("Please check your input code, selected languages, and ensure your Google API Key is valid and active.")

st.markdown("---")
st.info("💡 **Tip:** If you select the same input and output language, the tool will automatically enhance the code's readability!")
st.markdown("""
Developed with ❤️ using Streamlit, LangChain, and Google Gemini LLM.
""")
