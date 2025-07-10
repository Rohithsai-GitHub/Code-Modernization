import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("`GOOGLE_API_KEY` not found in environment variables. Please set it in a `.env` file or directly in your environment.")
    st.stop()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM. Check your `GOOGLE_API_KEY` and internet connection. Error: {e}")
    st.stop()

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
OUTPUT_LANGUAGES = {
    "JavaScript": "javascript",
    "Python": "python",
    "PHP": "php",
    "Ruby": "ruby"
}

checker = PromptTemplate.from_template("""
    Your task is to classify the given input. 
    Respond with just 'CODE' if the input appears to be a programming language snippet, script, or configuration file. 
    Respond with just 'TEXT' if the input is written in natural human language, such as an article, text, words, email, or casual conversation.
    Input: {code}
    """)

checker_chain = checker | llm


# Prompt for code conversion
conversion_template = ChatPromptTemplate([
    ("system", "You are an expert code converter."),
    ("user", """Convert the following {input_language} code to {output_language} code. 
    Focus on maintaining logic and functionality, and use idiomatic {output_language} where appropriate. 
    Provide only the converted code, without any extra explanations, comments outside the code, or markdown comments like 
    'Here is the converted code:'.\n\n{input_language} Code:\n```\n{code}\n```\n\n{output_language} Code:""")
    ])

conversion_chain = conversion_template | llm

# Prompt for code readability improvement
readability_template = ChatPromptTemplate([
    ("system", "You are an expert code improver."),
    ("user", """Improve the readability, clarity, and adherence to standard best practices of the following {language} code.
    This includes better variable names, consistent formatting, comments where necessary, and breaking down complex parts. 
    Provide only the improved code, without any extra explanations, comments outside the code, or markdown comments like 
    'Here is the improved code:'.\n\n{language} Code:\n\n{code}\n\n\nImproved {language} Code:""")
])
readability_chain = readability_template | llm

# Streamlit UI
st.set_page_config(layout="wide", page_title="Universal Code Converter & Enhancer")
st.title("🚀 Universal Code Translator & Enhancer")
st.write("Translate code between different programming languages or improve readability of existing code.")
st.info("💡 **Tip:** If you select the same input and output language, the tool will automatically enhance the code's readability!")

col1, col2 = st.columns(2)

with col1:
    st.header("Input Code")
    input_language_name = st.selectbox("Select Input Language", list(LANGUAGES.keys()), index=1) # Default to Python
    input_language_slug = LANGUAGES[input_language_name]
    input_code = st.text_area("Paste your code here...", height=400, key="input_code")

with col2:
    st.header("Output Code")
    output_language_name = st.selectbox("Select Output Language", list(OUTPUT_LANGUAGES.keys()), index=1) # Default to Python
    output_language_slug = LANGUAGES[output_language_name]
    output_code_display = st.empty()

st.markdown("---")
if st.button("✨ Process Code", use_container_width=True, help="Convert code or improve readability"):
    if not input_code:
        st.warning("Please enter some code to process.")
    else:
        with st.spinner("Processing your code... This might take a moment."):
            try:
                res = checker_chain.invoke({"code": input_code}).content
                if res == "TEXT":
                    raise ValueError("Expected CODE, not TEXT")
                if input_language_slug == output_language_slug:
                    processed_code = readability_chain.invoke({"language": input_language_name, "code": input_code}).content
                    output_code_display.code(processed_code, language=output_language_slug)
                    st.success(f"Code readability improved for {input_language_name}!")
                else:
                    processed_code = conversion_chain.invoke(
                        {"input_language": input_language_name,
                        "output_language": output_language_name,
                        "code": input_code}
                    ).content

                    output_code_display.code(processed_code, language=output_language_slug)
                    st.success(f"Code converted from {input_language_name} to {output_language_name}!")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.info("Please check your input code, selected languages, and ensure your Google API Key is valid and active.")

st.markdown("---")
st.markdown("""
Developed using Streamlit, LangChain, and Google Gemini LLM.
""")
