import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys
import io
import re
import time
import requests
import json

# Load environment variables
load_dotenv(override=True)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
JUDGE0_API_KEY = os.getenv('JUDGE0_API_KEY')  # New Judge0 API key
JUDGE0_ENDPOINT = "https://judge0-ce.p.rapidapi.com"  # Judge0 endpoint

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in .env file.")
if not JUDGE0_API_KEY:
    print("Warning: JUDGE0_API_KEY not found in .env file. Code execution will be limited to Python.")

genai.configure(api_key=GOOGLE_API_KEY)

# Set default parameters for the model
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8192,
}

MODEL_NAME = "gemini-2.0-flash"
try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini model ({MODEL_NAME}): {str(e)}")
    st.stop()

# --- Helper Functions ---
def generate_content_streaming(prompt_parts, chat_history_for_model):
    """Generates content using the Gemini model with streaming."""
    try:
        response_stream = model.generate_content(prompt_parts, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"Error generating content: {str(e)}"

def get_file_details(uploaded_file_obj):
    """Reads content from an uploaded file and returns details."""
    if uploaded_file_obj is None:
        return None
    try:
        content_bytes = uploaded_file_obj.getvalue()
        file_name = uploaded_file_obj.name
        file_type_short = file_name.split('.')[-1].upper() if '.' in file_name else "FILE"

        if uploaded_file_obj.type.startswith('text/') or \
           file_type_short in ['PY', 'JS', 'JAVA', 'C', 'CPP', 'GO', 'PHP', 'RB', 'SWIFT', 'KT', 'RS', 'TS', 'HTML', 'CSS', 'TXT', 'MD', 'JSON', 'YAML', 'H']:
            try:
                content_text = content_bytes.decode("utf-8")
                return {"name": file_name, "type": uploaded_file_obj.type, "content_text": content_text, "content_bytes": None, "display_type": file_type_short, "error": None}
            except UnicodeDecodeError:
                return {"name": file_name, "type": uploaded_file_obj.type, "content_text": None, "content_bytes": content_bytes, "display_type": file_type_short, "error": "UnicodeDecodeError"}
        elif uploaded_file_obj.type.startswith('image/'):
            return {"name": file_name, "type": uploaded_file_obj.type, "content_text": None, "content_bytes": content_bytes, "display_type": "IMG", "error": None}
        elif uploaded_file_obj.type == "application/pdf":
            return {"name": file_name, "type": uploaded_file_obj.type, "content_text": "[PDF Content - Bytes will be sent to AI if applicable]", "content_bytes": content_bytes, "display_type": "PDF", "error": None}
        else:
            return {"name": file_name, "type": uploaded_file_obj.type, "content_text": f"[Binary File: {file_type_short} - Bytes will be sent to AI if applicable]", "content_bytes": content_bytes, "display_type": file_type_short, "error": None}
    except Exception as e:
        st.error(f"Error reading file {uploaded_file_obj.name}: {str(e)}")
        return {"name": uploaded_file_obj.name, "type": "unknown", "content_text": None, "content_bytes": None, "display_type": "ERR", "error": str(e)}

def execute_python_code(code_to_execute):
    """Executes Python code in a restricted environment and captures output."""
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        restricted_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "int": int, "str": str, "float": float,
                "list": list, "dict": dict, "set": set, "tuple": tuple, "abs": abs, "max": max,
                "min": min, "sum": sum, "sorted": sorted, "zip": zip, "enumerate": enumerate,
                "True": True, "False": False, "None": None
            },
        }
        exec(code_to_execute, restricted_globals)
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully (no output)."
    except Exception as e:
        return f"Execution error: {str(e)}"
    finally:
        sys.stdout = old_stdout

def execute_with_judge0(code: str, language_id: int):
    """Executes code using Judge0 API."""
    if not JUDGE0_API_KEY:
        return "Judge0 API key not configured. Cannot execute non-Python code."
    
    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": JUDGE0_API_KEY,
        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com"
    }
    
    payload = {
        "source_code": code,
        "language_id": language_id,
        "stdin": "",
        "redirect_stderr_to_stdout": True
    }
    
    try:
        # Create submission
        response = requests.post(
            f"{JUDGE0_ENDPOINT}/submissions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        submission_id = response.json()["token"]
        
        # Get result (with retry logic)
        result = None
        for _ in range(5):
            time.sleep(1)  # Wait for execution
            result_response = requests.get(
                f"{JUDGE0_ENDPOINT}/submissions/{submission_id}",
                headers=headers
            )
            result = result_response.json()
            
            if result["status"]["id"] not in [1, 2]:  # 1: in queue, 2: processing
                break
        
        if result["status"]["id"] == 3:  # Accepted
            output = result.get("stdout", "")
            if not output:
                output = result.get("compile_output", "No output")
            return output
        else:
            error_msg = result.get("message", "Unknown error")
            stderr = result.get("stderr", "")
            compile_output = result.get("compile_output", "")
            return f"Execution error ({result['status']['description']}): {error_msg}\n{stderr}\n{compile_output}"
            
    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

# Language to Judge0 ID mapping
LANGUAGE_TO_JUDGE0_ID = {
    "python": 71,
    "javascript": 63,
    "java": 62,
    "c++": 54,
    "c#": 51,
    "ruby": 72,
    "go": 60,
    "swift": 83,
    "kotlin": 78,
    "php": 68,
    "rust": 73,
    "typescript": 74,
    "sql": 82
}

# Streamlit app configuration
st.set_page_config(
    layout="wide",
    page_title="AI Multi-Tool Assistant",
    page_icon="✨",
    initial_sidebar_state="collapsed"
)

# App title and caption
st.title("✨ SMART CODER: AI POWERED PROGRAMMING ASSISTANT")
st.caption("Your intelligent partner for text, code, and image-based queries!")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'staged_files' not in st.session_state:
    st.session_state.staged_files = []
if 'run_id_counter' not in st.session_state:
    st.session_state.run_id_counter = 0
if 'code_execution_output' not in st.session_state:
    st.session_state.code_execution_output = {}
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "Auto-detect"

# --- Chat Display Area ---
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        role = "user" if message.get("is_user") else "assistant"
        with st.chat_message(role):
            if message.get("prompt"):
                st.markdown(message["prompt"])
                if message.get("language_context"):
                    st.caption(f"Language context: {message['language_context']}")

            if message.get("attached_files_info"):
                st.markdown("**Attachments:**")
                for file_info in message["attached_files_info"]:
                    file_display_name = file_info.get("name", "Unnamed File")
                    file_display_type = file_info.get("display_type", "FILE")
                    st.markdown(
                        f"""
                        <div style="display: inline-block; border: 1px solid #ddd; background-color: #f0f0f0; border-radius: 10px; padding: 3px 8px; margin: 2px; font-size: 0.9em;">
                            📄 {file_display_name} <span style="color: #555; font-size: 0.8em;">({file_display_type})</span>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    if file_info.get("content_text_preview"):
                        with st.expander(f"View snippet of {file_display_name}"):
                            st.text(file_info["content_text_preview"])
                st.markdown("---")

            if message.get("error"):
                st.error(message["error"])
            else:
                if message.get("code_response"):
                    st.markdown("#### Generated Code:")
                    st.code(message["code_response"], language=message.get("language", "plaintext"))
                    
                    # Code execution button (now supports multiple languages via Judge0)
                    if message.get("language") and message["code_response"]:
                        exec_key = f"exec_{i}"
                        if st.button(f"Execute {message['language'].capitalize()} Code", key=exec_key):
                            st.session_state.code_execution_output[i] = "Executing..."
                            st.rerun()

                        if st.session_state.code_execution_output.get(i) == "Executing...":
                            with st.spinner("Executing code..."):
                                language = message["language"].lower()
                                if language == "python":
                                    execution_result = execute_python_code(message["code_response"])
                                elif language in LANGUAGE_TO_JUDGE0_ID and JUDGE0_API_KEY:
                                    execution_result = execute_with_judge0(
                                        message["code_response"],
                                        LANGUAGE_TO_JUDGE0_ID[language]
                                    )
                                else:
                                    execution_result = f"Cannot execute {language} code. {'Judge0 API key missing.' if not JUDGE0_API_KEY else 'Language not supported.'}"
                                
                                st.session_state.code_execution_output[i] = execution_result
                            st.rerun()

                        if isinstance(st.session_state.code_execution_output.get(i), str) and \
                           st.session_state.code_execution_output.get(i) != "Executing...":
                            st.markdown("#### Execution Output:")
                            st.text_area("Output", value=st.session_state.code_execution_output[i], height=100, disabled=True, key=f"out_{i}")
                
                if message.get("explanation_response"):
                    title = "#### Code Explanation:" if message.get("code_response") else "#### Explanation:"
                    st.markdown(title)
                    st.markdown(message["explanation_response"])
                
                elif not message.get("code_response") and not message.get("explanation_response") and message.get("response_content"):
                    st.markdown(message["response_content"])

st.markdown("---")

# --- Language Selection and File Upload Area ---
col1, col2 = st.columns([1, 3])

with col1:
    languages = ["Auto-detect", "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "Go", "Swift", "Kotlin", "PHP", "HTML", "CSS", "SQL", "TypeScript", "Rust", "Scala", "Markdown"]
    current_selected_lang_val = st.session_state.selected_language
    if current_selected_lang_val not in languages:
        current_selected_lang_val = "Auto-detect"
        st.session_state.selected_language = "Auto-detect"

    st.session_state.selected_language = st.selectbox(
        "Select Language Context:",
        options=languages,
        index=languages.index(current_selected_lang_val),
        key=f"language_select_{st.session_state.run_id_counter}"
    )

with col2:
    newly_uploaded_files = st.file_uploader(
        "📎 Attach files (text, code, images):",
        accept_multiple_files=True,
        type=["txt", "pdf", "py", "js", "java", "cpp", "c", "h", "cs", "go", "php", "rb", "swift", "kt", "rs", "ts", "html", "css", "sql", "md", "png", "jpg", "jpeg", "gif", "webp"],
        key=f"file_uploader_{st.session_state.run_id_counter}"
    )

if newly_uploaded_files:
    files_actually_added_to_stage = False
    for new_file in newly_uploaded_files:
        if not any(f.name == new_file.name and f.size == new_file.size for f in st.session_state.staged_files):
            st.session_state.staged_files.append(new_file)
            files_actually_added_to_stage = True
    if files_actually_added_to_stage:
        st.rerun()

if st.session_state.staged_files:
    st.markdown("**Staged files for next message:**")
    num_files = len(st.session_state.staged_files)
    cols_per_row = 4
    
    temp_staged_files = list(st.session_state.staged_files)

    for i in range(0, num_files, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            file_index = i + j
            if file_index < num_files:
                file_obj = temp_staged_files[file_index]
                with cols[j]:
                    file_ext = file_obj.name.split('.')[-1].upper() if '.' in file_obj.name else "FILE"
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ccc; background-color: #f9f9f9; border-radius: 8px; padding: 8px; margin-bottom: 5px; font-size: 0.9em; position: relative;">
                            📄 {file_obj.name[:25]}{'...' if len(file_obj.name)>25 else ''} 
                            <span style="color: #666; font-size: 0.8em;">({file_ext})</span>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    if st.button(f"Remove", key=f"remove_staged_{file_obj.file_id}_{file_index}"):
                        st.session_state.staged_files = [
                            f for f in st.session_state.staged_files 
                            if f.file_id != file_obj.file_id
                        ]
                        st.rerun()
    st.markdown("---")

# --- User Input Area ---
prompt_placeholder = "Ask about files, request code, or type your message..."
user_prompt = st.chat_input(prompt_placeholder, key=f"chat_input_{st.session_state.run_id_counter}")

# Sidebar for "Clear Chat" button
with st.sidebar:
    st.markdown("## Actions")
    if st.button("Clear Chat History & Staged Files", type="primary", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.staged_files = []
        st.session_state.code_execution_output = {}
        st.session_state.selected_language = "Auto-detect"
        st.session_state.run_id_counter += 1
        st.rerun()

if user_prompt:
    current_language_selection = st.session_state.selected_language
    user_message_entry = {
        "is_user": True,
        "prompt": user_prompt,
        "attached_files_info": [],
        "language_context": current_language_selection if current_language_selection != "Auto-detect" else "Auto-detected by AI"
    }
    
    processed_files_for_gemini = []
    
    if st.session_state.staged_files:
        for file_obj in st.session_state.staged_files:
            details = get_file_details(file_obj)
            if details:
                user_message_entry["attached_files_info"].append({
                    "name": details["name"],
                    "display_type": details["display_type"],
                    "content_text_preview": (details["content_text"][:200] + "..." if details["content_text"] and len(details["content_text"]) > 200 else details["content_text"]) if details.get("content_text") else None
                })
                
                if details["content_text"] and not details["content_bytes"]:
                    processed_files_for_gemini.append(f"--- START FILE: {details['name']} (Type: {details['display_type']}) ---")
                    processed_files_for_gemini.append(details["content_text"])
                    processed_files_for_gemini.append(f"--- END FILE: {details['name']} ---")
                elif details["content_bytes"]:
                    if details["type"].startswith("image/"):
                         processed_files_for_gemini.append({"mime_type": details["type"], "data": details["content_bytes"]})
                    elif details["type"] == "application/pdf" and details["content_bytes"]:
                        processed_files_for_gemini.append({"mime_type": details["type"], "data": details["content_bytes"]})
                        processed_files_for_gemini.append(f"[User has attached a PDF: {details['name']}. Process its content based on the user's query.]")
                    else:
                        processed_files_for_gemini.append(f"[User has attached a binary file: {details['name']} of type {details['type']}. Please respond based on user's query about it.]")

    st.session_state.chat_history.append(user_message_entry)
    
    gemini_prompt_parts = []
    code_instruction = (
        "If the request requires generating code, please provide the code block first, "
        "enclosed in triple backticks (e.g., ```python ... ```). "
        "Immediately after the code block, include the exact separator '---EXPLANATION---'. "
        "Following this separator, provide a detailed explanation of the code. "
        "If the request does not require code, respond naturally."
    )
    gemini_prompt_parts.append(code_instruction)

    prompt_prefix = ""
    if current_language_selection != "Auto-detect":
        prompt_prefix = f"The user's preferred language context is '{current_language_selection}'. "
    
    if processed_files_for_gemini:
        gemini_prompt_parts.extend(processed_files_for_gemini)
        gemini_prompt_parts.append(f"\n{prompt_prefix}Based on the preceding instruction, file(s) (if any), and the following user request, generate a response: {user_prompt}")
    else:
        gemini_prompt_parts.append(prompt_prefix + "Based on the preceding instruction and the following user request, generate a response: " + user_prompt)

    assistant_response_entry = {
        "is_user": False,
        "response_content": "",
        "code_response": "",
        "explanation_response": "",
        "language": current_language_selection.lower() if current_language_selection != "Auto-detect" else "plaintext",
        "error": ""
    }
    st.session_state.chat_history.append(assistant_response_entry)
    current_assistant_message_index = len(st.session_state.chat_history) - 1

    with st.spinner("AI is thinking..."):
        try:
            full_raw_response = ""
            temp_response_display_area = chat_container.empty()

            for chunk_text in generate_content_streaming(gemini_prompt_parts, st.session_state.chat_history[:-2]):
                full_raw_response += chunk_text
                with temp_response_display_area.container():
                     with st.chat_message("assistant"):
                        st.markdown(full_raw_response + "▌")

            temp_response_display_area.empty()
            st.session_state.chat_history[current_assistant_message_index]["response_content"] = full_raw_response

            code_part = ""
            explanation_part = ""
            detected_language = assistant_response_entry["language"]

            if "---EXPLANATION---" in full_raw_response:
                parts = full_raw_response.split("---EXPLANATION---", 1)
                code_part = parts[0].strip()
                explanation_part = parts[1].strip() if len(parts) > 1 else ""
                
                match = re.search(r"```(\w*)\n([\s\S]*?)```", code_part)
                if match:
                    lang_tag = match.group(1).lower()
                    actual_code = match.group(2).strip()
                    if lang_tag:
                        detected_language = lang_tag
                    code_part = actual_code
                else:
                    code_part = code_part
                    if current_language_selection != "Auto-detect":
                         detected_language = current_language_selection.lower()

                st.session_state.chat_history[current_assistant_message_index]["code_response"] = code_part
                st.session_state.chat_history[current_assistant_message_index]["explanation_response"] = explanation_part
                st.session_state.chat_history[current_assistant_message_index]["language"] = detected_language

            else:
                match = re.search(r"```(\w*)\n([\s\S]*?)```", full_raw_response)
                if match:
                    lang_tag = match.group(1).lower()
                    actual_code = match.group(2).strip()
                    if lang_tag:
                        detected_language = lang_tag
                    code_part = actual_code
                    st.session_state.chat_history[current_assistant_message_index]["code_response"] = code_part
                    st.session_state.chat_history[current_assistant_message_index]["language"] = detected_language
                    remaining_response = full_raw_response[match.end():].strip()
                    if remaining_response:
                        st.session_state.chat_history[current_assistant_message_index]["explanation_response"] = remaining_response
                else:
                    st.session_state.chat_history[current_assistant_message_index]["explanation_response"] = full_raw_response
                    if current_language_selection != "Auto-detect" and \
                       any(kw in full_raw_response.lower() for kw in ["def ", "class ", "function(", "import ", "public class"]):
                        st.session_state.chat_history[current_assistant_message_index]["language"] = current_language_selection.lower()

        except Exception as e:
            st.session_state.chat_history[current_assistant_message_index]["error"] = f"An error occurred: {str(e)}"

    st.session_state.staged_files = []
    st.session_state.run_id_counter += 1
    st.rerun()