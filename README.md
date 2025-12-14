# 🤖 SMART CODER: AI Powered Programming Assistant

> **Final Year Project (FYP)**  
> **Author:** Shadab Ali

**Smart Coder** is an intelligent programming assistant built with [Streamlit](https://streamlit.io/) and powered by [Google Gemini](https://ai.google.dev/). It helps developers with code generation, explanation, debugging, and even execution across multiple languages.

## ✨ Features

- **💬 AI Chat Interface**: Interactive chat with Gemini 2.0 Flash model.
- **📁 Multi-Modal File Support**: Upload and analyze text, code, images, and PDFs.
- **🛠️ Code Generation & Explanation**: Get code snippets with detailed explanations.
- **🚀 Code Execution**:
  - **Python**: Executed locally in a restricted environment.
  - **Multi-Language**: Execute C++, Java, JavaScript, Go, etc., using the [Judge0 API](https://judge0.com/).
- **👁️ File Preview**: View snippets of uploaded files directly in the chat.
- **📝 Syntax Highlighting**: Auto-detects languages for better readability.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)
- [Judge0 API Key](https://rapidapi.com/judge0-official/api/judge0-ce) (Optional, for non-Python code execution)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd SMART-CODER-AI-POWERED-PROGRAMMING-ASSISTANT
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the project root:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    JUDGE0_API_KEY=your_judge0_api_key_here  # Optional
    ```

### Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 📂 Project Structure

- `app.py`: Main application file containing the Streamlit UI and logic.
- `requirements.txt`: Python dependencies.
- `.env`: Environment variables configuration.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
Built with ❤️ using Streamlit and Google Gemini.
<br>
*Submitted as a Final Year Project by Shadab Ali*

## 🎓 Thesis Details

| Field | Detail |
| :--- | :--- |
| **Title** | SMART CODER: AI POWERED PROGRAMMING ASSISTANT |
| **Student** | Shadab Ali |
| **Degree** | Bachelor of Studies in Computer Science |
| **Session** | 2021-2025 |
| **Institution** | Department of Computer Science, The University of Swabi, Khyber Pakhtunkhwa, Pakistan |

> **Abstract:** This project aims to bridge the gap between novice and expert developers by providing an intelligent, context-aware programming assistant. Leveraging the power of Large Language Models (LLMs) like Gemini, "Smart Coder" assists in code generation, debugging, and execution, democratizing access to efficient software development tools.
