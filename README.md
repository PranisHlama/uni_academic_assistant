# Domain-Specific Assistant using RAG and LLM

## Project Title

Pranish Uni Assistant

---

## Project Description

This project is a domain-specific intelligent assistant built using Retrieval-Augmented Generation (RAG) and a Large Language Model (LLM). The assistant answers questions only related to university information.

The system retrieves relevant information from a JSON dataset, converts questions into embeddings, stores them in a FAISS vector database, retrieves the most relevant context for the user query, and sends that context to an LLM using Ollama.

The assistant is deployed using Streamlit.

---

## Features

- Domain-specific university assistant
- JSON-based knowledge base
- Sentence embedding generation
- FAISS vector database for similarity search
- Retrieval-Augmented Generation pipeline
- Local LLM integration using Ollama
- Streamlit chat interface
- Source/context display for retrieved answers
- Chat history sidebar

---

## Technologies Used

- Python
- Streamlit
- Ollama
- Mistral LLM
- FAISS
- Sentence Transformers
- JSON

---

# Project Structure

```text
university-rag-assistant/
│
├── app.py
├── uni_data.json
├── requirements.txt
└── README.md
```

---

# How to Run the project:
## 1. Install Ollama

Ollama is used to run the LLM locally.

### For Windows

Download and install Ollama from the official website:

```text
https://ollama.com/download/windows
```

Or run this command in PowerShell:

```bash
irm https://ollama.com/install.ps1 | iex
```

---

### For macOS

Download Ollama from:

```text
https://ollama.com/download
```

Then install it like a normal macOS application.

---

### For Linux

Run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 2. Pull the LLM Model

This project uses the `mistral` model.

After installing Ollama, open your terminal and run:

```bash
ollama pull mistral
```

You can test the model using:

```bash
ollama run mistral
```

---

# 3. Clone the Project Repository

Clone the GitHub repository using:

```bash
git clone <your-github-repository-link>
```

Example:

```bash
git clone https://github.com/yourusername/university-rag-assistant.git
```

Then move into the project directory:

```bash
cd university-rag-assistant
```

The repository should contain the following files:

```text
app.py
uni_data.json
requirements.txt
README.md
```

## 4. Prepare Dataset

There is  a dataset named uni_data.json, you can add additional questions to this json file if you want 
```text
uni_data.json
```

The dataset should contain domain-specific information only.

---

## 5. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 6. Install Requirements

Create a file named `requirements.txt`:

```text
streamlit
ollama
sentence-transformers
faiss-cpu
numpy
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Run Ollama

Make sure Ollama is running.

You can check by running:

```bash
ollama list
```

If `mistral` is not listed, run:

```bash
ollama pull mistral
```

---

## 8. Run the Streamlit App

Run the app using:

```bash
streamlit run app.py
```

Streamlit will start a local server and open the app in your browser.

Usually, the app opens at:

```text
http://localhost:8501
```

---

## How the RAG Pipeline Works

The system follows these steps:

1. Loads university data from `uni_data.json`
2. Extracts all question-answer pairs
3. Converts questions into embeddings using `all-MiniLM-L6-v2`
4. Stores embeddings in a FAISS vector database
5. Converts the user question into an embedding
6. Retrieves the most similar questions from FAISS
7. Sends retrieved context to the Mistral LLM through Ollama
8. Generates a short grounded answer
9. Displays the answer and source context in Streamlit

---

## Prompt Design

The LLM receives a prompt containing:

- System instruction
- Retrieved context
- User question
- Rule to answer only from context

Example:

```text
You are a university support assistant.

Rules:
- Answer using only the context.
- Keep the answer short and direct.
- If the answer is not in the context, say:
  "I don't have enough information in university data."

Context:
{retrieved_context}

Question:
{user_query}
```

This prevents the assistant from answering outside the university domain.

---

## Out-of-Domain Handling

If the retrieved similarity score is below the threshold, the assistant returns:

```text
I don't have enough information in university data.
```

This ensures the system does not generate unsupported answers.

---

## Example Questions

```text
What documents are required for admission?
```

```text
What are the library opening hours?
```

```text
What is the semester fee?
```

```text
Who won the football world cup?
```

The last question should return an out-of-domain response.

---

## Author

**Name:** Pranish  
**Project:** University Academic Assistant