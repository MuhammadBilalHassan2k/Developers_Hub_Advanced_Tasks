
# Developers Hub – Advanced AI Tasks

This repository contains solutions to three advanced machine learning and AI tasks. Each task explores a unique domain: from NLP classification to customer churn prediction and conversational AI using LangChain or RAG.



## Files Overview

### 1. `TASK_1(AG_News).ipynb` – AG News Text Classification  
A Hugging Face Transformers-based NLP classifier for AG News dataset with Gradio UI.

### 2. `TASK2_(Telco_Churn_Data_set.ipynb)` – Telco Customer Churn Prediction  
An end-to-end machine learning pipeline using Scikit-learn to predict customer churn.

### 3. `TASK_4_Context_Aware_Chatbot.ipynb` – Context-Aware Chatbot (LangChain or RAG)  
A conversational chatbot that uses retrieval-augmented generation and memory to answer questions based on custom documents.



## Task Details

### Task 1 – AG News Text Classification

Builds a news article classifier using:
- `transformers`, `datasets`, `scikit-learn`, `gradio`
- Text preprocessing, BERT fine-tuning
- Deployed with Gradio for real-time predictions



### Task 2 – Customer Churn Prediction

Machine learning pipeline for predicting customer churn:
- Data preprocessing with Pandas
- Models: Logistic Regression, Decision Tree, etc.
- Performance evaluation + model export via `joblib`



### Task 4 – Context-Aware Chatbot Using LangChain / RAG

Conversational AI system that:
- Embeds a document corpus (e.g., Wikipedia or internal docs)
- Uses **LangChain** or **RAG** to enable contextual Q&A
- Integrates context memory to maintain chat history
- Deployed using **Streamlit**

  **Skills Applied:**
- Document vectorization (FAISS, ChromaDB, etc.)
- Retrieval-Augmented Generation (RAG)
- LangChain integration
- Conversational memory handling
- LLM-backed responses



## Setup Instructions

### Environment Requirements

Install the following libraries depending on the task:

pip install transformers datasets scikit-learn gradio
pip install pandas joblib openpyxl
pip install langchain streamlit faiss-cpu chromadb



## How to Run

1. Open the desired `.ipynb` file in **Google Colab** or **Jupyter Notebook**
2. Follow each code cell step-by-step
3. For Task 4, run `streamlit run app.py` if the code is modularized into a script



## Contact

Developed by **Muhammad Bilal Hassan**
If you have any questions or feedback, feel free to reach out via GitHub.
