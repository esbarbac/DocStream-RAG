# ModiFace DocQA (Local Demo)

This project implements a **Document Question Answering agent** using OpenAI models and LangChain.  
Users can upload PDF or text documents and ask grounded questions based on their content.

---

## Features
- Upload and index multiple PDF or TXT documents.
- Retrieve answers from OpenAI models based on document context.
- Built with **Streamlit**, **LangChain**, and **FAISS**.
- Designed for **local use only**.

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/modiface-docqa.git
cd modiface-docqa
```

2. **Create and activate a Python environment**

```bash
conda create -n modiface-docqa python=3.11 -y
conda activate modiface-docqa
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set your OpenAI API key**

Create a file named `.env` in the project root (same directory as `app.py`) and add:

```
OPENAI_API_KEY=your_api_key_here
```

The application will automatically load this key when running.

---

## Run Locally

```bash
streamlit run app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

---
