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
git clone https://github.com/esbarbac/modiface-docqa.git
cd modiface-docqa
```

2. **Create and activate a Python environment**

If using conda:

```bash
conda create -n modiface-docqa python=3.11 -y
conda activate modiface-docqa
```
If using venv:

```bash
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install --upgrade pip
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set your OpenAI API key**

Go to the file named `.env` in the project root, and add your OpenAI API key:

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

## How to Use

1. **Upload Documents**

   * Use the file uploader to select one or more `.pdf` or `.txt` files.
   * As soon as you upload, the app automatically processes and indexes them for search.

2. **Ask a Question**

   * Enter a question (e.g., *"What is this paper about?"*) and click **Get Answer**.
   * The app retrieves relevant text chunks and uses the OpenAI model to generate a grounded response.

3. **View Sources**

   * Each answer includes expandable **Source** sections showing the retrieved document excerpts used to produce the answer.

4. **Clear Index**

   * Click **Clear Index** to remove all uploaded and indexed documents.
   * This resets the database.

---
