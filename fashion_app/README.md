# 👗 Fashion Magazine Generator — Setup Guide

A Flask web application that uses LLM agents, AI image generation (via OpenRouter), CLIP scoring, and BERT-based retrieval to automatically generate editorial fashion magazine pages.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Set Up Your API Key](#step-1--set-up-your-api-key)
4. [Step 2 — Create a Virtual Environment](#step-2--create-a-virtual-environment)
5. [Step 3 — Install Dependencies](#step-3--install-dependencies)
6. [Step 4 — Run the Application](#step-4--run-the-application)
7. [How to Use the App](#how-to-use-the-app)
8. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
fashion_app/
│
├── .env                          ← Your API key goes here (you create this)
├── requirements.txt              ← Python dependencies
├── setup.sh                      ← Automated setup script (Mac/Linux)
├── setup.ps1                     ← Automated setup script (Windows)
│
├── backend/                      ← All Python source code lives here
│   ├── main.py                   ← Entry point — run this to start the app
│   ├── pipeline.py               ← Core generation pipeline (3 LLM agents)
│   ├── prompts.py                ← Prompt templates for each agent
│   ├── bert_utils.py             ← BERT-based fashion data retrieval
│   ├── dataset_preprocessing_bert.py  ← How the BERT embeddings were generated. Dont need to run this.
│   ├── clip_utils.py             ← CLIP image-text alignment scoring
│   ├── html_pipeline.py          ← HTML generation and validation agent       
│
├── Data/                         ← Dataset and pre-computed embeddings
│   ├── fashion_combined.csv      ← Combined fashion dataset
│   ├── bert_embeddings.npy       ← Pre-computed BERT embeddings (provided)
│   ├── bert_descriptions.pkl     ← Corresponding text descriptions (provided)
│   ├── myntra_products_catalog.csv
│   └── fashion_data_2018_2022.xls
|
├── templates/                    ← Flask HTML templates (frontend UI)
│   ├── fashion_magazine.html     ← Main UI: article form and progress stream
│   ├── article.html              ← Viewer: displays a generated article page
│   └── error.html                ← Error page (e.g. article file not found)
│
└── output/                       ← Generated articles and images (auto-created)
    ├── article_*.html
    └── image_*.png
```

> **Note:** The `output/` folder and generation history are created automatically the first time you run the app. You do not need to create them manually.

---

## Prerequisites

Before you begin, you need **Python 3.12 or newer** installed on your computer.

### Check your Python version

**Mac:**
```bash
python3 --version
```

**Windows:**
```bash
python --version
```

You should see something like `Python 3.12.x` or `Python 3.13.x`. If you see a lower version (e.g. `3.9`, `3.10`), follow the instructions below.

---

### Installing Python 3.12+ (if needed)

**Mac:**

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click **Download Python 3.12.x** (or newer)
3. Open the downloaded `.pkg` file and follow the installer
4. Once installed, verify with `python3 --version`

**Windows:**

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click **Download Python 3.12.x** (or newer)
3. Run the installer — **important:** tick the box that says **"Add Python to PATH"** before clicking Install
4. Once installed, open a new Command Prompt and verify with `python --version`

---

## Step 1 — Set Up Your API Key

The application uses [OpenRouter](https://openrouter.ai) for both text and image generation, so it needs an OpenRouter API key (it starts with `sk-or-`).

**Easiest:** start the app (Step 4), paste your key into the **OpenRouter API Key** field at the top of the sidebar and click **Save Key**. The key is hidden while you type, checked with OpenRouter, and kept in memory only — you enter it again after restarting the app.

**Alternatively**, to avoid re-entering it, store it in a `.env` file:

1. Navigate to the `fashion_app/` folder
2. Create a new file called **`.env`** (note the dot at the start — this is intentional)
3. Open it in any text editor and add the following line:

```
OPENROUTER_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with the actual key provided to you. The file should look exactly like this — no quotes, no spaces around the `=`:

```
OPENROUTER_API_KEY=sk-or-v1-abc123...
```

> **Important:** Never share this file or commit it to Git. It contains your private credentials.

---

## Step 2 — Create a Virtual Environment

A virtual environment keeps the app's dependencies isolated from the rest of your computer. Think of it as a clean, dedicated box for this project.

Open a terminal (Mac) or Command Prompt (Windows) and navigate to the `fashion_app/` folder first.

**Mac:**
```bash
cd path/to/fashion_app
python3 -m venv venv
```

**Windows:**
```bash
cd path\to\fashion_app
python -m venv venv
```

This creates a folder called `venv/` inside `fashion_app/`. You only need to do this once.

---

### Activate the virtual environment

You need to activate the environment every time you open a new terminal to work on this project.

**Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

Once activated, you will see `(venv)` at the start of your terminal prompt — this confirms the environment is active.

To deactivate it later (when you're done), simply type:
```bash
deactivate
```

---

## Step 3 — Install Dependencies

With the virtual environment **active**, install all required packages:

**Mac:**
```bash
pip install -r requirements.txt
```

**Windows:**
```bash
pip install -r requirements.txt
```

This will download and install everything the app needs (Flask, OpenAI, CLIP, sentence-transformers, etc.). It may take a few minutes the first time.

> **Note:** Make sure you see `(venv)` in your prompt before running this command. If you don't, go back and activate the virtual environment first.

---

## Step 4 — Run the Application

> **Important:** The application must be launched from the `fashion_app/backend/` folder, not from `fashion_app/` directly.

### Navigate to the backend folder

**Mac:**
```bash
cd path/to/fashion_app/backend
```

**Windows:**
```bash
cd path\to\fashion_app\backend
```

### Start the server

**Mac:**
```bash
python3 main.py
```

**Windows:**
```bash
python main.py
```

You should see output like this in your terminal:
```
* Running on http://127.0.0.1:5001
* Debug mode: on
```

### Open the app in your browser

Open your browser and go to:

```
http://127.0.0.1:5001
```

The application is now running. Keep the terminal open — closing it will stop the server.

---

## How to Use the App

Once the app is open in your browser:

1. **Article Title** — Enter the title of your fashion magazine article (e.g. *"Spring Florals at ETH Campus"*)
2. **Image Styles** — Describe the visual style of each image you want generated (e.g. *"students in floral dresses, golden hour, blurred background"*). You can add multiple image descriptions.
3. **Article Description** — Write a brief description of what the article should be about. The system will use this to retrieve relevant fashion data and generate the article text.
4. Click **Generate** and watch the progress stream in real time.
5. Once complete, you can review the result and provide feedback to refine the text or images.

### What happens under the hood

| Step | What the app does |
|------|-------------------|
| Data retrieval | BERT embeddings find the 5 most relevant fashion records from the dataset to ground the article |
| Image generation | An image model on OpenRouter (`openai/gpt-image-1` by default) generates each image; CLIP scores how well it matches your description |
| Article writing | An LLM writes the article using the retrieved fashion data |
| HTML assembly | A third LLM agent assembles everything into a styled magazine page |
| Validation | The HTML is validated and corrected automatically (up to 3 attempts) |

Generated articles and images are saved to `fashion_app/output/`.

---

## Troubleshooting

**`python3: command not found` (Mac)**
Python is not installed or not on your PATH. Follow the installation steps in the Prerequisites section.

**`python: command not found` (Windows)**
Python was not added to PATH during installation. Re-run the Python installer and make sure to tick "Add Python to PATH".

**`ModuleNotFoundError: No module named 'flask'` (or similar)**
You ran `python3 main.py` without activating the virtual environment. Run `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows) first, then try again.

**"No OpenRouter API key set" or authentication errors**
Enter your key in the sidebar and click **Save Key**. If you use a `.env` file instead, check that it is inside `fashion_app/` (not inside `backend/`), that the filename is exactly `.env`, and that there are no extra spaces or quotes around the key.

**Port already in use (`Address already in use`)**
Another process is using port 5001. Either stop the other process, or restart your computer and try again.

**The page loads but generation fails or times out**
Check the terminal for error messages. Common causes are an invalid API key, no internet connection, or no credits left on the OpenRouter account.

**Images are generated but the article says "no relevant data found"**
The BERT embeddings may not have loaded correctly. Make sure `fashion_app/Data/bert_embeddings.npy` and `fashion_app/Data/bert_descriptions.pkl` are present in the `Data/` folder.