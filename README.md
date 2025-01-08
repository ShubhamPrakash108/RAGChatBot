# RAGChatBot

## Overview
This repository contains the implementation of a Retrieval-Augmented Generation (RAG) based ChatBot. The code is built using Python and utilizes Streamlit for creating an interactive web application.

## Requirements
Before running the code, ensure that you have the following installed:
- Python 3.x
- Visual Studio Code (VS Code)
- Streamlit
- Other dependencies listed in `requirements.txt`

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/ShubhamPrakash108/RAGChatBot.git
```

### 2. Install Virtual Environment (Optional but Recommended)
Navigate to the project folder and create a virtual environment to avoid conflicts with global packages:
```bash
cd RAGChatBot
python -m venv venv
```

Activate the virtual environment:

On Windows:
```bash
.\venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
Install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project directory and add any necessary environment variables. (Ensure you have the required API keys or any other credentials that might be needed by the project.)

### 5. Run the Application
To run the application using Streamlit, use the following command:
```bash
streamlit run app.py
```

This will launch the web app in your default web browser. You can interact with the RAG-based ChatBot from there.

## Project Structure
The project consists of the following files and directories:
- `app.py`: Main Streamlit application file for running the web interface
- `pdfs/`: Directory containing PDFs that might be used in the chatbot
- `venv/`: Virtual environment folder (created during setup)
- `.env`: Environment variables for the project
- `requirements.txt`: List of Python packages required for the project
