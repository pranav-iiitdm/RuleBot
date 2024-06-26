# RuleBot: Formula 1 and Cricket Regulations Assistant

RuleBot is an intelligent chatbot application designed to answer questions about Formula 1 and Cricket regulations. It uses Retrieval-Augmented Generation (RAG) to provide accurate and context-aware responses based on official sporting documents.

## Features

- Interactive chat interface for querying regulations
- Support for Formula 1 and Cricket (under development)
- Utilizes official PDF documents and web sources for up-to-date information
- Employs advanced natural language processing for accurate responses
- Conversation history tracking

## Technology Stack

- Python
- Streamlit for the web interface
- LangChain for document processing and retrieval
- HuggingFace's BGE embeddings for semantic search
- FAISS for efficient vector storage and retrieval
- Beautiful Soup for web scraping
- Groq's LLM for natural language generation

## Installation

1. Clone the repository
2. Install the required dependencies
3. Set up your Groq API key

## Usage

Run the Streamlit app:
streamlit run app.py

Navigate to the provided local URL in your web browser to interact with RuleBot.

## Data Sources

The application uses a combination of PDF documents and web pages as data sources:

- Official FIA Formula 1 regulations (Sporting, Technical, and Financial)
- ICC Cricket regulations
- Additional web resources for beginner's guides and explanations

For a full list of data sources, refer to the `data.py` file.

## Contributing

Contributions to improve RuleBot are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## Disclaimer

This project is not officially affiliated with, authorized, endorsed by, or in any way connected to Formula 1, FIA, ICC, or any associated entities. It is an independent project created for educational and informational purposes.
