# Monolitich RAG System

A multimodal (Agentic) Retrieval-Augmented Generation (RAG) system with web search, scrape, image processing, speech (TTS & STT), and communication capabilities (currently outgoing phone calls and SMS).

## Features

- **Web Search & Scraping**: Search the web for up-to-date information and extract content from websites
- **Image Processing**: Generate, analyze, and process images using DALL-E and GPT-4 Vision
- **Speech Recognition & Synthesis**: Convert speech to text and text to speech using OpenAI's APIs
- **Communication**: Send SMS messages and make phone calls via Twilio integration
- **Vector Storage**: Store and retrieve information using vector embeddings
- **Reactive UI**: Modern React frontend with real-time tool activation indicators

## Project Structure

```
Monolithic/
├── src/                               # Python backend Core RAG system implementation
│   ├── core/                          # Core functionality components
│   │   ├── __init__.py
│   │   ├── vector_store.py            # Vector database implementation (Annoy)
│   │   └── workflow.py                # RAG workflow implementation with LangChain
│   │
│   ├── tools/                         # Tool implementations for various functionalities
│   │   ├── __init__.py                # Exports all tools
│   │   ├── image_tools.py             # Image analysis and generation with DALL-E and GPT-4 Vision
│   │   ├── rag_tools.py               # RAG-specific tools for LangChain
│   │   ├── speech_tools.py            # Speech-to-text and text-to-speech with OpenAI
│   │   ├── twilio.py                  # SMS and call integration with Twilio
│   │   ├── web_scraper.py             # Web scraping with Selenium
│   │   └── web_searcher.py            # Web search with Google CSE
│   │
│   ├── __init__.py
│   ├── demo.py                        # Demo script showcasing the RAG system capabilities
│   └── web_rag_system.py              # Main RAG system implementation with LLM integration
│
├── app.py                             # FastAPI application
├── requirements.txt                   # Python dependencies
└── setup.py                           # Package setup
```

## Installation

### Backend Setup

1. Clone the repository
   ```bash
   git clone https://github.com/FrK06/Monolithic.git
   cd Monolithic
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_google_custom_search_engine_id
   TWILIO_ACCOUNT_SID=your_twilio_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone_number
   ```
