# IMDb Movie Recommendation Chatbot

An intelligent movie recommendation system powered by RAG (Retrieval-Augmented Generation) and AI agents. This chatbot provides personalized movie recommendations, detailed information, and interactive features using natural language processing and semantic search.

## Author
**Jesus Gerardo Galaz Montoya**
Email: jgalaz@gmail.com

## Overview

This project implements two chatbot systems:

1. **RAG-Based Chatbot**: Uses FAISS vector similarity search with OpenAI embeddings to provide context-aware movie recommendations
2. **Agent-Based Chatbot**: Employs specialized AI agents with distinct tools for searching movies, retrieving details, providing summaries, showing trending films, and generating movie trivia

## Features

- **Natural Language Queries**: Ask questions in plain English (e.g., "Show me the best sci-fi movies about time travel")
- **Semantic Search**: Vector embeddings enable intelligent matching beyond keyword search
- **Multiple Tools**:
  - Search Movies: Find films by genre, theme, or description
  - Get Movie Details: Retrieve director, cast, year, and ratings
  - Trending Movies: View top-rated films in the database
  - Summarize Recommendations: AI-generated explanations for movie suggestions
  - Movie Quiz: Interactive trivia questions
- **Interactive Gradio UI**: User-friendly web interface for chatting with the bot
- **Conversation History**: All interactions are logged for persistence

## Dataset

- **Source**: IMDb movie dataset (3,173 records, deduplicated to 2,762)
- **Fields**: Title, Genre, Director, Cast, Year, Rating, MetaScore, Duration, Certificate
- **Preprocessing**: Data cleaning, deduplication, cast verification, and format standardization

## Project Structure

```
case_study_2_imdb_ml/
├── ik_case_study_2_imdb_ML_chatbot.ipynb  # Main notebook
├── IMDb_Dataset.csv                        # Movie dataset
├── environment.yml                         # Conda environment specification
├── requirements.txt                        # Pip dependencies
├── movie_bot/                              # Chat history storage
│   └── chat_history.json                   # Conversation logs
└── README.md                               # This file
```

## Technologies Used

- **Python 3.11+**
- **LangChain**: Agent framework and retrieval chains
- **OpenAI API**: GPT-4.1 for responses, text-embedding-3 for vectors
- **FAISS**: Vector similarity search
- **Gradio**: Web UI for chatbot interaction
- **Pandas**: Data manipulation and analysis
- **NLTK**: Text processing and tokenization
- **Matplotlib/Seaborn**: Data visualization

## Setup Instructions

### 1. Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Conda or virtualenv for environment management

### 2. Environment Setup

**Option A: Using Conda (Recommended)**

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate ml
```

**Option B: Using pip**

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option C: Manual Installation**

```bash
# Create conda environment
conda create -n ml python=3.11
conda activate ml

# Install dependencies
pip install langchain langchain-openai langchain-community
pip install openai faiss-cpu pandas numpy
pip install gradio matplotlib seaborn nltk inflect
```

### 3. API Key Configuration

Create a file `~/.openai_env` with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-proj-your-key-here
```

For VSCode, create `.vscode/settings.json` in the project directory:

```json
{
  "python.envFile": "/Users/yourusername/.openai_env",
  "jupyter.envFile": "/Users/yourusername/.openai_env"
}
```

### 4. Update File Paths

In the notebook, update these paths to match your system:

- `MOVIE_BOT_DIR`: Directory for chat history logs
- `file_path`: Path to IMDb_Dataset.csv

### 5. Run the Notebook

```bash
# Open in Jupyter
jupyter notebook ik_case_study_2_imdb_ML_chatbot.ipynb

# Or open in VSCode with Jupyter extension
code .
```

Run all cells sequentially. The Gradio interface will launch at the end.

## Usage

### RAG Chatbot (First Interface)

The first Gradio UI provides general movie search using semantic similarity:

```
Query: "Recommend the best esoteric, mysterious, magical movies"
Response: [List of relevant movies with context]
```

### Agent-Based Chatbot (Second Interface)

Select a tool from the dropdown and enter your query:

**Search Movies**
```
Query: "Top 5 vampire movies"
Response:
1. Interview with the Vampire (Genre: Drama, Director: Neil Jordan)
2. Nosferatu (Genre: Fantasy, Director: Robert Eggers)
...
```

**Get Movie Details**
```
Query: "The Matrix"
Response:
Title: The Matrix
Year: 1999
Director: Lana Wachowski, Lilly Wachowski
Stars: Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss
Rating: 8.7
```

**Trending Movies**
```
Query: (any text)
Response: Top 5 highest-rated movies in database
```

**Movie Quiz**
```
Query: (any text)
Response: Multiple-choice trivia question with answer revealed
```

**Summarize Recommendations**
```
Query: "Why are these good sci-fi picks?"
Response: AI-generated 2-sentence overview
```

## Exploratory Data Analysis

The notebook includes comprehensive EDA:

- **Title Analysis**: Tokenization and keyword frequency analysis
- **Numerical Statistics**: IMDb ratings, MetaScore, duration, year distributions
- **Categorical Analysis**: Genre distribution, certificate ratings, top directors/actors
- **Data Quality**: Duplicate detection, missing value handling, cast verification

## Technical Implementation

### Data Pipeline

1. **Data Loading & Cleaning**: Remove duplicates, fix formatting issues
2. **Text Processing**: Tokenization, stopword removal, translation
3. **Chunking**: Break movie descriptions into 50-word chunks
4. **Embedding**: Generate OpenAI embeddings for semantic search
5. **Vector Store**: FAISS index for fast similarity retrieval

### RAG Chain

```
User Query → Embedding → FAISS Retrieval → Prompt Template → LLM → Response
```

### Agent System

```
User Query → Tool Selection → Tool Execution → Response Formatting → User
```

## Limitations & Future Work

### Current Limitations
- Dataset limited to 2,762 movies
- Some cast information marked as "not verified"
- Local execution only (not deployed)

### Future Improvements
- **Multimodal Input**: Accept images and text
- **Poster Display**: Show movie posters with recommendations
- **Expanded Dataset**: Include more recent films and streaming data
- **Performance Metrics**: Measure retrieval speed and accuracy
- **User Feedback**: Collect satisfaction ratings for continuous improvement
- **Deployment**: Host as web service with authentication

## Troubleshooting

### Kernel Hangs
- Ensure no `input()` calls are uncommented
- Set Gradio `share=False` for local use
- Restart kernel if stuck

### Import Errors
```bash
pip install --upgrade langchain langchain-openai
```

### API Rate Limits
- Reduce batch sizes for embeddings
- Add delays between API calls
- Use caching where possible

## License

This project is for educational purposes. Movie data belongs to IMDb.

## Acknowledgments

- IMDb for the movie dataset
- OpenAI for GPT and embedding models
- LangChain community for the agent framework
