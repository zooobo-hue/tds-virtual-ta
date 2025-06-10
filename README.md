# tds-project-1

fastapi==0.110.0
uvicorn==0.29.0
aiohttp==3.9.3
beautifulsoup4==4.12.3
pydantic==2.6.4
sqlite3
tiktoken==0.6.0
nltk==3.8.1
numpy==1.26.4
scikit-learn==1.4.1.post1

# TDS Virtual TA

A virtual Teaching Assistant API for IIT Madras' Online Degree in Data Science, "Tools in Data Science" course.

## Setup

1. Clone the repository:
   ```bash
2.# Install dependencies:
    pip install -r requirements.txt
3.# Run the scraper to populate the database
    python main.py
4. #Start the FastAPI server
   uvicorn main:app --host 0.0.0.0 --port 8000

   
