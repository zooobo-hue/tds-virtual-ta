

# TDS Virtual TA

A virtual Teaching Assistant API for IIT Madras' Online Degree in Data Science, "Tools in Data Science" course.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/zooobo-hue/tds-virtual-ta.git
   cd tds-virtual-ta
2.# Install dependencies:
    pip install -r requirements.txt
3.# Run the scraper to populate the database
    python main.py
4. #Start the FastAPI server
   uvicorn main:app --host 0.0.0.0 --port 8000

   
