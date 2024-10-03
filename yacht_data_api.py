from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from urllib.parse import urlparse
import yacht_data_llm_scraper as scraper
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend origin
    allow_credentials=False,  # Change this to False
    allow_methods=["*"],
    allow_headers=["*"],
)

class YachtUrlInput(BaseModel):
    url: HttpUrl

@app.post("/extract-yacht-data/")
async def extract_yacht_data(input_data: YachtUrlInput):
    url = str(input_data.url)
    
    # Extract the domain from the URL
    parsed_url = urlparse(url)
    domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

    try:
        # Fetch the HTML content
        html_content = scraper.fetch_yacht_listing(url)

        if html_content is None:
            raise HTTPException(status_code=404, detail="Failed to fetch the yacht listing.")

        # Process the HTML content
        processed_content = scraper.process_html_content(html_content)

        # Check if the character count is less than 40,000
        char_count = len(processed_content)
        if char_count >= 40000:
            raise HTTPException(status_code=400, detail=f"Character count ({char_count}) exceeds 40,000. Data not processed.")

        # Extract yacht data using GPT-4
        yacht_data = scraper.extract_yacht_data(processed_content)

        return yacht_data.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Add a simple test endpoint
@app.get("/hello")
async def hello():
    return {"message": "Hello from Yacht Data Extractor backend!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)