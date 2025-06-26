import os
import requests
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

def test_api():
    api_url = os.getenv('API_URL')
    if not api_url:
        logging.error("API_URL not found in .env file")
        return
        
    test_image = os.path.join('images', next((f for f in os.listdir('images') 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None))
    
    if not test_image or not os.path.exists(test_image):
        logging.error("No test image found in the images directory")
        return
        
    logging.info(f"Testing API with image: {test_image}")
    logging.info(f"API URL: {api_url}")
    
    try:
        with open(test_image, 'rb') as f:
            response = requests.post(api_url, files={'file': f}, timeout=30)
            response.raise_for_status()
            
            # Log the raw response
            logging.info("\n=== API Response ===")
            logging.info(f"Status Code: {response.status_code}")
            logging.info("Headers:")
            for header, value in response.headers.items():
                logging.info(f"  {header}: {value}")
                
            # Try to parse as JSON
            try:
                data = response.json()
                logging.info("\nResponse JSON:")
                logging.info(json.dumps(data, indent=2, default=str))
                
                # Log all top-level keys
                if isinstance(data, dict):
                    logging.info("\nTop-level keys in response:")
                    for key in data.keys():
                        logging.info(f"- {key}")
                        
            except json.JSONDecodeError:
                logging.error("Response is not valid JSON")
                logging.info("\nRaw response content:")
                logging.info(response.text)
                
    except Exception as e:
        logging.error(f"Error making API request: {str(e)}", exc_info=True)

if __name__ == '__main__':
    test_api()
