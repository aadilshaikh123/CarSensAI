import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_wikipedia_page(topic):
    search_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(topic.replace(' ', '_'))}"
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    try:
        response = session.get(search_url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {topic}: {e}")
        return None

def parse_wikipedia_page(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract all text content
    content_sections = []
    for section in soup.select('h2, h3, h4, h5, h6, p, ul, ol'):
        text = section.text.strip().replace('\n', ' ')
        content_sections.append(text)
    
    return {'full_text': ' '.join(content_sections)}

def scrape_mechanical_engineering_topics(topics):
    scraped_data = {}
    for topic in topics:
        print(f"Scraping {topic}...")
        html_content = get_wikipedia_page(topic)
        if html_content:
            data = parse_wikipedia_page(html_content)
            scraped_data[topic] = data
    return scraped_data

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    mechanical_topics = [
        "Mechanical engineering", "Fluid mechanics", "Thermodynamics", "Heat transfer", "Machine design", "Automobile engineering",
        "Kinematics", "Dynamics", "Vibration analysis", "Control systems", "Robotics", "Finite element analysis", "CAD/CAM",
        "Mechatronics", "Tribology", "Manufacturing engineering", "Aerodynamics", "Combustion", "Internal combustion engines",
        "Hydraulics", "Pneumatics", "Structural analysis", "Material science", "Fracture mechanics", "Welding technology",
        "Turbomachinery", "HVAC", "Renewable energy", "Acoustics", "Nanotechnology in mechanical engineering"
    ]
    
    scraped_data = scrape_mechanical_engineering_topics(mechanical_topics)
    save_to_json(scraped_data, "mechanical_engineering_wikipedia.json")
    
    print("Scraping completed. Data saved to mechanical_engineering_wikipedia.json")
