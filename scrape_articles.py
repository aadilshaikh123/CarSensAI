import requests
from bs4 import BeautifulSoup
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BASE_URL = 'https://www.2carpros.com/questions'
MAX_QUESTIONS = 500000000
OUTPUT_FILE = 'car_questions_live.json'

# Locks for thread-safe file writing and question count update
file_lock = threading.Lock()
count_lock = threading.Lock()
question_count = 0  # Global counter

def get_total_pages():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.content, 'html.parser')
    pagination_links = soup.select('a[href*="?page="]')
    page_numbers = [int(link.get_text()) for link in pagination_links if link.get_text().isdigit()]
    return max(page_numbers) if page_numbers else 1

def get_question_links(page_num):
    url = f'{BASE_URL}?page={page_num}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = {
        'https://www.2carpros.com' + a['href']
        for a in soup.select('a[href^="/questions/"]')
        if a['href'].startswith('/questions/') and not a['href'].endswith('/images')
    }
    return list(links)

def scrape_question(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract question title
        h1_tag = soup.find('h1')
        title = h1_tag.get_text(strip=True) if h1_tag else 'No title available'
        
        # Extract question content
        question_div = soup.find('div', class_='text')
        question_text = question_div.get_text(strip=True) if question_div else 'No question content available'
        
        # Extract answer (if available)
        answer_div = soup.find('div', class_='reply staff')
        answer_text = ''
        if answer_div:
            answer_text_div = answer_div.find('div', class_='text')
            answer_text = answer_text_div.get_text(strip=True) if answer_text_div else 'No answer available'

        # Format the scraped data for LLM training
        formatted_entry = {
            "instruction": f"Q: {title}\n{question_text}",
            "input": "",
            "output": f"A: {answer_text}"
        }
        return formatted_entry
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def write_to_json(entry):
    with file_lock:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write(",\n")

def main():
    global question_count
    total_pages = get_total_pages()
    print(f'Total pages: {total_pages}')
    
    # Initialize the JSON file with an opening bracket for a JSON array
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    # Outer thread pool: scrape pages concurrently (16 at a time)
    with ThreadPoolExecutor(max_workers=16) as page_executor:
        page_futures = [page_executor.submit(get_question_links, page_num) for page_num in range(1, total_pages + 1)]
        
        # Process each page as soon as it's done
        for page_future in as_completed(page_futures):
            with count_lock:
                if question_count >= MAX_QUESTIONS:
                    break
            page_links = page_future.result()
            
            # Inner thread pool: scrape questions from this page concurrently (16 threads)
            with ThreadPoolExecutor(max_workers=16) as question_executor:
                # Map each question URL to a future
                question_futures = {question_executor.submit(scrape_question, link): link for link in page_links}
                for q_future in as_completed(question_futures):
                    with count_lock:
                        if question_count >= MAX_QUESTIONS:
                            break
                    entry = q_future.result()
                    if entry:
                        write_to_json(entry)
                        with count_lock:
                            question_count += 1
                            current_count = question_count
                        print(f"Scraped {current_count}/{MAX_QUESTIONS}: {question_futures[q_future]}")
                    with count_lock:
                        if question_count >= MAX_QUESTIONS:
                            break
            with count_lock:
                if question_count >= MAX_QUESTIONS:
                    break
    
    # Finalize the JSON file by removing the trailing comma and closing the JSON array
    with open(OUTPUT_FILE, 'rb+') as f:
        f.seek(0, 2)  # Move to end of file
        pos = f.tell() - 2  # Remove last comma and newline
        f.truncate(pos)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    print("Scraping complete.")

if __name__ == '__main__':
    main()
