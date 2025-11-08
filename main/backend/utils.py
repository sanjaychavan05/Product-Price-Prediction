# backend/utils.py
import requests
import os
from concurrent.futures import ThreadPoolExecutor
import time

def download_images(image_links, output_dir='../data/images', max_workers=10):
    os.makedirs(output_dir, exist_ok=True)
    
    def download(item):
        idx, url = item
        if not url: return None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    path = f"{output_dir}/{idx}.jpg"
                    with open(path, 'wb') as f:
                        f.write(r.content)
                    return path
            except: time.sleep(1)
        return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download, enumerate(image_links)))
    return results

def extract_ipq(text):
    import re
    patterns = [
        r'(\d+)\s*(?:pack|pcs|pieces|count|items)',
        r'pack\s*of\s*(\d+)',
        r'(\d+)\s*x\s*\d+',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return int(match.group(1))
    return 1
