import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin

def download_images(url, actress_name='downloaded_images', image_to_download=5):
    output_folder = os.path.join('downloaded_images', actress_name)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    images = soup.find_all('img', src=re.compile(r'/tgpx/thumbs/'))
    
    # Download each image
    for i, img in enumerate(images):
        if i >= image_to_download:
            break
        img_url = img['src']
        if not img_url.startswith('http'):
            # If the URL is relative, convert it to an absolute URL
            if img_url.startswith('/'):
                img_url = f"https://{response.url.split('/')[2]}{img_url}"
            else:
                img_url = f"{response.url.rstrip('/')}/{img_url}"
                
        try:
            # Check if the image already exists
            if os.path.exists(os.path.join(output_folder, f"image_{i}_{img_url.split('/')[-1]}")):
                print(f"Image {img_url.split('/')[-1]} already exists")
                continue

            # Download the image
            img_response = requests.get(img_url, headers=headers)
            
            # Extract the filename from the URL
            filename = f"image_{i}_{img_url.split('/')[-1]}"
            filepath = os.path.join(output_folder, filename)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(img_response.content)
            print(f"Downloaded: {filename}")
            
        except Exception as e:
            print(f"Error while downloading {img_url}: {str(e)}")

def get_actress_links():
    # URL de base
    base_url = "https://www.thumbnailseries.com/pornstars"
    
    try:
        # Make the HTTP request
        response = requests.get(base_url)
        response.raise_for_status()  # Check if the request was successful
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links
        links = soup.find_all('a')
        
        # Filter the links that match the pattern test.com/birds/something
        actress_links = {}
        for link in links:
            href = link.get('href')
            if href:
                # Convert the relative link to an absolute link if necessary
                full_url = urljoin(base_url, href)
                # Check if the link matches the desired pattern
                if full_url.startswith(base_url + '/'):
                    print(full_url)
                    actress_name = full_url.split('/')[-2]
                    actress_links[full_url] = actress_name
        
        return actress_links
        
    except requests.RequestException as e:
        print(f"Error: while getting actress links: {e}")
        return []

# Example of usage
if __name__ == "__main__":
    actress_links = get_actress_links()
    for url, name in actress_links.items():
        print(f"URL: {url}")
        print(f"Nom: {name}")
        download_images(url, name, 5)