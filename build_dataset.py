import requests
import os
from PIL import Image
import io
import json
import sys


def parse_page(page, res):
    url_list = []
    data = json.loads(page)
    images = data['results']
    for img in images:
        img_url = img['urls'][res]
        url_list.append(img_url)
    return url_list


def fetch_images(base_url, maximum, res):
    page_size = 30
    url_list = []

    page_index = 1
    while len(url_list) < maximum:
        page = requests.get('%s&page=%d&per_page=%d' % (base_url, page_index, page_size)).text
        page_urls = parse_page(page, res)
        url_list.append(page_urls)
        page_index += 1
    url_list = url_list[:maximum]
    return url_list


def download_image(folder_path, file_name, img_url):
    image_content = requests.get(img_url).content
    image_file = io.BytesIO(image_content)
    image = Image.open(image_file).convert('RGB')
    if os.path.exists(folder_path):
        file_path = os.path.join(folder_path, file_name)
    else:
        os.mkdir(folder_path)
        file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'wb') as f:
        image.save(f, "JPEG", quality=85)


keyword = sys.argv[1]
path = sys.argv[2]
img_number = int(sys.argv[3])
base = 'https://unsplash.com/napi/search/photos?query='
url = base + keyword + "="

urls = fetch_images(url, img_number, 'regular')

dir_path = os.path.join(path, keyword)
os.mkdir(os.path.join(path, keyword))

for url_index in range(len(urls)):
    download_image(dir_path, str(url_index) + ".jpg", urls[url_index])
