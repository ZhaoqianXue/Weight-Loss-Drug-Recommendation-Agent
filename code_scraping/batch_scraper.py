# code_scraping/batch_scraper.py

import os
import scraper

# Create save directory
save_dir = 'data_webmd'
os.makedirs(save_dir, exist_ok=True)

# Drug generic name, brand name, and WebMD review URL
url_list = [
    ("Semaglutide", "Wegovy", "https://reviews.webmd.com/drugs/drugreview-181658-wegovy-subcutaneous"),
    ("Semaglutide", "Ozempic", "https://reviews.webmd.com/drugs/drugreview-174491-ozempic-subcutaneous"),
    ("Semaglutide", "Rybelsus", "https://reviews.webmd.com/drugs/drugreview-178019-rybelsus-oral"),
    ("Tirzepatide", "Zepbound", "https://reviews.webmd.com/drugs/drugreview-187794-zepbound-subcutaneous"),
    ("Tirzepatide", "Mounjaro", "https://reviews.webmd.com/drugs/drugreview-184168-mounjaro-subcutaneous"),
    ("Liraglutide", "Victoza", "https://reviews.webmd.com/drugs/drugreview-153566-liraglutide-subcutaneous"),
    ("Liraglutide", "Saxenda", "https://reviews.webmd.com/drugs/drugreview-168195-saxenda-subcutaneous"),
]

for drug_name, brand_name, url in url_list:
    # Auto-complete first page parameter
    base_url = url
    if "page=" not in base_url:
        if "?" in base_url:
            base_url = base_url + "&page=1&next_page=true"
        else:
            base_url = base_url + "?page=1&next_page=true"
    csv_file = os.path.join(save_dir, f"webmd_{brand_name.lower()}_reviews.csv")
    print(f"\n==== Scraping: {brand_name} ({drug_name}) ====")
    scraper.scrape_all_reviews_from_url(base_url, csv_file, drug_name, brand_name)
    print(f"==== {brand_name} Finished ====") 