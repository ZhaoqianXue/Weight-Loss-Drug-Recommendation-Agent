# code_scraping/scraper.py

from lxml import etree
import csv
import requests
import time
import re

def scrape_all_reviews_from_url(base_url, csv_file_path, drug_name=None, brand_name=None):
    """
    Automatically scrape all reviews by paging until no new reviews are found.
    drug_name: generic name, brand_name: brand name, both as the first two columns of each row.
    """
    try:
        header = ['Drug Name', 'Brand Name', 'Date', 'User', 'Age', 'Gender', 'Patient Type', 'Medication Duration', 
                  'Condition', 'Overall Rating', 'Effectiveness', 'Ease of Use', 
                  'Satisfaction', 'Likes', 'Dislikes', 'Textual Review']
        all_data = []
        page = 1
        while True:
            url = base_url.replace('page=1', f'page={page}')
            print(f"Scraping page {page}: {url}")
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            html_content = response.content
            parser = etree.HTMLParser()
            tree = etree.fromstring(html_content, parser)
            review_elements = tree.xpath("//div[contains(@class, 'review-details-holder')]")
            if not review_elements:
                print(f"No reviews found on page {page}, scraping finished.")
                break
            page_data = []
            for review_element in review_elements:
                def get_text(element, xpath):
                    result = element.xpath(xpath)
                    return result[0].strip() if result and result[0].strip() else None
                def get_attribute_value(element, xpath, attribute_name):
                    result = element.xpath(xpath)
                    return result[0].get(attribute_name) if result else None
                date = get_text(review_element, ".//div[@class='date']/text()")
                details_spans = review_element.xpath(".//div[@class='details']/span/text()")
                details_text = [span.strip() for span in details_spans if span.strip()]
                user = None
                age = None
                gender = None
                patient_type = None
                medication_duration_text = get_text(review_element, ".//div[@class='details']/text()[normalize-space()]")
                if len(details_text) > 0: user = details_text[0].replace('|','').strip() if details_text[0] != '|' else None
                if len(details_text) > 1: age = details_text[1].replace('|','').strip() if details_text[1] != '|' else None
                if len(details_text) > 2: gender = details_text[2].replace('|','').strip() if details_text[2] != '|' else None
                med_duration_raw = review_element.xpath(".//div[@class='details']/text()")
                medication_duration = None
                for text_node in med_duration_raw:
                    cleaned_text = text_node.strip()
                    if cleaned_text and cleaned_text != '|':
                        medication_duration = cleaned_text.replace("On medication for", "").replace("|","").strip()
                        break
                if details_text and "Patient" in details_text[-1]:
                    patient_type = details_text[-1]
                condition = get_text(review_element, ".//strong[@class='condition']/text()")
                if condition:
                    condition = condition.replace("Condition: ", "").strip()
                overall_rating_val = get_attribute_value(review_element, ".//div[@class='overall-rating']/div[contains(@class, 'webmd-rate')]", "aria-valuenow")
                effectiveness_val = get_attribute_value(review_element, ".//div[@class='categories']/section[1]/div[contains(@class, 'webmd-rate')]", "aria-valuenow")
                ease_of_use_val = get_attribute_value(review_element, ".//div[@class='categories']/section[2]/div[contains(@class, 'webmd-rate')]", "aria-valuenow")
                satisfaction_val = get_attribute_value(review_element, ".//div[@class='categories']/section[3]/div[contains(@class, 'webmd-rate')]", "aria-valuenow")
                likes = get_text(review_element, ".//div[contains(@class, 'like-dislikes')]//div[@class='helpful']/span[@class='likes']/text()")
                dislikes_element = review_element.xpath(".//div[contains(@class, 'like-dislikes')]//div[@class='not-helpful']/span[@class='dislikes']/text()")
                dislikes = dislikes_element[0].strip() if dislikes_element and dislikes_element[0].strip() else "0"
                # Prefer to concatenate showSec and hiddenSec for long reviews
                show_sec = review_element.xpath(".//div[@class='description']/p[@class='description-text']/span[contains(@class, 'showSec')]/text()")
                hidden_sec = review_element.xpath(".//div[@class='description']/p[@class='description-text']/span[contains(@class, 'hiddenSec')]/text()")
                if show_sec or hidden_sec:
                    textual_review = "".join([s.strip() for s in show_sec + hidden_sec if s.strip()])
                else:
                    # Compatible with short reviews
                    textual_review_parts = review_element.xpath(".//div[@class='description']/p[@class='description-text']/text()")
                    textual_review = " ".join([part.strip() for part in textual_review_parts if part.strip()])
                    if not textual_review:
                        desc_div = review_element.xpath(".//div[@class='description']")
                        if desc_div:
                            for node in desc_div[0].itertext():
                                t = node.strip()
                                if t:
                                    textual_review = t
                                    break
                # Remove extra quotes and spaces at the beginning and end
                if textual_review:
                    textual_review = textual_review.strip().strip('"')
                full_details_str = "".join(review_element.xpath(".//div[@class='details']//text()") ).strip()
                parts = [p.strip() for p in full_details_str.split('|') if p.strip()]
                if not user and len(parts) > 0: user = parts[0].replace('|','').strip()
                if not age and len(parts) > 1: age = parts[1].replace('|','').strip()
                if not gender and len(parts) > 2: gender = parts[2].replace('|','').strip()
                if not medication_duration and len(parts) > 3 and "On medication for" in parts[3]:
                    medication_duration = parts[3].replace("On medication for", "").strip()
                if not patient_type and len(parts) > 4: patient_type = parts[4]
                data = [drug_name, brand_name, date, user, age, gender, patient_type, medication_duration, 
                        condition, overall_rating_val, effectiveness_val, ease_of_use_val, 
                        satisfaction_val, likes, dislikes, textual_review]
                page_data.append(data)
            # If the data on this page is the same as the previous page, stop (prevent infinite loop)
            if page_data and page_data == all_data[-len(page_data):]:
                print(f"Page {page} content is the same as the previous page, scraping finished.")
                break
            all_data.extend(page_data)
            page += 1
            time.sleep(1)  # Avoid being blocked by sending requests too quickly
        # Write to CSV
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(all_data)
        print(f"Successfully scraped all reviews from all pages and saved to {csv_file_path}")
        print(f"Total reviews: {len(all_data)}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # By default, scrape Zepbound. You need to manually specify drug_name and brand_name
    base_url = "https://reviews.webmd.com/drugs/drugreview-187794-zepbound-subcutaneous"
    drug_name = "Tirzepatide"
    brand_name = "Zepbound"
    # Auto-complete first page parameter
    if "page=" not in base_url:
        if "?" in base_url:
            base_url = base_url + "&page=1&next_page=true"
        else:
            base_url = base_url + "?page=1&next_page=true"
    csv_file = f"webmd_{brand_name.lower()}_reviews.csv"
    scrape_all_reviews_from_url(base_url, csv_file, drug_name, brand_name)
