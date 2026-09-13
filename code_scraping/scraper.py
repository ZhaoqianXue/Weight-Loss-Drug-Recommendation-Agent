"""Validated WebMD collector using embedded page data and visible demographics."""
import csv
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from lxml import html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADER = ['Drug Name','Brand Name','Date','User','Age','Gender','Patient Type','Medication Duration','Condition','Overall Rating','Effectiveness','Ease of Use','Satisfaction','Likes','Dislikes','Textual Review','Review ID','Source URL','Source Page','Collected At','Source Visibility']

def parse_page(content, drug, page, collected_at):
    state = json.JSONDecoder().raw_decode(content.split('window.__INITIAL_STATE__=', 1)[1])[0]['all_reviews']
    title = state['drug_name'].lower()
    if drug['brand'].lower() not in title or drug['generic'].lower() not in title:
        raise ValueError('Unexpected drug page: ' + title)
    total = int(state['total_review'])
    records = [r for group in state['drug_review_nimvs'] for r in group.get('review_nimvs', [])]
    cards = html.fromstring(content).xpath('//div[contains(@class,"review-details-holder")]')
    if not records:
        raise ValueError('Missing embedded reviews')
    rows = []
    remaining = list(cards)
    for record in records:
        date = record['DatePosted'].split()[0]
        author = (record.get('DisplayName') or 'Anonymous').strip() or 'Anonymous'
        if '@' in author: author = '[email protected]'
        candidates = [card for card in remaining
                      if card.xpath('.//div[@class="date"]')[0].text_content().strip() == date
                      and card.xpath('.//div[@class="details"]')[0].text_content().split('|')[0].strip().replace('\xa0',' ') == author]
        card = candidates[0] if candidates else None
        parts = []
        if card is not None:
            remaining.remove(card)
            parts = [p.strip() for p in card.xpath('.//div[@class="details"]')[0].text_content().split('|')]
            ratings = card.xpath('.//div[@class="overall-rating"]//*[@role="slider"]/@aria-valuenow')
            if not ratings or abs(float(ratings[0])-float(record['OverAll_UserReviewRating'])) > .11:
                raise ValueError('DOM/state rating mismatch')
        row = dict.fromkeys(HEADER, '')
        row.update({'Drug Name':drug['generic'], 'Brand Name':drug['brand'], 'Date':date,
                    'User':author if author != 'Anonymous' else '', 'Condition':record.get('SecondaryName_s',''),
                    'Overall Rating':record.get('OverAll_UserReviewRating',''),
                    'Effectiveness':record.get('RatingCriteria1',''), 'Ease of Use':record.get('RatingCriteria2',''),
                    'Satisfaction':record.get('RatingCriteria3',''), 'Likes':record.get('FoundHelpfulCount',0),
                    'Dislikes':record.get('FoundHarmfulCount',0), 'Textual Review':record.get('UserExperience','').strip(),
                    'Review ID':str(record['userReviewId']), 'Source URL':drug['url'], 'Source Page':page,
                    'Collected At':collected_at, 'Source Visibility':'rendered' if card is not None else 'embedded_only'})
        # Embedded enum labels are inconsistent with the displayed page. Use visible labels.
        for part in parts[1:]:
            if re.fullmatch(r'\d{1,3}[-–]\d{1,3}|\d{1,3}\+|\d{1,3} (?:and|or) over',part): row['Age'] = part
            elif part in ('Male','Female','Transgender','Non-binary'): row['Gender'] = part
            elif part in ('Patient','Caregiver'): row['Patient Type'] = part
            elif re.match(r'On (medication|supplement) for ',part): row['Medication Duration'] = re.sub(r'^On (medication|supplement) for ','',part)
        rows.append(row)
    if remaining: raise ValueError("Unmatched rendered review cards")
    return total, rows

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)

def collect(drug, cache_dir, delay=.4):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=[429,500,502,503,504])))
    rows, seen, expected, page = [], set(), None, 1
    while expected is None or len(rows) < expected:
        cached = cache_dir / ('%s-%03d.json' % (drug['brand'],page))
        if cached.exists():
            data = json.loads(cached.read_text()); total, batch = data['total'], data['rows']
            for row in batch: row.setdefault('Source Visibility','rendered')
        else:
            response = session.get(drug['url'], params={'conditionid':'','sortval':1,'page':page,'next_page':'true'},timeout=60)
            response.raise_for_status()
            total, batch = parse_page(response.content.decode('utf-8'),drug,page,datetime.now(timezone.utc).isoformat())
            cached.write_text(json.dumps({'total':total,'rows':batch}, ensure_ascii=False),encoding='utf-8')
            time.sleep(delay)
        if expected is None: expected = total
        if total != expected: raise ValueError('Review count changed during collection; use a fresh run directory')
        for row in batch:
            if row['Review ID'] in seen: raise ValueError('Duplicate review ID/page: ' + row['Review ID'])
            seen.add(row['Review ID']); rows.append(row)
        if page > math.ceil(expected / 20) + 1: raise ValueError('Pagination exceeded expected limit')
        print('%s page %d: %d/%d' % (drug['brand'],page,len(rows),expected),flush=True)
        page += 1
    if len(rows) != expected: raise ValueError('Final count mismatch')
    return rows, {'brand':drug['brand'],'generic':drug['generic'],'url':drug['url'],'headline_count':expected,'collected_count':len(rows),'pages':page-1,'unique_review_ids':len(seen)}

def scrape_all_reviews_from_url(base_url,csv_file_path,drug_name=None,brand_name=None):
    rows, _ = collect({'url':base_url.split('?')[0],'generic':drug_name,'brand':brand_name},Path(csv_file_path).parent / '.page-cache')
    write_csv(csv_file_path,rows)
