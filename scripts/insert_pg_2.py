from inserting_file import load_pdf_to_dicts, load_pdf_to_dicts_s3 # returns list of dicts with "title" and "content"
from jac_functions import insert_publications
from store_inserted import load_seen_titles, save_seen_titles, content_hash
# input_dir = "s3://esutlibrary/media/bulletins/"
input_dir = "data"
# 1. Load what you already processed
seen = load_seen_titles()

# 2. Run your loader (it loads many docs into 'publication')
publication = load_pdf_to_dicts(input_dir, max_workers=2)

# 3. Filter only new items
new_publications = []
new_titles = set()
for doc in publication:
    title = doc.get("title")
    if title is None:
        # Optionally generate title from content hash if missing
        title = f"untitled_{content_hash(doc.get('content',''))}"
        doc['title'] = title

    if title not in seen:
        new_publications.append(doc)
        new_titles.add(title)

for pub in publication:
    if '\x00' in pub.get('content', ''):
        print(f"⚠️ Null byte found in: {pub.get('title', 'Unknown')}")

# 4. Insert only new ones
if new_publications:
    db = insert_publications(new_publications)  # your existing insertion
    # 5. Persist updated seen list
    seen.update(new_titles)
    save_seen_titles(seen)
    print(f"Inserted {len(new_publications)} new publications; {len(seen)} total tracked.")
else:
    print("No new publications to insert.")
