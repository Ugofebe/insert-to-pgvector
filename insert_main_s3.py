import tempfile
import os
import json
import shutil
from pathlib import Path
import boto3

from inserting_file import load_pdf_to_dicts_s3, load_pdf_to_dicts
from jac_functions import insert_publications


# ---------------------------------------------------------
# TRACKING ALREADY PROCESSED PDF FILES
# ---------------------------------------------------------

def get_processed_pdfs(tracking_file="processed_pdfs.json"):
    """Load the list of already processed PDF files."""
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_processed_pdfs(pdf_files, tracking_file="processed_pdfs.json"):
    """Save processed PDF files to tracking file."""
    with open(tracking_file, 'w') as f:
        json.dump(list(pdf_files), f, indent=2)


# ---------------------------------------------------------
# LIST PDF FILES (LOCAL + S3 SUPPORT)
# ---------------------------------------------------------

def get_pdf_files(input_dir):
    """
    Returns PDF file paths:
    - Local filesystem → full local paths
    - S3 → s3://bucket/prefix/file.pdf paths
    """

    # ----------- CASE 1: S3 DIRECTORY -----------
    if input_dir.startswith('s3://'):
        s3 = boto3.client("s3")

        # Parse bucket + prefix
        path_no_scheme = input_dir.replace("s3://", "")
        bucket = path_no_scheme.split("/")[0]
        prefix = "/".join(path_no_scheme.split("/")[1:])

        print(f"🔍 Listing PDFs from S3 bucket: {bucket}, prefix: {prefix}")

        pdf_files = []
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    pdf_files.append(f"s3://{bucket}/{key}")

        return pdf_files

    # ----------- CASE 2: LOCAL DIRECTORY -----------

    pdf_files = list(Path(input_dir).glob("**/*.pdf"))
    return [str(pdf) for pdf in pdf_files]


# ---------------------------------------------------------
# DOWNLOAD A SINGLE S3 FILE → TEMP DIRECTORY
# ---------------------------------------------------------

def download_s3_pdf(s3_path, temp_dir):
    """Download one PDF from S3 into the temp directory."""
    s3 = boto3.client("s3")

    path_no_scheme = s3_path.replace("s3://", "")
    bucket = path_no_scheme.split("/")[0]
    key = "/".join(path_no_scheme.split("/")[1:])
    filename = os.path.basename(key)

    local_path = os.path.join(temp_dir, filename)

    print(f"⬇️ Downloading from S3: {s3_path}")
    s3.download_file(bucket, key, local_path)

    return local_path


# ---------------------------------------------------------
# PROCESS PDFs ONE BY ONE (LOCAL OR S3)
# ---------------------------------------------------------

def process_pdfs_one_by_one(input_dir, tracking_file="processed_pdfs.json", max_workers=1):
    """Process PDFs one by one, with full S3 support."""

    processed_pdfs = get_processed_pdfs(tracking_file)
    print(f"📋 Found {len(processed_pdfs)} already processed PDF files")

    all_pdf_files = get_pdf_files(input_dir)
    print(f"🔍 Found {len(all_pdf_files)} PDF files in source")

    new_pdf_files = [pdf for pdf in all_pdf_files if pdf not in processed_pdfs]
    print(f"🆕 New PDF files to process: {len(new_pdf_files)}")

    if not new_pdf_files:
        print("✅ All PDFs processed!")
        return

    total_processed = 0

    for pdf_file in new_pdf_files:
        print(f"\n🔄 Processing: {os.path.basename(pdf_file)}")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # STEP 1: Copy/download PDF into temp_dir
                if pdf_file.startswith("s3://"):
                    temp_pdf_path = download_s3_pdf(pdf_file, temp_dir)
                else:
                    temp_pdf_path = os.path.join(temp_dir, os.path.basename(pdf_file))
                    shutil.copy2(pdf_file, temp_pdf_path)

                # STEP 2: Process the single PDF
                publications = load_pdf_to_dicts(temp_dir, max_workers=max_workers)
                print(f"   📊 Extracted {len(publications)} chunks")

                if publications:
                    insert_publications(publications)
                    print("   ✅ Inserted chunks into database")

                # STEP 3: Mark as processed
                processed_pdfs.add(pdf_file)
                save_processed_pdfs(processed_pdfs, tracking_file)

                total_processed += 1
                print(f"✅ Completed: {os.path.basename(pdf_file)} ({total_processed}/{len(new_pdf_files)})")

            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                continue

    print(f"\n🎉 Fully processed {total_processed} new PDFs")


# ---------------------------------------------------------
# PROCESS PDFs IN BATCHES (S3 + LOCAL)
# ---------------------------------------------------------

def process_pdfs_in_batches(input_dir, pdfs_per_batch=1, tracking_file="processed_pdfs.json", max_workers=1):
    """Process multiple PDFs per batch (local or S3)."""

    processed_pdfs = get_processed_pdfs(tracking_file)
    print(f"📋 Found {len(processed_pdfs)} already processed PDF files")

    all_pdf_files = get_pdf_files(input_dir)
    print(f"🔍 Found {len(all_pdf_files)} PDF files")

    new_pdf_files = [pdf for pdf in all_pdf_files if pdf not in processed_pdfs]
    print(f"🆕 New PDF files to process: {len(new_pdf_files)}")

    if not new_pdf_files:
        print("✅ All PDFs processed!")
        return

    total_processed = 0

    for i in range(0, len(new_pdf_files), pdfs_per_batch):
        batch = new_pdf_files[i:i + pdfs_per_batch]

        print(f"\n🔄 Processing batch {i//pdfs_per_batch + 1}")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # STEP 1: Add PDFs to batch folder
                for pdf_file in batch:
                    if pdf_file.startswith("s3://"):
                        download_s3_pdf(pdf_file, temp_dir)
                    else:
                        shutil.copy2(pdf_file, os.path.join(temp_dir, os.path.basename(pdf_file)))
                    print(f"   📄 Added: {os.path.basename(pdf_file)}")

                # STEP 2: Load & insert all PDFs in the batch
                publications = load_pdf_to_dicts(temp_dir, max_workers=max_workers)
                print(f"   📊 Extracted {len(publications)} chunks")

                if publications:
                    insert_publications(publications)
                    print(f"   ✅ Inserted chunks into database")

                # STEP 3: Mark all as processed
                processed_pdfs.update(batch)
                save_processed_pdfs(processed_pdfs, tracking_file)

                total_processed += len(batch)
                print(f"✅ Batch done. Total: {total_processed}/{len(new_pdf_files)}")

            except Exception as e:
                print(f"❌ Error in batch: {e}")
                continue

    print(f"\n🎉 Finished processing {total_processed} PDFs")


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    # input_dir = "data" 
    input_dir = "s3://esutlibrary/media/theses/"

    print("🚀 Starting PDF processing (batches)...")
    process_pdfs_in_batches(
        input_dir=input_dir,
        pdfs_per_batch=50,
        tracking_file="processed_pdfs.json",
        max_workers=10
    )
