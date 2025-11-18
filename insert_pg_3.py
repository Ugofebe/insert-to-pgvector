import tempfile
import os
import json
import shutil
from pathlib import Path
from inserting_file import load_pdf_to_dicts_s3, load_pdf_to_dicts 
from jac_functions import insert_publications

def get_processed_pdfs(tracking_file="processed_pdfs.json"):
    """Load the list of already processed PDF files"""
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_pdfs(pdf_files, tracking_file="processed_pdfs.json"):
    """Save processed PDF files to tracking file"""
    with open(tracking_file, 'w') as f:
        json.dump(list(pdf_files), f, indent=2)

def get_pdf_files(input_dir):
    """Get list of PDF files from directory"""
    if input_dir.startswith('s3://'):
        # For S3, we can't list files locally, so we'll process differently
        return []
    else:
        pdf_files = list(Path(input_dir).glob("**/*.pdf"))
        return [str(pdf) for pdf in pdf_files]

def process_pdfs_one_by_one(input_dir, tracking_file="processed_pdfs.json", max_workers=1):
    """
    Process PDFs one by one to avoid memory issues and track actual PDF files
    """
    # Get already processed PDFs
    processed_pdfs = get_processed_pdfs(tracking_file)
    print(f"📋 Found {len(processed_pdfs)} already processed PDF files")
    
    # Get all PDF files
    all_pdf_files = get_pdf_files(input_dir)
    print(f"🔍 Found {len(all_pdf_files)} PDF files in directory")
    
    # Filter out already processed PDFs
    new_pdf_files = [pdf for pdf in all_pdf_files if pdf not in processed_pdfs]
    print(f"🆕 New PDF files to process: {len(new_pdf_files)}")
    
    if not new_pdf_files:
        print("✅ All PDF files have already been processed!")
        return
    
    total_processed = 0
    for pdf_file in new_pdf_files:
        print(f"\n🔄 Processing: {Path(pdf_file).name}")
        
        # Create a temporary directory with only this PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Copy only this PDF to temporary directory
                temp_pdf_path = os.path.join(temp_dir, Path(pdf_file).name)
                shutil.copy2(pdf_file, temp_pdf_path)
                
                # Load only this single PDF
                publications = load_pdf_to_dicts_s3(temp_dir, max_workers=max_workers)
                print(f"   📊 Generated {len(publications)} chunks from this PDF")
                
                if publications:
                    # Insert into vector database
                    db = insert_publications(publications)
                    print(f"   ✅ Inserted {len(publications)} chunks into database")
                
                # Mark this PDF as processed
                processed_pdfs.add(pdf_file)
                save_processed_pdfs(processed_pdfs, tracking_file)
                
                total_processed += 1
                print(f"✅ Completed: {Path(pdf_file).name} ({total_processed}/{len(new_pdf_files)})")
                
            except Exception as e:
                print(f"❌ Error processing {Path(pdf_file).name}: {str(e)}")
                continue
    
    print(f"\n🎉 Processing complete! Processed {total_processed} new PDF files")

def process_pdfs_in_batches(input_dir, pdfs_per_batch=1, tracking_file="processed_pdfs.json", max_workers=1):
    """
    Process PDFs in batches (batch of PDF files, not chunks)
    """
    # Get already processed PDFs
    processed_pdfs = get_processed_pdfs(tracking_file)
    print(f"📋 Found {len(processed_pdfs)} already processed PDF files")
    
    # Get all PDF files
    all_pdf_files = get_pdf_files(input_dir)
    print(f"🔍 Found {len(all_pdf_files)} PDF files in directory")
    
    # Filter out already processed PDFs
    new_pdf_files = [pdf for pdf in all_pdf_files if pdf not in processed_pdfs]
    print(f"🆕 New PDF files to process: {len(new_pdf_files)}")
    
    if not new_pdf_files:
        print("✅ All PDF files have already been processed!")
        return
    
    total_processed = 0
    for i in range(0, len(new_pdf_files), pdfs_per_batch):
        batch_files = new_pdf_files[i:i + pdfs_per_batch]
        
        print(f"\n🔄 Processing batch {i//pdfs_per_batch + 1}/{(len(new_pdf_files)-1)//pdfs_per_batch + 1}")
        
        # Create a temporary directory with only the batch PDFs
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Copy batch PDFs to temporary directory
                for pdf_file in batch_files:
                    temp_pdf_path = os.path.join(temp_dir, Path(pdf_file).name)
                    shutil.copy2(pdf_file, temp_pdf_path)
                    print(f"   📄 Added to batch: {Path(pdf_file).name}")
                
                # Load only the batch PDFs
                publications = load_pdf_to_dicts_s3(temp_dir, max_workers=max_workers)
                print(f"   📊 Generated {len(publications)} chunks from {len(batch_files)} PDFs")
                
                if publications:
                    # Insert into vector database
                    db = insert_publications(publications)
                    print(f"   ✅ Inserted {len(publications)} chunks into database")
                
                # Mark PDFs as processed
                processed_pdfs.update(batch_files)
                save_processed_pdfs(processed_pdfs, tracking_file)
                
                total_processed += len(batch_files)
                print(f"✅ Batch completed. Total PDFs processed: {total_processed}/{len(new_pdf_files)}")
                
            except Exception as e:
                print(f"❌ Error processing batch: {str(e)}")
                continue
    
    print(f"\n🎉 Processing complete! Processed {total_processed} new PDF files")

# SIMPLE VERSION - Just change the function call
if __name__ == "__main__":
    # CONFIGURATION - CHANGE THESE AS NEEDED:
    input_dir = "data"  # For local files
    # input_dir = "s3://esutlibrary/media/bulletins/"  # For S3 files
    
    # CHOOSE YOUR PROCESSING METHOD:
    
    # Option 1: Process one PDF at a time (most memory efficient)
    # print("🚀 Starting PDF processing (one by one)...")
    # process_pdfs_one_by_one(
    #     input_dir=input_dir,
    #     tracking_file="processed_pdfs.json",
    #     max_workers=1
    # )
    
    # Option 2: Process multiple PDFs per batch
    print("🚀 Starting PDF processing (in batches)...")
    process_pdfs_in_batches(
        input_dir=input_dir,
        pdfs_per_batch=2,  # Number of PDFs to process together
        tracking_file="processed_pdfs.json",
        max_workers=1
    )