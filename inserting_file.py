import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from concurrent.futures import ThreadPoolExecutor
import boto3
import pypdf
# def load_txt_to_strings(documents_path):
#     """Load research publications from .txt files and return as list of strings"""
    
#     # List to store all documents
#     documents = []
    
#     # Load each .txt file in the documents folder
#     for file in os.listdir(documents_path):
#         if file.endswith(".txt"):
#             file_path = os.path.join(documents_path, file)
#             try:
#                 loader = TextLoader(file_path)
#                 loaded_docs = loader.load()
#                 documents.extend(loaded_docs)
#                 print(f"Successfully loaded: {file}")
#             except Exception as e:
#                 print(f"Error loading {file}: {str(e)}")
    
#     print(f"\nTotal documents loaded: {len(documents)}")
    
#     # Extract content as strings and return
#     publications = []
#     for doc in documents:
#         publications.append(doc.page_content)
    
#     return publications

def load_txt_to_strings(documents_path):
    """Load research publications from .txt files and return as list of dictionaries with content and title"""
    
    # List to store all documents
    documents = []
    
    # Load each .txt file in the documents folder
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                
                # Extract title from filename (remove .txt extension)
                title = os.path.splitext(file)[0]
                
                # Create dictionary with content and title for each document
                for doc in loaded_docs:
                    documents.append({
                        "content": doc.page_content,
                        "title": title
                    })
                
                print(f"Successfully loaded: {file} as '{title}'")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    return documents

def load_pdf_to_strings(documents_path):
    """Load research publications from PDF files (including subfolders) and return as list of strings"""
    
    # List to store all documents
    documents = []
    
    # Walk through all subfolders
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.lower().endswith(".pdf"):  # check for PDFs
                file_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    print(f"✅ Successfully loaded: {file}")
                except Exception as e:
                    print(f"❌ Error loading {file}: {str(e)}")
    
    print(f"\n📂 Total documents loaded: {len(documents)}")
    
    # # Extract content as strings and return
    # publications = [doc.page_content for doc in documents]
    # Extract content as strings and return
    publications = [doc.page_content for doc in documents if doc.page_content.strip()]

    print(f"\n📂 Total documents after stripping: {len(publications)}")

    return publications


# def load_pdf_to_dicts(documents_path):
#     """Load research publications from PDF files (including subfolders) and return as list of dicts with title and content"""
    
#     # List to store all document dicts
#     publications = []
    
#     # Walk through all subfolders
#     for root, dirs, files in os.walk(documents_path):
#         for file in files:
#             if file.lower().endswith(".pdf"):  # check for PDFs
#                 file_path = os.path.join(root, file)
#                 try:
#                     loader = PyPDFLoader(file_path)
#                     loaded_docs = loader.load()  # list of Document objects
                    
#                     # For each page (Document object) in the file
#                     for doc in loaded_docs:
#                         content = doc.page_content.strip()
#                         if content:
#                             publications.append({
#                                 "title": file,  # PDF file name
#                                 "content": content,  # Extracted text
#                                 "source": file_path  # Optional: keep full path
#                             })
#                     print(f"✅ Successfully loaded: {file}")
#                 except Exception as e:
#                     print(f"❌ Error loading {file}: {str(e)}")
    
#     print(f"\n📂 Total documents loaded: {len(publications)}")
#     return publications


import os
from concurrent.futures import ThreadPoolExecutor, as_completed



def load_pdf_to_dicts(documents_path, max_workers=None):
    """
    Load research publications from PDF files (including subfolders)
    and return as list of dicts with title and content.
    
    Args:
        documents_path (str): Path to folder containing PDFs.
        max_workers (int): Number of parallel workers for faster processing.
    """
    
    publications = []
    pdf_files = []

    # Step 1: Collect all PDF file paths (including subfolders)
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    print(f"🔍 Found {len(pdf_files)} PDF files to process...")

    # Step 2: Define worker function
    def process_pdf(file_path):
        file_name = os.path.basename(file_path)
        results = []
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                content = doc.page_content.strip()
                if content:
                    results.append({
                        "title": file_name,
                        "content": content,
                        "source": file_path
                    })
            print(f"✅ Successfully loaded: {file_name}")
        except Exception as e:
            print(f"❌ Error loading {file_name}: {str(e)}")
        return results

    # Step 3: Process PDFs concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, f): f for f in pdf_files}
        for future in as_completed(futures):
            try:
                publications.extend(future.result())
            except Exception as e:
                file_path = futures[future]
                print(f"⚠️ Error processing {file_path}: {e}")

    print(f"\n📂 Total documents loaded: {len(publications)}")
    return publications


def load_pdf_to_dicts_s3(s3_path, max_workers=2):
    """Load PDFs directly from S3 without permanent downloading"""
    publications = []
    
    # Parse S3 path
    bucket = s3_path.replace('s3://', '').split('/')[0]
    prefix = '/'.join(s3_path.replace('s3://', '').split('/')[1:])
    
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    def process_s3_pdf(obj_key, current, total):
        try:
            print(f"🔄 Processing {current}/{total}: {os.path.basename(obj_key)}")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
                s3.download_file(bucket, obj_key, temp_file.name)
                
                # Extract text from PDF
                with open(temp_file.name, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                
                filename = os.path.basename(obj_key)
                print(f"✅ Completed {current}/{total}: {filename}")
                return {
                    "title": filename.replace('.pdf', ''),
                    "content": text
                }
        except Exception as e:
            print(f"❌ Error processing {current}/{total} - {obj_key}: {e}")
            return None
    
    pdf_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]
    total_files = len(pdf_keys)
    print(f"🔍 Found {total_files} PDF files to process...")
    
    # Create list of arguments for each task (including current index and total)
    tasks = [(key, i+1, total_files) for i, key in enumerate(pdf_keys)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use starmap to pass multiple arguments
        results = list(executor.map(lambda args: process_s3_pdf(*args), tasks))
        publications = [r for r in results if r is not None]
    
    print(f"🎉 Finished processing! Successfully processed {len(publications)}/{total_files} files")
    return publications


def main():
    publication_pdfs = load_pdf_to_strings("data/400 Level")
    print(len(publication_pdfs))

    
if __name__ == "__main__":
    main()
