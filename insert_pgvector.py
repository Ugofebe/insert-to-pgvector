# import boto3
import tempfile
import os
from inserting_file import load_pdf_to_dicts_s3, load_pdf_to_dicts 
from jac_functions import insert_publications



if __name__ == "__main__":
    # input_dir = "s3://esutlibrary/media/bulletins/"
    input_dir = "data/BIO 102"
    # Process PDFs directly from S3
    publications = load_pdf_to_dicts(input_dir, max_workers=2)
    # publications = load_pdf_to_dicts(input_dir, max_workers=2)
    print(f" Total documents loaded: {len(publications)}")
    
    # Insert into vector database
    db = insert_publications(publications)