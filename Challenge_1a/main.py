import os
import json
import pdfplumber
from typing import List, Dict, Optional
from collections import defaultdict

def extract_structure(pdf_path: str) -> Dict:
    """
    Extract title and headings (H1-H3) from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing title and outline in the required format
    """
    title = None
    outline = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # First try to get title from metadata
        metadata = pdf.metadata
        title = metadata.get('Title', '').strip() if metadata else None
        
        # Process each page to find headings
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
                
            # Split into lines and process each line
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                # Simple heuristic for heading detection
                # In a real solution, you'd want more sophisticated detection
                if not title and len(line) < 50 and not line.endswith('.'):
                    title = line
                    continue
                    
                # Detect headings based on common patterns
                if is_heading(line):
                    level = determine_heading_level(line, page)
                    if level:
                        outline.append({
                            "level": f"H{level}",
                            "text": line,
                            "page": page_num
                        })
    
    # If we couldn't find a title in metadata or first page, use filename
    if not title:
        title = os.path.basename(pdf_path).replace('.pdf', '')
    
    return {
        "title": title,
        "outline": outline
    }

def is_heading(text: str) -> bool:
    """Determine if a text line is likely a heading."""
    # Skip very long lines (likely paragraphs)
    if len(text) > 100:
        return False
        
    # Skip lines that look like regular sentences
    if text.endswith('.') or text.endswith(','):
        return False
        
    # Skip lines with many lowercase letters in the middle
    if sum(1 for c in text if c.islower()) > len(text) / 3:
        return False
        
    return True

def determine_heading_level(text: str, page) -> Optional[int]:
    """Determine heading level (1-3) based on simple heuristics."""
    # In a real solution, you'd analyze font size, weight, etc.
    # This is a simplified version for demonstration
    
    # Count capital letters as a simple proxy for importance
    capital_ratio = sum(1 for c in text if c.isupper()) / len(text)
    
    if capital_ratio > 0.7:
        return 1
    elif capital_ratio > 0.5:
        return 2
    elif capital_ratio > 0.3:
        return 3
    return None

def process_pdfs(input_dir: str, output_dir: str):
    """Process all PDFs in input directory and save JSONs to output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            try:
                result = extract_structure(pdf_path)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                print(f"Processed {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    print("Starting PDF processing...")
    process_pdfs(input_dir, output_dir)
    print("Processing complete.")