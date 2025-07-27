import os
import json
import pdfplumber
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK data (punkt for sentence tokenization)
nltk.download('punkt')

class DocumentAnalyzer:
    def __init__(self):
        # Load a lightweight sentence transformer model
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text from PDF with page numbers."""
        text_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    text_pages.append((page_num, text))
        return text_pages
    
    def split_into_sections(self, text_pages: List[Tuple[int, str]]) -> List[Dict]:
        """Split document into logical sections with page numbers."""
        sections = []
        current_section = []
        current_page = 1
        
        for page_num, text in text_pages:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            for para in paragraphs:
                # Simple heuristic for section breaks
                if self.is_section_header(para):
                    if current_section:
                        sections.append({
                            "text": "\n".join(current_section),
                            "page": current_page
                        })
                        current_section = []
                    current_section.append(para)
                    current_page = page_num
                else:
                    current_section.append(para)
        
        if current_section:
            sections.append({
                "text": "\n".join(current_section),
                "page": current_page
            })
            
        return sections
    
    def is_section_header(self, text: str) -> bool:
        """Determine if text is likely a section header."""
        sentences = sent_tokenize(text)
        if len(sentences) != 1:
            return False
        return (len(text) < 100 and 
                not text.endswith('.') and 
                sum(1 for c in text if c.isupper()) > len(text)/3)
    
    def rank_sections(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Rank sections based on relevance to persona and job."""
        # Encode persona and job as a query
        query_embedding = self.model.encode([f"{persona}. {job}"], convert_to_tensor=True)
        
        # Encode all sections
        section_texts = [section["text"] for section in sections]
        section_embeddings = self.model.encode(section_texts, convert_to_tensor=True)
        
        # Calculate similarity scores
        scores = cosine_similarity(
            query_embedding.cpu().numpy(),
            section_embeddings.cpu().numpy()
        )[0]
        
        # Rank sections by score
        ranked_sections = []
        for idx, score in enumerate(scores):
            ranked_sections.append({
                **sections[idx],
                "importance_score": float(score),
                "importance_rank": 0  # Will be set after sorting
            })
        
        # Sort by score descending and set ranks
        ranked_sections.sort(key=lambda x: x["importance_score"], reverse=True)
        for rank, section in enumerate(ranked_sections, start=1):
            section["importance_rank"] = rank
            
        return ranked_sections
    
    def extract_key_subsections(self, section: Dict, persona: str, job: str) -> List[Dict]:
        """Extract and rank key subsections from a section."""
        sentences = sent_tokenize(section["text"])
        if len(sentences) <= 1:
            return []
            
        # Encode persona and job as query
        query_embedding = self.model.encode([f"{persona}. {job}"], convert_to_tensor=True)
        
        # Encode all sentences
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # Calculate similarity scores
        scores = cosine_similarity(
            query_embedding.cpu().numpy(),
            sentence_embeddings.cpu().numpy()
        )[0]
        
        # Get top 3 most relevant sentences
        top_indices = np.argsort(scores)[-3:][::-1]
        subsections = []
        for idx in top_indices:
            subsections.append({
                "refined_text": sentences[idx],
                "page": section["page"],
                "relevance_score": float(scores[idx])
            })
            
        return subsections
    
    def process_documents(self, input_dir: str, persona: str, job: str) -> Dict:
        """Process all documents and generate the required output."""
        documents = []
        all_sections = []
        
        # Process each PDF in input directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                text_pages = self.extract_text_from_pdf(pdf_path)
                sections = self.split_into_sections(text_pages)
                ranked_sections = self.rank_sections(sections, persona, job)
                
                documents.append({
                    "filename": filename,
                    "sections": ranked_sections
                })
                all_sections.extend([{
                    **section,
                    "document": filename
                } for section in ranked_sections])
        
        # Rank all sections across documents
        all_sections.sort(key=lambda x: x["importance_score"], reverse=True)
        for rank, section in enumerate(all_sections, start=1):
            section["importance_rank"] = rank
            
        # Prepare final output
        output = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in documents],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Populate extracted sections (top 10 across all documents)
        for section in all_sections[:10]:
            output["extracted_sections"].append({
                "document": section["document"],
                "page_number": section["page"],
                "section_title": section["text"][:100] + ("..." if len(section["text"]) > 100 else ""),
                "importance_rank": section["importance_rank"]
            })
            
            # Add subsection analysis for top sections
            if section["importance_rank"] <= 5:
                subsections = self.extract_key_subsections(section, persona, job)
                for sub in subsections:
                    output["subsection_analysis"].append({
                        "document": section["document"],
                        "page_number": sub["page"],
                        "refined_text": sub["refined_text"],
                        "relevance_score": sub["relevance_score"]
                    })
        
        return output

def load_config(config_path: str) -> Tuple[str, str]:
    """Load persona and job from config file."""
    with open(config_path) as f:
        config = json.load(f)
    return config.get("persona"), config.get("job_to_be_done")

if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"
    config_path = "/app/config.json"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    try:
        persona, job = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        persona = "Generic Researcher"
        job = "Extract relevant information"
    
    # Process documents
    analyzer = DocumentAnalyzer()
    print("Starting document processing...")
    result = analyzer.process_documents(input_dir, persona, job)
    
    # Save output
    output_path = os.path.join(output_dir, "output.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Processing complete. Results saved to {output_path}")