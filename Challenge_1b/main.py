import os
import json
from datetime import datetime
from pdfminer.high_level import extract_text # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
import string

# Initialize models and resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    """Basic text preprocessing"""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    return ' '.join([word for word in tokens if word not in stop_words and word.isalpha()])

def extract_sections_from_pdf(pdf_path):
    """Extract structured sections from PDF"""
    text = extract_text(pdf_path)
    # Basic section extraction - would be enhanced from Round 1A solution
    sections = []
    current_page = 1
    for i, part in enumerate(text.split('\n\n')):
        if part.strip():
            # Simple heuristic for section detection - would be improved
            if part.strip().isupper() or (len(part) < 100 and '\n' not in part):
                sections.append({
                    'text': part.strip(),
                    'page': current_page  # PDFMiner doesn't track pages well - would improve
                })
            else:
                if sections:
                    sections[-1]['content'] = sections[-1].get('content', '') + '\n' + part.strip()
    return sections

def compute_relevance(persona_embedding, job_embedding, text_embedding):
    """Compute combined relevance score"""
    persona_sim = cosine_similarity([persona_embedding], [text_embedding])[0][0]
    job_sim = cosine_similarity([job_embedding], [text_embedding])[0][0]
    return 0.6 * job_sim + 0.4 * persona_sim  # Weighted combination

def process_documents(input_dir, persona, job):
    """Main processing function"""
    # Embed persona and job description
    persona_embedding = model.encode(preprocess_text(persona))
    job_embedding = model.encode(preprocess_text(job))
    
    results = {
        'metadata': {
            'input_documents': [],
            'persona': persona,
            'job_to_be_done': job,
            'processing_timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'extracted_sections': [],
        'sub_section_analysis': []
    }
    
    # Process each PDF in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            results['metadata']['input_documents'].append(filename)
            
            sections = extract_sections_from_pdf(pdf_path)
            for section in sections:
                # Compute relevance for each section
                text_embedding = model.encode(preprocess_text(section['text']))
                relevance = compute_relevance(persona_embedding, job_embedding, text_embedding)
                
                results['extracted_sections'].append({
                    'document': filename,
                    'page_number': section['page'],
                    'section_title': section['text'],
                    'importance_rank': relevance
                })
                
                # Simple subsection analysis - would be enhanced
                if 'content' in section:
                    content = section['content']
                    sentences = [s for s in content.split('.') if s.strip()]
                    if sentences:
                        top_sentence = max(sentences, 
                                         key=lambda s: compute_relevance(
                                             persona_embedding, 
                                             job_embedding, 
                                             model.encode(preprocess_text(s))
                                         ))
                        
                        results['sub_section_analysis'].append({
                            'document': filename,
                            'page_number': section['page'],
                            'refined_text': top_sentence.strip(),
                            'importance_rank': relevance
                        })
    
    # Sort results by importance
    results['extracted_sections'].sort(key=lambda x: x['importance_rank'], reverse=True)
    results['sub_section_analysis'].sort(key=lambda x: x['importance_rank'], reverse=True)
    
    return results

if __name__ == "__main__":
    # Read input files
    input_dir = '/app/input'
    output_dir = '/app/output'
    
    # Read persona and job from files
    with open(os.path.join(input_dir, 'persona.txt'), 'r') as f:
        persona = f.read()
    with open(os.path.join(input_dir, 'job.txt'), 'r') as f:
        job = f.read()
    
    # Process documents
    result = process_documents(input_dir, persona, job)
    
    # Write output
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'output.json'), 'w') as f:
        json.dump(result, f, indent=2)