import json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Any
import re

class ManagerRAGSystem:
    def __init__(self, api_key: str, json_file: str = "managerial_dataset.json"):
        """
        ✅ FIXED: Correct model + error handling
        """
        genai.configure(api_key=api_key)
        # ✅ FIXED: Correct model name
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ✅ FIXED: File existence check
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"❌ '{json_file}' not found! Run create_combined_dataset() first")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        self.questions, self.answers, self.metadata = [], [], []
        for item in self.dataset:
            self.questions.append(item['question'])
            self.answers.append(json.dumps(item['answer'], ensure_ascii=False))
            self.metadata.append({
                'id': item['id'], 'role_level': item['role_level'], 
                'domain': item['domain'], 'question': item['question']
            })
        
        print("🔄 Generating embeddings...")
        self.question_embeddings = self.embedding_model.encode(self.questions)
        print(f"✅ Loaded {len(self.questions)} examples")
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents."""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_docs = []
        for idx in top_indices:
            relevant_docs.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarities[idx])
            })
        return relevant_docs
    
    def print_retrieval_info(self, docs: List[Dict[str, Any]]):
        """🎯 NEW: Print retrieval transparency"""
        print(f"\n📊 RAG RETRIEVAL INFO")
        print(f"   🔍 Chunks retrieved: {len(docs)}")
        print("   📈 Top matches:")
        for i, doc in enumerate(docs, 1):
            sim_pct = doc['similarity'] * 100
            color = "🟢" if sim_pct > 70 else "🟡" if sim_pct > 40 else "🔴"
            print(f"     {color} #{i}: {sim_pct:.1f}% | {doc['metadata']['role_level']} | {doc['metadata']['domain']}")
        print()
    
    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Concise context for LLM."""
        context = "SIMILAR EXAMPLES:\n"
        for doc in docs:
            context += f"Q: {doc['metadata']['question']}\n{doc['answer']}\n---\n"
        return context
    
    def generate_response(self, query: str, top_k: int = 3) -> str:
        """✅ Generate RAG response with transparency."""
        relevant_docs = self.retrieve_relevant_docs(query, top_k=top_k)
        
        # 🎯 NEW: Print retrieval info BEFORE generation
        self.print_retrieval_info(relevant_docs)
        
        context = self.format_context(relevant_docs)
        prompt = f"""EXPERT MANAGERIAL CONSULTANT:

QUERY: {query}

{context}

MANDATORY FORMAT:
**Problem Explanation**
**Steps to Solve** (6-8 numbered steps)
**Alternate Approaches** (3 bullet points)

Respond:"""
        
        print(prompt)

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Generation error: {str(e)}"

def main():
    API_KEY = "AIzaSyAKD3mmegeIJEZ82FhPhJuHMuXKy1dR_ig"
    
    try:
        rag = ManagerRAGSystem(api_key=API_KEY)
        print("🚀 Managerial RAG System Ready! (top_k=3)")
        
        while True:
            query = input("\n👤 Manager: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                print("\n🤖 RAG Consultant Response:")
                print("=" * 80)
                response = rag.generate_response(query)
                print(response)
                print("=" * 80)
                
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()
