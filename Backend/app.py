from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime

# Add RAG directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RAG'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))

from rag_managerial_system import ManagerRAGSystem
from model_utils import ModelPredictor

app = Flask(__name__)
CORS(app)

# Initialize systems
print("🚀 Initializing Flask Application...")

# RAG System
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
try:
    rag_system = ManagerRAGSystem(
        api_key=GEMINI_API_KEY,
        json_file=os.path.join(os.path.dirname(__file__), '..', 'RAG', 'managerial_dataset.json')
    )
    print("✅ RAG System initialized")
except Exception as e:
    print(f"⚠️ RAG System failed to initialize: {e}")
    rag_system = None

# Model Predictor
try:
    model_predictor = ModelPredictor(
        model_path=os.path.join(os.path.dirname(__file__), '..', 'Model', 'startup_success_model.pkl'),
        config_path=os.path.join(os.path.dirname(__file__), '..', 'Model', 'model_config.json')
    )
    print("✅ Model Predictor initialized")
except Exception as e:
    print(f"⚠️ Model Predictor failed to initialize: {e}")
    model_predictor = None

print("🎉 Flask app ready!")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns system status and availability of services
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'rag_system': 'available' if rag_system else 'unavailable',
            'model_predictor': 'available' if model_predictor else 'unavailable'
        },
        'message': 'Startup Success Prediction API is running'
    }), 200


@app.route('/rag/query', methods=['POST'])
def rag_query():
    """
    RAG System endpoint for managerial queries
    
    Request body:
    {
        "query": "How to improve team productivity?",
        "top_k": 3  // optional, defaults to 3
    }
    """
    if not rag_system:
        return jsonify({
            'error': 'RAG system not available',
            'message': 'RAG system failed to initialize'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Query parameter is required'
            }), 400
        
        query = data['query']
        top_k = data.get('top_k', 3)
        
        if not query.strip():
            return jsonify({
                'error': 'Invalid query',
                'message': 'Query cannot be empty'
            }), 400
        
        # Get relevant documents
        relevant_docs = rag_system.retrieve_relevant_docs(query, top_k=top_k)
        
        # Generate response
        response_text = rag_system.generate_response(query, top_k=top_k)
        
        return jsonify({
            'query': query,
            'response': response_text,
            'retrieval_info': {
                'chunks_retrieved': len(relevant_docs),
                'top_matches': [
                    {
                        'similarity': doc['similarity'],
                        'role_level': doc['metadata']['role_level'],
                        'domain': doc['metadata']['domain'],
                        'question': doc['metadata']['question']
                    }
                    for doc in relevant_docs
                ]
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Query processing failed',
            'message': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_startup():
    """
    Model prediction endpoint for startup success prediction
    
    Request body (single startup):
    {
        "company_name": "AI Innovations Inc.",
        "founded_year": 2020,
        "industry": "Artificial Intelligence",
        "region": "North America",
        "funding_amount_usd": 50000000,
        "estimated_revenue_usd": 15000000,
        "employee_count": 100,
        "funding_round": "Series B",
        "co_investors": "Sequoia Capital,Andreessen Horowitz"
    }
    
    Request body (batch prediction):
    {
        "startups": [
            {...startup1...},
            {...startup2...}
        ]
    }
    """
    if not model_predictor:
        return jsonify({
            'error': 'Model predictor not available',
            'message': 'Model predictor failed to initialize'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body is required'
            }), 400
        
        # Check if batch or single prediction
        if 'startups' in data:
            # Batch prediction
            startups = data['startups']
            predictions = model_predictor.predict_batch(startups)
            
            return jsonify({
                'type': 'batch',
                'count': len(predictions),
                'predictions': predictions,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            # Single prediction
            prediction = model_predictor.predict_single(data)
            
            return jsonify({
                'type': 'single',
                'prediction': prediction,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid input',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Flask Server on http://localhost:{port}")
    print(f"{'='*80}")
    print(f"\n📍 Available Endpoints:")
    print(f"   GET  /health        - Health check")
    print(f"   POST /rag/query     - RAG managerial queries")
    print(f"   POST /predict       - Startup success prediction")
    print(f"\n{'='*80}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
