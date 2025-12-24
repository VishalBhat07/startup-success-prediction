# Startup Success Prediction - Flask Backend

Flask API backend for startup success prediction using ML and RAG system.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)

```bash
export GEMINI_API_KEY="your-api-key-here"
export FLASK_DEBUG="True"
export PORT="5000"
```

### 3. Run the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

## 📍 API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and services are available.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-12-24T10:30:00",
  "services": {
    "rag_system": "available",
    "model_predictor": "available"
  },
  "message": "Startup Success Prediction API is running"
}
```

### 2. RAG Managerial Query

**POST** `/rag/query`

Get AI-powered managerial advice using RAG system.

**Request:**

```json
{
  "query": "How to improve team productivity?",
  "top_k": 3
}
```

**Response:**

```json
{
  "query": "How to improve team productivity?",
  "response": "**Problem Explanation**\n...\n**Steps to Solve**\n...",
  "retrieval_info": {
    "chunks_retrieved": 3,
    "top_matches": [
      {
        "similarity": 0.85,
        "role_level": "Manager",
        "domain": "Operations",
        "question": "..."
      }
    ]
  },
  "timestamp": "2025-12-24T10:30:00"
}
```

### 3. Startup Success Prediction

**POST** `/predict`

Predict startup success probability.

#### Single Prediction

**Request:**

```json
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
```

**Response:**

```json
{
  "type": "single",
  "prediction": {
    "company_name": "AI Innovations Inc.",
    "predicted_success": true,
    "success_probability": 0.87,
    "prediction_label": "High Value (Top 25%)",
    "confidence_level": "Very High",
    "input_data": {...}
  },
  "timestamp": "2025-12-24T10:30:00"
}
```

#### Batch Prediction

**Request:**

```json
{
  "startups": [
    {
      "company_name": "Startup 1",
      ...
    },
    {
      "company_name": "Startup 2",
      ...
    }
  ]
}
```

**Response:**

```json
{
  "type": "batch",
  "count": 2,
  "predictions": [
    {...prediction1...},
    {...prediction2...}
  ],
  "timestamp": "2025-12-24T10:30:00"
}
```

## 📦 Project Structure

```
startup-success-prediction/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Model/
│   ├── model_utils.py             # Model prediction utilities
│   ├── startup_success_model.pkl  # Trained ML model
│   ├── model_config.json          # Model configuration
│   └── predict_model.ipynb        # Original prediction notebook
└── RAG/
    ├── rag_managerial_system.py   # RAG system implementation
    └── managerial_dataset.json    # RAG knowledge base
```

## 🧪 Testing the API

### Using cURL

**Health Check:**

```bash
curl http://localhost:5000/health
```

**RAG Query:**

```bash
curl -X POST http://localhost:5000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to scale a startup team?"}'
```

**Prediction:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Test Startup",
    "founded_year": 2020,
    "industry": "Technology",
    "region": "North America",
    "funding_amount_usd": 5000000,
    "estimated_revenue_usd": 1000000,
    "employee_count": 50,
    "funding_round": "Series A",
    "co_investors": "Sequoia Capital"
  }'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# RAG query
response = requests.post('http://localhost:5000/rag/query', json={
    'query': 'How to improve team productivity?'
})
print(response.json())

# Prediction
response = requests.post('http://localhost:5000/predict', json={
    'company_name': 'AI Innovations Inc.',
    'founded_year': 2020,
    'industry': 'Artificial Intelligence',
    'region': 'North America',
    'funding_amount_usd': 50000000,
    'estimated_revenue_usd': 15000000,
    'employee_count': 100,
    'funding_round': 'Series B',
    'co_investors': 'Sequoia Capital,Andreessen Horowitz'
})
print(response.json())
```

## 🚨 Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable

Error response format:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

## 🔧 Configuration

- **Port**: Set via `PORT` environment variable (default: 5000)
- **Debug Mode**: Set via `FLASK_DEBUG` environment variable
- **API Key**: Set via `GEMINI_API_KEY` environment variable

## 🌐 Production Deployment

For production, use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📝 Notes

- Ensure model files (`startup_success_model.pkl`, `model_config.json`) exist in `Model/` directory
- Ensure RAG dataset (`managerial_dataset.json`) exists in `RAG/` directory
- API key is hardcoded in RAG system - consider using environment variables for production
