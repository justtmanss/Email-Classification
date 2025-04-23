# Email Classification API

## Overview
This project provides an API for classifying emails into four categories: Incident, Request, Problem, and Change. It includes PII/PCI data masking capabilities to ensure sensitive information is protected before processing.

## Features
- Email classification into four categories using a trained SVM model
- PII/PCI data masking for sensitive information
- RESTful API endpoints for classification services
- Lightweight, containerized solution that works locally or in cloud environments

## Model Performance
The SVM model achieves the following performance metrics:

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Change   | 0.93      | 0.85   | 0.89     | 504     |
| Incident | 0.72      | 0.82   | 0.76     | 1917    |
| Problem  | 0.56      | 0.42   | 0.48     | 1007    |
| Request  | 0.92      | 0.94   | 0.93     | 1372    |

- **Overall Accuracy**: 0.77
- **Best CV Score**: 0.7587
- **Best Parameters**:
  - C: 1.0
  - max_features: 10000
  - ngram_range: (1, 1)

## Data Distribution
- Incident: 9,586 samples
- Request: 6,860 samples
- Problem: 5,037 samples
- Change: 2,517 samples

## Installation

### Using Docker
```bash
# Pull the image
docker pull justtmansss/email-classification-api

# Run the container
docker run -p 7860:7860 justtmansss/email-classification-api
```

### Manual Setup
```bash
# Clone the repository
git clone https://huggingface.co/spaces/justtmansss/email-classification-api

# Navigate to the project directory
cd email-classification-api

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 7860
```

## API Usage

### Classification Endpoint
```
POST /classify
```

#### Request Format
```json
{
  "email_text": "Your email content here..."
}
```

#### Response Format
```json
{
  "category": "Incident",
  "confidence": 0.85,
  "masked_text": "Your email with masked PII..."
}
```

### PII Masking Endpoint
```
POST /mask
```

#### Request Format
```json
{
  "text": "Text with PII to mask"
}
```

#### Response Format
```json
{
  "masked_text": "Text with masked PII"
}
```

## PII/PCI Masking

The API masks the following sensitive information:
- Full names
- Email addresses
- Phone numbers
- Date of birth
- Aadhar card numbers
- Credit/debit card numbers
- CVV numbers
- Expiry dates

## Technical Architecture

### Components
1. **Preprocessing Module**: Handles text cleaning and PII masking
2. **Classification Model**: SVM with TF-IDF vectorization
3. **API Layer**: FastAPI implementation of REST endpoints

### Technologies Used
- **Model**: Scikit-learn (SVM classifier with TF-IDF)
- **API Framework**: FastAPI
- **Containerization**: Docker
- **Deployment**: Hugging Face Spaces

## Development

### Model Training Process
1. Data preprocessing and PII masking
2. Feature extraction using TF-IDF
3. Grid search for hyperparameter optimization
4. Model evaluation and serialization

### Running Tests
```bash
# Run unit tests
pytest tests/
```

## Deployment

The API is deployed on Hugging Face Spaces and can be accessed at:
https://huggingface.co/spaces/justtmansss/email-classification-api

## License
[Specify your license here]

## Contact
[Your contact information]