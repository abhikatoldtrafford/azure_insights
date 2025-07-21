# Review Insights - AI-Powered Customer Feedback Analysis

[![Build and deploy](https://github.com/yourusername/reviewinsights/actions/workflows/main_reviewinsights.yml/badge.svg)](https://github.com/yourusername/reviewinsights/actions/workflows/main_reviewinsights.yml)

## 🌐 Live Application

**Web Application**: https://reviewinsights.azurewebsites.net/

## 📋 Overview

Review Insights is an advanced AI-powered customer feedback analysis platform that leverages Azure OpenAI (GPT-4) to automatically analyze, categorize, and extract insights from customer feedback across multiple file formats. The platform identifies key problem areas, feature requests, and sentiment patterns to help businesses make data-driven decisions.

## ✨ Key Features

### 1. **Multi-Format File Support**
- **Direct Processing**: CSV, Excel (XLSX, XLS, XLSM)
- **Converted Processing**: PDF, DOCX, TXT, RTF, ODT
- **Automatic text extraction** from complex documents
- **Intelligent column detection** for structured data

### 2. **Advanced Analysis Capabilities**
- **Key Area Identification**: Automatically identifies 5-15 key problem areas/feature requests
- **Smart Classification**: Distinguishes between "issues" (problems/bugs) and "features" (requests/enhancements)
- **Sentiment Analysis**: Calculates sentiment scores (-1.0 to 1.0) for each feedback item
- **Insight Generation**: 
  - What users love most
  - Most requested features
  - Biggest pain points
  - Comprehensive overall summary

### 3. **Processing Modes**
- **Auto Mode** (Default): Enhanced processing with insight generation
- **CSV Mode**: Original CSV/Excel processing without enhancements
- **Completions Mode**: Force all processing through GPT-4 completions

### 4. **Single Text Classification**
- Classify any text (reviews, emails, meeting notes, transcripts)
- Multi-category classification (2-8 categories per text)
- Works with existing category frameworks
- Supports file uploads for text extraction

### 5. **Web Interface Features**
- Drag-and-drop file upload
- Live API testing interface
- Comprehensive API documentation
- Real-time results visualization
- Export capabilities

## 🚀 API Endpoints

### 1. **GET /** - Web Interface
Serves the interactive web application with three main tabs:
- Analyze Feedback
- Classify Review
- API Documentation

### 2. **POST /analyze-feedback** - Bulk Feedback Analysis

Analyzes customer feedback from uploaded files.

**Parameters:**
- `file` (required): File to analyze (CSV, Excel, PDF, DOCX, TXT)
- `return_raw_feedback` (optional, default: false): Include individual feedback items
- `source` (optional): Override source column value
- `extraction_prompt` (optional): Custom extraction instructions
- `mode` (optional, default: "auto"): Processing mode ("auto", "csv", "completions")

**Request Example (cURL):**
```bash
curl -X POST "https://reviewinsights.azurewebsites.net/analyze-feedback" \
  -F "file=@customer_feedback.csv" \
  -F "return_raw_feedback=true" \
  -F "source=App Store" \
  -F "mode=auto"
```

**Response Structure:**
```json
{
  "analysis_results": [
    {
      "key_area": "App Performance",
      "customer_problem": "App crashes frequently during extended usage sessions",
      "number_of_users": 45,
      "type": "issue",
      "raw_feedbacks": [
        {
          "Customer Feedback": "The app keeps crashing when I try to view reports",
          "sentiment_score": -0.8,
          "Received": "2024-01-15",
          "Name": "John Doe",
          "Source": "App Store"
        }
      ]
    }
  ],
  "summary": {
    "total_feedback_items": 250,
    "total_key_areas": 8
  },
  "insight_summary": {
    "user_loves": "Users appreciate the intuitive interface and real-time data updates",
    "feature_request": "Most requested feature is offline mode for accessing reports",
    "pain_point": "App performance issues and frequent crashes are the biggest concerns",
    "overall_summary": "The feedback shows a mix of appreciation for the interface and frustration with stability issues"
  },
  "metadata": {
    "job_id": "feedback_analysis_1234567890",
    "original_filename": "feedback.csv",
    "file_size_kb": 125.5,
    "processing_time_seconds": 8.3,
    "file_type": ".csv",
    "source": "App Store",
    "processing_method": "csv_direct",
    "mode": "auto"
  }
}
```

### 3. **POST /classify-single-review** - Single Text Classification

Classifies a single piece of text into multiple categories.

**Parameters:**
- `review` (required): Text to classify
- `existing_json` (optional): Existing categories for classification
- `file` (optional): File to extract text from (alternative to `review`)

**Request Example (JSON):**
```bash
curl -X POST "https://reviewinsights.azurewebsites.net/classify-single-review" \
  -H "Content-Type: application/json" \
  -d '{
    "review": "The app crashes every time I try to upload photos. Also, please add dark mode!",
    "existing_json": [
      {
        "key_area": "App Performance",
        "customer_problem": "App crashes frequently during usage"
      }
    ]
  }'
```

**Request Example (Form Data):**
```bash
curl -X POST "https://reviewinsights.azurewebsites.net/classify-single-review" \
  -F "review=The app crashes when uploading photos" \
  -F 'existing_json=[{"key_area":"App Performance","customer_problem":"App crashes frequently"}]'
```

**Response Structure:**
```json
[
  {
    "key_area": "App Performance",
    "customer_problem": "App crashes during photo upload operations",
    "sentiment_score": -0.85,
    "area_type": "issue"
  },
  {
    "key_area": "UI Customization",
    "customer_problem": "Users want dark mode theme option",
    "sentiment_score": 0.3,
    "area_type": "feature"
  }
]
```

## 🛠️ Technical Architecture

### Core Technologies
- **Framework**: FastAPI (Python)
- **AI/ML**: Azure OpenAI (GPT-4.1-mini model)
- **Embeddings**: text-embedding-3-small model
- **Document Processing**: Unstructured library + fallbacks
- **Data Processing**: Pandas, NumPy
- **Async Operations**: asyncio, aiohttp
- **Web Server**: Uvicorn

### Key Components
1. **Document Extraction** (`UnstructuredDocumentExtractor`)
   - Universal file content extraction
   - Multiple fallback strategies
   - Support for 20+ file formats

2. **Feedback Classification**
   - Embedding-based similarity matching
   - Type-aware classification (feature vs issue)
   - Sentiment analysis integration

3. **Insight Generation**
   - Direct content analysis
   - Pattern recognition
   - Summary generation

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Azure subscription (for deployment)
- OpenAI API access

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/reviewinsights.git
cd reviewinsights
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set environment variables:**
```bash
# Create .env file (not tracked in git)
AZURE_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_API_KEY="your-api-key"
AZURE_API_VERSION="2024-12-01-preview"
PORT=8081  # Optional, defaults to 8081
HOST=0.0.0.0  # Optional, defaults to 0.0.0.0
```

5. **Run the application:**
```bash
python review_classifier.py
```

The application will be available at:
- Web Interface: http://localhost:8081/
- API Documentation: http://localhost:8081/docs

### Azure Deployment

The application is configured for Azure App Service deployment via GitHub Actions.

1. **Azure Resources Required:**
   - Azure App Service (Python 3.10)
   - Azure OpenAI resource

2. **GitHub Secrets Required:**
   - `AZUREAPPSERVICE_PUBLISHPROFILE_*`: Azure publish profile

3. **Deployment:**
   - Push to `main` branch triggers automatic deployment
   - Manual deployment available via GitHub Actions workflow

## 📊 Data Processing Pipeline

### 1. File Upload & Validation
- File size limit: 50MB
- Automatic file type detection
- Encoding detection for text files

### 2. Content Extraction
- **Structured files** (CSV/Excel): Direct pandas processing
- **Unstructured files**: Convert to CSV via external API
- **Fallback**: Raw text extraction with multiple encoding attempts

### 3. Column Detection (for structured data)
- AI-powered column identification
- Fallback heuristics for common patterns
- Automatic standardization

### 4. Feedback Analysis
- Key area identification (5-15 areas)
- Embedding generation for all feedback
- Similarity-based classification
- Type-aware weighting (feature vs issue)

### 5. Insight Generation
- Sentiment analysis
- Pattern extraction
- Summary generation
- Metadata enrichment

## 🔧 Configuration Options

### Processing Modes
- **Auto Mode**: Best for general use, includes enhanced insights
- **CSV Mode**: For strict CSV/Excel processing only
- **Completions Mode**: Uses GPT-4 for entire pipeline

### File-Specific Handling
- **5-star reviews**: Configurable filtering (currently processes all)
- **Small datasets**: Adjusted thresholds for better results
- **Large files**: Chunked processing with progress tracking

## 📈 Performance Considerations

- **Timeout**: 180 seconds for large files
- **Batch Processing**: 300 feedback items per embedding batch
- **Rate Limiting**: Based on Azure OpenAI quotas
- **Concurrent Requests**: Limited to prevent overwhelming the API

## 🐛 Error Handling

The application includes comprehensive error handling:
- Automatic retries for API failures
- Fallback processing methods
- Graceful degradation
- Detailed error logging
- User-friendly error messages

## 📝 Best Practices

### Data Preparation
- **Column Headers**: Use clear, descriptive names
- **Feedback Column**: Should contain actual customer text
- **File Encoding**: UTF-8 preferred
- **Data Quality**: Remove duplicate entries

### API Usage
- **Batch Operations**: Use bulk analysis for multiple reviews
- **Caching**: Store analysis results for reuse
- **Error Handling**: Implement retry logic
- **Rate Limiting**: Respect API quotas

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: https://reviewinsights.azurewebsites.net/ (API Documentation tab)

## 🔮 Future Enhancements

- [ ] Real-time analysis streaming
- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] Advanced visualization dashboard
- [ ] Batch file processing
- [ ] API key management
- [ ] Historical trend analysis
- [ ] Export to various formats (JSON, Excel, PDF)

---

**Note**: This application uses Azure OpenAI services. Ensure you have appropriate credentials and comply with Azure's usage policies.
