# Face Recognition System 🚀

A powerful tool for scanning images, detecting faces, and storing facial recognition data in MongoDB. Captures detailed face embeddings, bounding boxes, and metadata for analysis.

## ✨ Features

- 🖼️ Process multiple image formats (JPG, JPEG, PNG, BMP, GIF)
- 🔍 Detect and extract face embeddings
- 💾 Store results in MongoDB with detailed metadata
- 📊 Rich console output with progress tracking
- ⚙️ Configurable through environment variables
- 🔒 Secure API communication

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB instance (local or Atlas)
- API endpoint for face recognition

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd facial-recog
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Update `.env` with your configuration (see [Configuration](#-configuration) section)

### Usage

1. Place your images in the `images` directory or specify a custom path
2. Run the application:
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Configuration
API_URL=http://your-api-endpoint/scan_faces
API_MAX_RETRIES=3
API_RETRY_DELAY=2.0

# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster0.xxx.mongodb.net/dbname?retryWrites=true&w=majority
MONGODB_DB_NAME=your_database
MONGODB_COLLECTION_NAME=people

# Image Processing
IMAGES_DIR=images
SUPPORTED_EXTENSIONS=.jpg,.jpeg,.png,.bmp,.gif

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(levelname)s - %(message)s
```

### Command Line Arguments

```
usage: main.py [-h] [--images-dir IMAGES_DIR] [--api-url API_URL] [--mongodb-uri MONGODB_URI]
               [--db-name DB_NAME] [--collection COLLECTION] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Facial Recognition System

options:
  -h, --help            show this help message and exit
  --images-dir IMAGES_DIR
                        Directory containing images to process
  --api-url API_URL     Face recognition API endpoint
  --mongodb-uri MONGODB_URI
                        MongoDB connection URI
  --db-name DB_NAME     MongoDB database name
  --collection COLLECTION
                        MongoDB collection name
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level
```

## 🌐 API Integration

### Request Format

```python
import requests

url = "http://api-endpoint/scan_faces"
files = [
    ('file', ('image.jpg', open('image.jpg', 'rb'), 'application/octet-stream'))
]
response = requests.post(url, files=files)
```

### Response Format

```json
{
  "faces": [
    {
      "x": 100,
      "y": 120,
      "w": 50,
      "h": 50,
      "embedding": [0.123, 0.456, ...]
    }
  ],
  "execution_time": {
    "calculator": 49,
    "detector": 93
  }
}
```

## 🔒 Security

### Best Practices

1. **Never commit sensitive data**
   - Add `.env` to `.gitignore`
   - Use `.env.example` for documentation

3. **MongoDB Security**
   - Use strong authentication
   - Implement proper access controls
   - Regularly rotate credentials

4. **Input Validation**
   - Validate all API inputs
   - Sanitize file uploads
   - Implement rate limiting

## 📊 Data Model

### Faces Collection

```json
{
  "_id": ObjectId("..."),
  "image_path": "/path/to/image.jpg",
  "faces": [
    {
      "bounding_box": {
        "x": 100,
        "y": 120,
        "width": 50,
        "height": 50
      },
      "embedding": [0.123, 0.456, ...],
      "confidence": 0.98
    }
  ],
  "metadata": {
    "file_size": 123456,
    "file_type": "image/jpeg",
    "processing_time_ms": 150,
    "status": "processed"
  },
  "created_at": ISODate("2023-01-01T12:00:00Z"),
  "updated_at": ISODate("2023-01-01T12:00:00Z")
}
```

## 🛠️ Development

### Project Structure

```
facial-recog/
├── src/
│   ├── __init__.py
│   ├── config/           # Configuration management
│   ├── core/             # Core processing logic
│   ├── services/         # External service integrations
│   └── utils/            # Utility functions
├── tests/                # Test files
├── images/               # Default image directory
├── .env.example          # Example environment variables
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Running Tests

```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Face Recognition API](https://example.com) - For the face detection service
- [MongoDB](https://www.mongodb.com/) - For the database solution
- [Python](https://www.python.org/) - For being awesome!
```

## 📝 Notes

- 🗄️ Ensure MongoDB is running and accessible
- 🛠️ Adjust detection thresholds in the script for better accuracy
- 🌐 API endpoint must be active and reachable

## 🤝 Contributing

Open issues or submit pull requests for:
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- ⚡ Performance optimizations

## 📜 License

MIT License - see LICENSE file for details

## 📞 Support

For questions or issues, please open a GitHub issue or contact the maintainers.
