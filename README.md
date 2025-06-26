# FaceRecog 📸

## 📋 Overview

FaceRecog is a powerful tool for scanning images and storing facial recognition data in MongoDB. It captures detailed face embeddings, bounding boxes, and metadata for analysis and use.

## 🛠️ Requirements

- 🐍 Python 3.8+
- 🗄️ MongoDB instance (local or remote)
- 📦 Required packages (install via `pip install -r requirements.txt`)

## 🚀 Setup

1. 📦 Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. 📁 Copy the environment template and set your credentials:

    ```bash
    cp .env.example .env
    ```

    Update the `.env` file with your configuration:

    ```
    # 🌐 API Configuration
    API_URL=API URL
    API_MAX_RETRIES=3
    API_RETRY_DELAY=2.0

    # 🗄️ MongoDB Configuration
    MONGODB_URI=mongodb+srv://${MONGODB_USERNAME}:${MONGODB_PASSWORD}@cluster0.s35kdmn.mongodb.net/${MONGODB_DB_NAME}?retryWrites=true&w=majority&appName=Cluster0
    MONGODB_DB_NAME=your_database_name
    MONGODB_COLLECTION_NAME=your_collection_name
    MONGODB_USERNAME=your_username
    MONGODB_PASSWORD=your_secure_password

    # 📸 Image Processing
    IMAGES_DIR=images
    SUPPORTED_EXTENSIONS=.jpg,.jpeg,.png,.bmp,.gif

    # 📝 Logging
    LOG_LEVEL=INFO
    LOG_FORMAT=%(asctime)s - %(levelname)s - %(message)s
    ```

    ⚠️ **Important**: Update MongoDB credentials and sensitive values before use.

## 📚 Usage

### 📂 Scan a Directory of Images

```bash
python scan_and_store.py /path/to/images
```

The script uses configuration from your `.env` file. Override values using command line arguments.

### 🌐 API Endpoint

The API endpoint is configured via `API_URL` environment variable. Default:

```
http://47.129.240.165:3000/scan_faces
```

Override in `.env` file or using `--url` command line argument.

### 🗄️ MongoDB Storage

MongoDB configuration is handled via environment variables. Default values:

- Database: `phottam`
- Collection: `people`
- Connection URI: `mongodb+srv://${MONGODB_USERNAME}:${MONGODB_PASSWORD}@cluster0.s35kdmn.mongodb.net/${MONGODB_DB_NAME}`

Customize values by modifying your `.env` file.

## 📝 Example Document

A typical MongoDB document looks like:

```json
{
  "image": "path/or/frame",
  "faces": [
    {
      "x": 100,
      "y": 120,
      "w": 50,
      "h": 50,
      "embedding": [0.123, 0.456, ...]
    }
  ],
  "timestamp": "2023-01-01T12:00:00Z"
}
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
