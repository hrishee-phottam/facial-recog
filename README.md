
# FaceRecog

## Overview

FaceRecog is a simple tool for scanning images or webcam feeds to detect faces and store the results in MongoDB. It captures face embeddings, bounding boxes, and other metadata for further analysis or use.

## Requirements

* Python 3.8+
* MongoDB instance (local or remote)

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy the environment template and set your credentials:

    ```bash
    cp .env.example .env
    ```

    Update the `.env` file with the following environment variables:

    ```
    # API Configuration
    API_URL=http://47.129.240.165:3000/scan_faces
    API_MAX_RETRIES=3
    API_RETRY_DELAY=2.0

    # MongoDB Configuration
    MONGODB_URI=mongodb+srv://${MONGODB_USERNAME}:${MONGODB_PASSWORD}@cluster0.s35kdmn.mongodb.net/${MONGODB_DB_NAME}?retryWrites=true&w=majority&appName=Cluster0
    MONGODB_DB_NAME=phottam
    MONGODB_COLLECTION_NAME=people
    MONGODB_USERNAME=phottam
    MONGODB_PASSWORD=jWZLNOwfTAQVo7vQ

    # Image Processing
    IMAGES_DIR=images
    SUPPORTED_EXTENSIONS=.jpg,.jpeg,.png,.bmp,.gif

    # Logging
    LOG_LEVEL=INFO
    LOG_FORMAT=%(asctime)s - %(levelname)s - %(message)s
    ```

    Note: Ensure you update the MongoDB credentials and any other sensitive values before use.

## Usage

### Scan a Directory of Images

```bash
python scan_and_store.py /path/to/images
```

The script will automatically use the configuration from your `.env` file. You can override specific values using command line arguments if needed.

### API Endpoint

The API endpoint is configured via the `API_URL` environment variable. It defaults to:

```
http://47.129.240.165:3000/scan_faces
```

You can override this value in your `.env` file or using the `--url` command line argument.

### MongoDB Storage

MongoDB configuration is handled via environment variables. The default values are:

- Database: `phottam`
- Collection: `people`
- Connection URI: `mongodb+srv://${MONGODB_USERNAME}:${MONGODB_PASSWORD}@cluster0.s35kdmn.mongodb.net/${MONGODB_DB_NAME}`

You can customize these values by modifying your `.env` file.

## Example Document

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

## Notes

* Ensure MongoDB is running and accessible.
* You may adjust detection thresholds or parameters in the script for better accuracy.
* API endpoint `http://47.129.240.165:3000/scan_faces` should be active and reachable.

## Contributing

Open issues or submit pull requests for improvements, bug fixes, or new features.


