import argparse
import os
import sys
from pathlib import Path
from typing import List
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
from PIL import Image

# Import application components
from src.config import setup_logging, get_settings
from src.services.api_service import APIService, APIError
from src.services.db_service import DBService, DBError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Face Database Generator - Build face embeddings database for testing'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input directory containing images to process (overrides IMAGES_DIR from .env)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    return parser.parse_args()


def validate_environment() -> None:
    """Validate that all required environment variables are set."""
    required_vars = [
        'API_URL',
        'MONGODB_URI', 
        'MONGODB_DB_NAME',
        'MONGODB_COLLECTION_NAME',
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing)}")
        print("Please check your .env file and try again.")
        sys.exit(1)


def save_detected_faces(image_path: Path, api_result: dict, output_dir: str = "others/generate/faces") -> None:
    """Extract and save detected faces from the image."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the original image
        original_image = Image.open(image_path)
        
        # Extract each detected face
        faces = api_result.get('result', [])
        for face_idx, face in enumerate(faces, 1):
            box = face.get('box', {})
            if not box or not all(k in box for k in ['x_min', 'x_max', 'y_min', 'y_max']):
                print(f"    ⚠️ Face {face_idx}: No bounding box coordinates")
                continue
            
            # Crop the face using bounding box
            face_crop = original_image.crop((
                int(box['x_min']),
                int(box['y_min']),
                int(box['x_max']),
                int(box['y_max'])
            ))
            
            # Save the face
            face_filename = f"{image_path.stem}_face{face_idx}.jpg"
            face_path = os.path.join(output_dir, face_filename)
            face_crop.save(face_path, "JPEG", quality=90)
            
            print(f"    💾 Saved face {face_idx}: {face_filename}")
            
    except Exception as e:
        print(f"    ⚠️ Error saving faces: {str(e)}")


def process_single_image(api_service: APIService, db_service: DBService, image_path: Path) -> dict:
    """
    Process a single image and store embeddings in database.
    
    Returns:
        dict: Processing result with stats
    """
    result = {
        'success': False,
        'filename': image_path.name,
        'faces_found': 0,
        'embeddings_stored': 0,
        'error': None
    }
    
    try:
        print(f"📸 Processing: {image_path.name}")
        
        # Call API to get embeddings
        api_result = api_service.scan_image(image_path)
        if not api_result or 'result' not in api_result:
            result['error'] = "Invalid API response"
            return result
        
        faces_found = len(api_result.get('result', []))
        result['faces_found'] = faces_found
        
        if faces_found == 0:
            print(f"  ⚠️ No faces found in {image_path.name}")
            result['success'] = True
            return result
        
        # 🆕 NEW: Save detected faces
        save_detected_faces(image_path, api_result)
        
        # Store each face embedding
        embeddings_stored = 0
        for face in api_result['result']:
            if face.get('embedding'):
                doc = {
                    'filename': image_path.name,
                    'file_path': str(image_path),
                    'processed_at': datetime.utcnow(),
                    'embedding': face.get('embedding'),
                    'box': face.get('box', {}),
                    'execution_time': face.get('execution_time', {})
                }
                
                people_id = db_service.store_result(doc)
                embeddings_stored += 1
                print(f"  ✅ Stored embedding {embeddings_stored}: {people_id}")
        
        result['embeddings_stored'] = embeddings_stored
        result['success'] = True
        
        print(f"  📊 Found {faces_found} face(s), stored {embeddings_stored} embedding(s)")
        
    except APIError as e:
        result['error'] = f"API error: {str(e)}"
        print(f"  ❌ API error: {str(e)}")
    except DBError as e:
        result['error'] = f"Database error: {str(e)}"
        print(f"  ❌ Database error: {str(e)}")
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        print(f"  ❌ Error: {str(e)}")
    
    return result


def scan_directory_for_images(directory: Path, supported_extensions: List[str]) -> List[Path]:
    """Scan directory for image files with deduplication."""
    print(f"🔍 Scanning for images in: {directory}")
    
    image_files_set = set()
    for ext in supported_extensions:
        found_files = directory.rglob(f"*{ext}")
        for file_path in found_files:
            resolved_path = file_path.resolve()
            image_files_set.add(resolved_path)
    
    image_files = sorted(list(image_files_set))
    print(f"📂 Found {len(image_files)} unique images")
    
    return image_files


def main() -> int:
    """Main function for database generation."""
    start_time = datetime.now()
    
    # Parse arguments
    args = parse_arguments()
    
    # Load environment
    load_dotenv('.env')
    validate_environment()
    
    # Get settings
    settings = get_settings()
    if args.log_level:
        settings.LOG_LEVEL = args.log_level
    
    # Setup logging
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        log_file='face_database_generator.log'
    )
    
    print("🏗️  Face Database Generator - Starting...")
    print(f"🎯 Building face database (clustering disabled)")
    print(f"💾 Detected faces will be saved to: others/generate/faces/")
    
    # Initialize services  
    api_service = APIService(settings=settings)
    db_service = DBService(settings=settings)
    
    # Determine input directory
    input_dir = Path(args.input or settings.IMAGES_DIR)
    if not input_dir.is_dir():
        print(f"❌ Error: Directory not found: {input_dir}")
        return 1
    
    try:
        # Scan for images
        image_files = scan_directory_for_images(input_dir, settings.SUPPORTED_EXTENSIONS)
        
        if not image_files:
            print("⚠️ No images found to process")
            return 0
        
        # Process images
        results = []
        processed_files = set()
        
        for i, image_path in enumerate(image_files, 1):
            # Skip duplicates within this run
            resolved_path = image_path.resolve()
            if resolved_path in processed_files:
                print(f"⏭️ Skipping {image_path.name} - already processed in this run")
                continue
            
            processed_files.add(resolved_path)
            
            print(f"\n📸 Processing image {i}/{len(image_files)}")
            result = process_single_image(api_service, db_service, image_path)
            results.append(result)
        
        # Generate summary
        total_processed = len(results)
        successful = len([r for r in results if r['success']])
        failed = total_processed - successful
        total_faces = sum(r['faces_found'] for r in results)
        total_embeddings = sum(r['embeddings_stored'] for r in results)
        
        processing_time = datetime.now() - start_time
        
        print(f"\n" + "="*60)
        print(f"🏗️  DATABASE GENERATION COMPLETE")
        print(f"="*60)
        print(f"📊 Images processed: {total_processed}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"👥 Total faces found: {total_faces}")
        print(f"💾 Total embeddings stored: {total_embeddings}")
        print(f"⏱️ Processing time: {processing_time}")
        print(f"📈 Average: {total_embeddings/total_processed:.1f} embeddings per image" if total_processed > 0 else "")
        print(f"💾 Face images saved to: others/generate/faces/")
        
        # Show database stats
        try:
            db_stats = db_service.get_stats()
            print(f"🗄️ Database total documents: {db_stats.get('total_documents', 'Unknown')}")
        except Exception as e:
            print(f"⚠️ Could not get database stats: {str(e)}")
        
        print(f"="*60)
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())