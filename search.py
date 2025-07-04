import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
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
        description='Face Search - Test new images against existing face database'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input directory containing images to search (overrides IMAGES_DIR from .env)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.85,
        help='Similarity threshold for matching (default: 0.85)'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=3,
        help='Maximum number of matches to show per face (default: 3)'
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


def save_detected_faces(image_path: Path, api_result: dict, output_dir: str = "others/search/faces") -> None:
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


def search_face_matches(db_service: DBService, embedding: List[float], 
                       threshold: float, limit: int) -> List[Dict[str, Any]]:
    """
    Search for similar faces in the database.
    
    Returns:
        List of matching faces with similarity scores (ALL scores, not just above threshold)
    """
    try:
        # 🆕 NEW: Get ALL results regardless of threshold to see actual scores
        all_matches = db_service.vector_search_people(
            target_embedding=embedding,
            similarity_threshold=0.0,  # Get ALL results
            limit=limit * 3  # Get more results to analyze
        )
        return all_matches
        
    except Exception as e:
        print(f"    ⚠️ Search error: {str(e)}")
        return []


def format_similarity_score(score: float, threshold: float) -> str:
    """Format similarity score with threshold status."""
    if score >= threshold:
        if score >= 0.95:
            return f"🔥 EXCELLENT ({score:.3f}) ✅ ABOVE THRESHOLD"
        elif score >= 0.90:
            return f"🎯 VERY HIGH ({score:.3f}) ✅ ABOVE THRESHOLD"
        elif score >= 0.85:
            return f"✅ HIGH ({score:.3f}) ✅ ABOVE THRESHOLD"
        else:
            return f"👍 GOOD ({score:.3f}) ✅ ABOVE THRESHOLD"
    else:
        # Below threshold - show with different indicator
        if score >= 0.80:
            return f"📊 CLOSE ({score:.3f}) ❌ BELOW THRESHOLD ({threshold})"
        elif score >= 0.70:
            return f"📉 MODERATE ({score:.3f}) ❌ BELOW THRESHOLD ({threshold})"
        else:
            return f"❌ LOW ({score:.3f}) ❌ BELOW THRESHOLD ({threshold})"


def process_single_image(api_service: APIService, db_service: DBService, 
                        image_path: Path, threshold: float, limit: int) -> Dict[str, Any]:
    """
    Process a single image and search for similar faces.
    
    Returns:
        dict: Processing result with search results
    """
    result = {
        'success': False,
        'filename': image_path.name,
        'faces_found': 0,
        'matches_above_threshold': 0,
        'total_comparisons': 0,
        'unique_faces': 0,
        'face_results': [],
        'error': None
    }
    
    try:
        print(f"\n🔍 Searching: {image_path.name}")
        
        # Call API to get embeddings
        api_result = api_service.scan_image(image_path)
        if not api_result or 'result' not in api_result:
            result['error'] = "Invalid API response"
            return result
        
        faces_found = len(api_result.get('result', []))
        result['faces_found'] = faces_found
        
        if faces_found == 0:
            print(f"  ⚠️ No faces detected in {image_path.name}")
            result['success'] = True
            return result
        
        # 🆕 NEW: Save detected faces
        save_detected_faces(image_path, api_result)
        
        print(f"  👥 Found {faces_found} face(s) - searching for matches...")
        
        # Search for each face
        for face_idx, face in enumerate(api_result['result'], 1):
            embedding = face.get('embedding')
            if not embedding:
                print(f"    👤 Face {face_idx}: No embedding available")
                continue
            
            print(f"    👤 Face {face_idx}: Searching...")
            
            # Search for similar faces (ALL scores)
            all_matches = search_face_matches(db_service, embedding, threshold, limit)
            
            # Separate matches above and below threshold
            matches_above = [m for m in all_matches if m['similarity_score'] >= threshold]
            matches_below = [m for m in all_matches if m['similarity_score'] < threshold]
            
            face_result = {
                'face_index': face_idx,
                'matches_above': matches_above,
                'matches_below': matches_below,
                'is_unique': len(matches_above) == 0
            }
            
            result['total_comparisons'] += len(all_matches)
            result['matches_above_threshold'] += len(matches_above)
            
            if matches_above:
                print(f"      🎯 MATCHES ABOVE THRESHOLD ({threshold}): {len(matches_above)}")
                
                for match_idx, match in enumerate(matches_above, 1):
                    similarity = match['similarity_score']
                    filename = match.get('filename', 'unknown')
                    similarity_desc = format_similarity_score(similarity, threshold)
                    
                    print(f"        {match_idx}. {similarity_desc} → {filename}")
            else:
                result['unique_faces'] += 1
                print(f"      🆕 NO MATCHES ABOVE THRESHOLD ({threshold})")
            
            # 🆕 NEW: Show matches below threshold for analysis
            if matches_below:
                print(f"      📊 SIMILAR FACES BELOW THRESHOLD: {len(matches_below)}")
                
                for match_idx, match in enumerate(matches_below[:3], 1):  # Show top 3 below threshold
                    similarity = match['similarity_score']
                    filename = match.get('filename', 'unknown')
                    similarity_desc = format_similarity_score(similarity, threshold)
                    
                    print(f"        {match_idx}. {similarity_desc} → {filename}")
                
                if len(matches_below) > 3:
                    print(f"        ... and {len(matches_below) - 3} more below threshold")
            
            result['face_results'].append(face_result)
        
        result['success'] = True
        
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
    print(f"📂 Found {len(image_files)} unique images to search")
    
    return image_files


def check_database_status(db_service: DBService) -> Dict[str, Any]:
    """Check if database has data to search against."""
    try:
        stats = db_service.get_stats()
        total_docs = stats.get('total_documents', 0)
        
        return {
            'has_data': total_docs > 0,
            'total_embeddings': total_docs,
            'last_processed': stats.get('last_processed')
        }
    except Exception as e:
        return {
            'has_data': False,
            'total_embeddings': 0,
            'error': str(e)
        }


def main() -> int:
    """Main function for face searching."""
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
        log_file='face_search.log'
    )
    
    print("🔍 Face Search - Starting...")
    print(f"🎯 Searching for faces (threshold: {args.threshold}, limit: {args.limit})")
    print(f"💾 Detected faces will be saved to: others/search/faces/")
    
    # Initialize services
    api_service = APIService(settings=settings)
    db_service = DBService(settings=settings)
    
    # Check database status
    print(f"📊 Checking database status...")
    db_status = check_database_status(db_service)
    
    if not db_status['has_data']:
        print(f"❌ Error: No face data found in database!")
        print(f"💡 Run 'python generate.py' first to build the face database")
        if 'error' in db_status:
            print(f"   Database error: {db_status['error']}")
        return 1
    
    print(f"✅ Database ready: {db_status['total_embeddings']} face embeddings available")
    
    # Determine input directory
    input_dir = Path(args.input or settings.IMAGES_DIR)
    if not input_dir.is_dir():
        print(f"❌ Error: Directory not found: {input_dir}")
        return 1
    
    try:
        # Scan for images
        image_files = scan_directory_for_images(input_dir, settings.SUPPORTED_EXTENSIONS)
        
        if not image_files:
            print("⚠️ No images found to search")
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
            result = process_single_image(api_service, db_service, image_path, 
                                        args.threshold, args.limit)
            results.append(result)
        
        # Generate summary
        total_processed = len(results)
        successful = len([r for r in results if r['success']])
        failed = total_processed - successful
        total_faces = sum(r['faces_found'] for r in results)
        total_matches_above = sum(r['matches_above_threshold'] for r in results)
        total_comparisons = sum(r['total_comparisons'] for r in results)
        total_unique = sum(r['unique_faces'] for r in results)
        
        processing_time = datetime.now() - start_time
        
        print(f"\n" + "="*60)
        print(f"🔍 FACE SEARCH COMPLETE")
        print(f"="*60)
        print(f"📊 Images processed: {total_processed}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"👥 Total faces analyzed: {total_faces}")
        print(f"🎯 Matches above threshold ({args.threshold}): {total_matches_above}")
        print(f"📊 Total similarity comparisons: {total_comparisons}")
        print(f"🆕 Unique faces (no matches above threshold): {total_unique}")
        print(f"📈 Above-threshold match rate: {(total_matches_above/total_faces*100):.1f}%" if total_faces > 0 else "N/A")
        print(f"⏱️ Processing time: {processing_time}")
        print(f"🎚️ Similarity threshold used: {args.threshold}")
        print(f"💾 Face images saved to: others/search/faces/")
        print(f"="*60)
        
        # Show detailed breakdown for failed images
        failed_images = [r for r in results if not r['success']]
        if failed_images:
            print(f"\n❌ Failed Images:")
            for result in failed_images:
                print(f"  • {result['filename']}: {result['error']}")
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Search interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())