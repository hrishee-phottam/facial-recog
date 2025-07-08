"""
Production SQS Face Recognition System - Main Entry Point

This module serves as the production entry point for SQS-based face recognition processing.
It orchestrates the complete workflow: SQS → AI Check → API → Clustering → Storage → Thumbnails.
"""
import asyncio
import os
import signal
import sys
import logging
from datetime import datetime
from typing import Dict, Any

from src.config import setup_logging, initialize_settings
from src.core.processor import ImageProcessor
from src.core.console_observer import ConsoleObserver


class SQSFaceRecognitionApp:
    """Main application class for SQS-based face recognition processing."""
    
    def __init__(self):
        """Initialize the SQS face recognition application."""
        self.settings = None
        self.processor = None
        self.observer = None
        self.running = False
        self.shutdown_requested = False
        
        # Reset counter on startup
        from src.core.processor import ImageProcessor
        ImageProcessor._message_counter = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_name = "SIGNAL" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n🛑 Received {signal_name}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize(self):
        """Initialize all components."""
        try:
            print("🚀 Initializing SQS Face Recognition System...")
            
            # Initialize and validate settings
            self.settings = initialize_settings()
            
            # Setup logging
            setup_logging(
                log_level=self.settings.LOG_LEVEL,
                log_format=self.settings.LOG_FORMAT,
                log_file=self.settings.LOG_FILE
            )
            
            logger = logging.getLogger(__name__)
            logger.info("🎯 HYBRID FACE RECOGNITION SYSTEM - PRODUCTION MODE")
            logger.info("   System 2 Architecture + Production SQS Integration + Thumbnails")
            
            # Log configuration summary
            logger.info("🔧 CONFIGURATION:")
            logger.info(f"   📡 SQS Queue: {self.settings.SQS_QUEUE_URL.split('/')[-1]}")
            logger.info(f"   🗄️ MongoDB: {self.settings.MONGODB_DB_NAME}")
            logger.info(f"   🪣 S3 Bucket: {self.settings.WASABI_BUCKET}")
            logger.info(f"   🌐 API Endpoint: {self.settings.API_URL}")
            logger.info(f"   🧩 Clustering: {'✅ Enabled' if self.settings.ENABLE_FACE_CLUSTERING else '❌ Disabled'}")
            logger.info(f"   ⚡ Similarity Threshold: {self.settings.SIMILARITY_THRESHOLD}")
            logger.info(f"   📸 Thumbnails: ✅ Enabled")  # 🆕 NEW
            logger.info(f"   💾 Local Face Saving: {'✅ Enabled' if self.settings.SAVE_DETECTED_FACES_LOCALLY else '❌ Disabled'}")
            
            if self.settings.SAVE_DETECTED_FACES_LOCALLY:
                logger.info(f"   📁 Save Directory: {self.settings.LOCAL_FACES_DIR}")
            
            # Initialize console observer
            self.observer = ConsoleObserver(
                show_progress=self.settings.SHOW_PROGRESS,
                show_summary=self.settings.SHOW_SUMMARY
            )
            
            # Initialize enhanced processor with all services
            logger.info("🔧 Initializing services...")
            self.processor = ImageProcessor(settings=self.settings)
            self.processor.register_observer(self.observer)
            
            logger.info("✅ All components initialized successfully")
            logger.info("🚀 Ready to process SQS messages...")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            logging.error(f"❌ Initialization error: {str(e)}", exc_info=True)
            return False
    
    async def run_processing_loop(self):
        """Run the main SQS processing loop."""
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("🔄 Starting SQS message processing loop...")
            
            # Print startup summary
            self._print_startup_summary()
            
            # Initialize stats
            stats = {
                'start_time': datetime.utcnow(),
                'messages_processed': 0,
                'messages_successful': 0,
                'messages_failed': 0,
                'messages_skipped': 0,
                'total_faces_detected': 0,
                'total_faces_clustered': 0,
                'total_thumbnails_uploaded': 0  # 🆕 NEW: Track thumbnail uploads
            }
            
            # Main processing loop
            while not self.shutdown_requested:
                try:
                    # Get SQS service for queue monitoring
                    sqs_service = self.processor.sqs_service
                    
                    # Check queue status periodically
                    if stats['messages_processed'] % 20 == 0:  # Every 20 messages
                        try:
                            queue_stats = sqs_service.get_queue_stats()
                            if queue_stats.get('visible_messages', 0) > 0:
                                logger.info(f"📥 Queue status: {queue_stats['visible_messages']} messages waiting")
                        except:
                            pass  # Don't fail on queue stats
                    
                    # Receive and process messages
                    messages = await sqs_service.receive_messages()
                    
                    if not messages:
                        # No messages, brief pause
                        await asyncio.sleep(2)
                        continue
                    
                    logger.info(f"📨 Received {len(messages)} message(s) from SQS")
                    
                    # Process each message
                    for message in messages:
                        if self.shutdown_requested:
                            logger.info("🛑 Shutdown requested, stopping message processing")
                            break
                        
                        try:
                            # Process the SQS message
                            result = await self.processor.process_sqs_message(message)
                            
                            # Update stats
                            stats['messages_processed'] += 1
                            
                            if result.success:
                                if result.ai_enabled:
                                    stats['messages_successful'] += 1
                                    stats['total_faces_detected'] += result.faces_detected
                                    stats['total_faces_clustered'] += result.faces_clustered
                                    stats['total_thumbnails_uploaded'] += getattr(result, 'thumbnails_uploaded', 0)  # 🆕 NEW
                                    
                                    # Success log with detailed info
                                    filename = os.path.basename(result.s3_path)
                                    logger.info(f"✅ SUCCESS: {filename}")
                                    logger.info(f"   Media: {result.media_id}")
                                    logger.info(f"   Faces detected: {result.faces_detected}")
                                    logger.info(f"   Faces clustered: {result.faces_clustered}")
                                    logger.info(f"   Thumbnails uploaded: {getattr(result, 'thumbnails_uploaded', 0)}")  # 🆕 NEW
                                    logger.info(f"   Processing time: {result.processing_time:.1f}s")
                                    
                                    # Local face saving summary
                                    if result.face_saver_result and result.face_saver_result.get('faces_saved', 0) > 0:
                                        logger.info(f"   💾 Faces saved locally: {result.face_saver_result['faces_saved']}")
                                    
                                else:
                                    stats['messages_skipped'] += 1
                                    filename = os.path.basename(result.s3_path)
                                    logger.info(f"⏭️ SKIPPED: {filename} (AI disabled for event {result.event_id})")
                                
                                # Delete successful message
                                await sqs_service.delete_message(message['ReceiptHandle'])
                                
                            else:
                                stats['messages_failed'] += 1
                                error_msg = str(result.error) if result.error else "Unknown error"
                                filename = os.path.basename(result.s3_path) if result.s3_path != 'unknown' else 'unknown'
                                
                                logger.error(f"❌ FAILED: {filename}")
                                logger.error(f"   Media: {result.media_id}")
                                logger.error(f"   Error: {error_msg}")
                                logger.error(f"   Processing time: {result.processing_time:.1f}s")
                                
                                # Delete failed message (to prevent infinite retries)
                                await sqs_service.delete_message(message['ReceiptHandle'])
                            
                            # Print periodic summary
                            if stats['messages_processed'] % 10 == 0:
                                self._print_processing_stats(stats)
                            
                        except Exception as msg_error:
                            stats['messages_failed'] += 1
                            logger.error(f"❌ Message processing error: {str(msg_error)}", exc_info=True)
                            
                            # Delete problematic message
                            try:
                                await sqs_service.delete_message(message['ReceiptHandle'])
                            except:
                                pass
                    
                except Exception as loop_error:
                    logger.error(f"❌ Error in processing loop: {str(loop_error)}", exc_info=True)
                    await asyncio.sleep(5)  # Wait before retrying
            
            # Final summary
            stats['end_time'] = datetime.utcnow()
            self._print_final_summary(stats)
            
        except Exception as e:
            logger.error(f"❌ Fatal error in processing loop: {str(e)}", exc_info=True)
            raise
    
    def _print_startup_summary(self):
        """Print startup summary."""
        print("\n" + "="*80)
        print("🎯 HYBRID FACE RECOGNITION SYSTEM - PRODUCTION MODE")
        print("="*80)
        print(f"📡 SQS Queue: {self.settings.SQS_QUEUE_URL.split('/')[-1]}")
        print(f"🗄️ Database: {self.settings.MONGODB_DB_NAME}")
        print(f"🌐 API: {self.settings.API_URL}")
        print(f"🧩 Clustering: {'✅ Enabled' if self.settings.ENABLE_FACE_CLUSTERING else '❌ Disabled'}")
        print(f"📸 Thumbnails: ✅ Enabled")  # 🆕 NEW
        print(f"💾 Local Saving: {'✅ Enabled' if self.settings.SAVE_DETECTED_FACES_LOCALLY else '❌ Disabled'}")
        
        if self.settings.SAVE_DETECTED_FACES_LOCALLY:
            print(f"📁 Save Directory: {self.settings.LOCAL_FACES_DIR}")
        
        print(f"⚡ Similarity Threshold: {self.settings.SIMILARITY_THRESHOLD}")
        print("="*80)
        print("🚀 Ready to process SQS messages... (Ctrl+C to stop)")
        print("="*80 + "\n")
    
    def _print_processing_stats(self, stats: Dict[str, Any]):
        """Print periodic processing statistics."""
        elapsed = (datetime.utcnow() - stats['start_time']).total_seconds()
        rate = stats['messages_processed'] / elapsed if elapsed > 0 else 0
        
        logger = logging.getLogger(__name__)
        logger.info(f"📊 PROCESSING STATS:")
        logger.info(f"   Messages: {stats['messages_processed']} processed ({rate:.1f}/min)")
        logger.info(f"   Results: {stats['messages_successful']} success, {stats['messages_failed']} failed, {stats['messages_skipped']} skipped")
        logger.info(f"   Faces: {stats['total_faces_detected']} detected, {stats['total_faces_clustered']} clustered")
        logger.info(f"   Thumbnails: {stats['total_thumbnails_uploaded']} uploaded")  # 🆕 NEW
        logger.info(f"   Runtime: {elapsed/60:.1f} minutes")
    
    def _print_final_summary(self, stats: Dict[str, Any]):
        """Print final processing summary."""
        elapsed = (stats['end_time'] - stats['start_time']).total_seconds()
        
        logger = logging.getLogger(__name__)
        logger.info("🏁 PROCESSING COMPLETE - FINAL SUMMARY")
        logger.info(f"⏱️ Total Runtime: {elapsed/60:.1f} minutes")
        logger.info(f"📨 Messages Processed: {stats['messages_processed']}")
        logger.info(f"✅ Successful: {stats['messages_successful']}")
        logger.info(f"❌ Failed: {stats['messages_failed']}")
        logger.info(f"⏭️ Skipped (AI disabled): {stats['messages_skipped']}")
        logger.info(f"👥 Total Faces Detected: {stats['total_faces_detected']}")
        logger.info(f"🧩 Total Faces Clustered: {stats['total_faces_clustered']}")
        logger.info(f"📸 Total Thumbnails Uploaded: {stats['total_thumbnails_uploaded']}")  # 🆕 NEW
        
        if stats['messages_processed'] > 0:
            success_rate = (stats['messages_successful'] / stats['messages_processed']) * 100
            logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        
        if elapsed > 0:
            rate = stats['messages_processed'] / elapsed * 60
            logger.info(f"⚡ Processing Rate: {rate:.1f} messages/minute")
        
        # Face saver stats
        if self.processor.face_saver_service.is_enabled():
            try:
                saver_stats = self.processor.face_saver_service.get_stats()
                logger.info(f"💾 Faces Saved Locally: {saver_stats['faces_saved']}")
            except:
                pass
        
        # Clustering stats
        if self.processor.clustering_service:
            try:
                cluster_stats = self.processor.get_clustering_stats()
                if cluster_stats:
                    logger.info(f"🧩 Unique Faces Created: {cluster_stats['unique_faces']}")
                    logger.info(f"🔗 Clustering Rate: {cluster_stats['clustering_rate']}")
            except:
                pass
        
        print("\n" + "="*80)
        print("🏁 PROCESSING COMPLETE - FINAL SUMMARY")
        print("="*80)
        print(f"⏱️ Total Runtime: {elapsed/60:.1f} minutes")
        print(f"📨 Messages Processed: {stats['messages_processed']}")
        print(f"✅ Successful: {stats['messages_successful']}")
        print(f"❌ Failed: {stats['messages_failed']}")
        print(f"⏭️ Skipped (AI disabled): {stats['messages_skipped']}")
        print(f"👥 Total Faces Detected: {stats['total_faces_detected']}")
        print(f"🧩 Total Faces Clustered: {stats['total_faces_clustered']}")
        print(f"📸 Total Thumbnails Uploaded: {stats['total_thumbnails_uploaded']}")  # 🆕 NEW
        
        if stats['messages_processed'] > 0:
            success_rate = (stats['messages_successful'] / stats['messages_processed']) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if elapsed > 0:
            rate = stats['messages_processed'] / elapsed * 60
            print(f"⚡ Processing Rate: {rate:.1f} messages/minute")
        
        print("="*80)
        print("✨ Thank you for using the Hybrid Face Recognition System with Thumbnails!")
        print("="*80 + "\n")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("🧹 Cleaning up resources...")
            
            # Close database connections
            if self.processor and self.processor.db_service:
                self.processor.db_service.close()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {str(e)}")
    
    async def run(self):
        """Main run method."""
        try:
            # Initialize
            if not await self.initialize():
                return 1
            
            # Run processing loop
            await self.run_processing_loop()
            
            return 0
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
            return 0
        except Exception as e:
            print(f"\n❌ Fatal error: {str(e)}")
            logging.error(f"❌ Fatal error: {str(e)}", exc_info=True)
            return 1
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    app = SQSFaceRecognitionApp()
    exit_code = await app.run()
    return exit_code


def cli_main():
    """CLI entry point."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()