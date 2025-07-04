import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from src.config import get_settings


class SQSError(Exception):
    """Custom exception for SQS-related errors."""
    pass


class SQSService:
    """Service for AWS SQS message operations."""
    
    def __init__(self, settings=None):
        """
        Initialize SQS service.
        
        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # SQS configuration
        self.queue_url = self.settings.SQS_QUEUE_URL
        self.poll_wait_time = self.settings.SQS_POLL_WAIT_TIME
        self.max_messages = self.settings.SQS_MAX_MESSAGES
        self.visibility_timeout = self.settings.SQS_VISIBILITY_TIMEOUT
        
        # Initialize SQS client
        self.sqs_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AWS SQS client."""
        try:
            self.sqs_client = boto3.client(
                'sqs',
                region_name=self.settings.AWS_REGION,
                aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY
            )
            
            # Test connection by getting queue attributes
            self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['QueueArn']
            )
            
            self.logger.info(f"SQS client initialized for queue: {self.queue_url}")
            
        except Exception as e:
            error_msg = f"Failed to initialize SQS client: {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
    
    async def receive_messages(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Receive messages from SQS queue.
        
        Args:
            max_messages: Maximum number of messages to receive (defaults to configured value)
            
        Returns:
            List[Dict]: List of received messages
            
        Raises:
            SQSError: If receiving messages fails
        """
        if not self.sqs_client:
            raise SQSError("SQS client not initialized")
        
        max_msgs = max_messages or self.max_messages
        
        try:
            self.logger.debug(f"Polling SQS queue for up to {max_msgs} messages...")
            
            # Use asyncio to run the synchronous SQS call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs_client.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=max_msgs,
                    WaitTimeSeconds=self.poll_wait_time,
                    VisibilityTimeout=self.visibility_timeout,
                    MessageAttributeNames=['All']
                )
            )
            
            messages = response.get('Messages', [])
            
            if messages:
                self.logger.info(f"Received {len(messages)} message(s) from SQS")
            else:
                self.logger.debug("No messages received from SQS")
            
            return messages
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"SQS client error ({error_code}): {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
            
        except BotoCoreError as e:
            error_msg = f"SQS service error: {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error receiving SQS messages: {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
    
    async def delete_message(self, receipt_handle: str) -> bool:
        """
        Delete a message from SQS queue.
        
        Args:
            receipt_handle: Receipt handle of the message to delete
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            SQSError: If deleting message fails
        """
        if not self.sqs_client:
            raise SQSError("SQS client not initialized")
        
        if not receipt_handle:
            raise SQSError("Receipt handle is required")
        
        try:
            # Use asyncio to run the synchronous SQS call
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sqs_client.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle
                )
            )
            
            self.logger.debug("Successfully deleted SQS message")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"SQS delete error ({error_code}): {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error deleting SQS message: {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
    
    async def change_message_visibility(self, receipt_handle: str, timeout: int) -> bool:
        """
        Change the visibility timeout of a message.
        
        Args:
            receipt_handle: Receipt handle of the message
            timeout: New visibility timeout in seconds
            
        Returns:
            bool: True if change was successful
            
        Raises:
            SQSError: If changing visibility fails
        """
        if not self.sqs_client:
            raise SQSError("SQS client not initialized")
        
        try:
            # Use asyncio to run the synchronous SQS call
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sqs_client.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=timeout
                )
            )
            
            self.logger.debug(f"Changed message visibility timeout to {timeout} seconds")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"SQS visibility change error ({error_code}): {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error changing message visibility: {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
    
    def parse_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse SQS message and extract relevant data.
        
        Args:
            message: Raw SQS message
            
        Returns:
            Dict: Parsed message data containing mediaId, path, eventId
            
        Raises:
            SQSError: If message parsing fails
        """
        try:
            # Extract message body
            body = message.get('Body', '{}')
            
            # Parse JSON body
            try:
                message_data = json.loads(body)
            except json.JSONDecodeError as e:
                raise SQSError(f"Invalid JSON in message body: {str(e)}")
            
            # Extract required fields
            media_id = message_data.get('mediaId')
            s3_path = message_data.get('path')
            event_id = message_data.get('eventId')
            
            # Validate required fields
            if not media_id:
                raise SQSError("Missing 'mediaId' in message")
            if not s3_path:
                raise SQSError("Missing 'path' in message")
            if not event_id:
                raise SQSError("Missing 'eventId' in message")
            
            # Extract additional metadata
            receipt_handle = message.get('ReceiptHandle')
            message_id = message.get('MessageId')
            
            if not receipt_handle:
                raise SQSError("Missing 'ReceiptHandle' in message")
            
            parsed_data = {
                'mediaId': media_id,
                'path': s3_path,
                'eventId': event_id,
                'receiptHandle': receipt_handle,
                'messageId': message_id,
                'receivedAt': datetime.utcnow(),
                'originalMessage': message_data
            }
            
            self.logger.debug(f"Parsed SQS message: mediaId={media_id}, path={s3_path}, eventId={event_id}")
            return parsed_data
            
        except SQSError:
            # Re-raise SQSError as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error parsing message: {str(e)}"
            self.logger.error(error_msg)
            raise SQSError(error_msg) from e
    
    def get_queue_attributes(self) -> Dict[str, Any]:
        """
        Get queue attributes for monitoring.
        
        Returns:
            Dict: Queue attributes
        """
        try:
            if not self.sqs_client:
                return {}
            
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=[
                    'ApproximateNumberOfMessages',
                    'ApproximateNumberOfMessagesNotVisible',
                    'ApproximateNumberOfMessagesDelayed'
                ]
            )
            
            return response.get('Attributes', {})
            
        except Exception as e:
            self.logger.error(f"Error getting queue attributes: {str(e)}")
            return {}
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get formatted queue statistics.
        
        Returns:
            Dict: Formatted queue statistics
        """
        try:
            attributes = self.get_queue_attributes()
            
            return {
                'queue_url': self.queue_url,
                'visible_messages': int(attributes.get('ApproximateNumberOfMessages', 0)),
                'invisible_messages': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                'delayed_messages': int(attributes.get('ApproximateNumberOfMessagesDelayed', 0)),
                'total_messages': (
                    int(attributes.get('ApproximateNumberOfMessages', 0)) +
                    int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)) +
                    int(attributes.get('ApproximateNumberOfMessagesDelayed', 0))
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting queue stats: {str(e)}")
            return {
                'queue_url': self.queue_url,
                'visible_messages': 0,
                'invisible_messages': 0,
                'delayed_messages': 0,
                'total_messages': 0,
                'error': str(e)
            }


# Global SQS service instance
_sqs_service = None


def get_sqs_service() -> SQSService:
    """
    Get or create the global SQS service instance.
    
    Returns:
        SQSService: The SQS service instance
    """
    global _sqs_service
    if _sqs_service is None:
        _sqs_service = SQSService()
    return _sqs_service