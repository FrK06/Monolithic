from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

class RAGWorkflow:
    @staticmethod
    def create_system_prompt(
        latest_message: Union[HumanMessage, Dict], 
        conversation_history: List[str], 
        image_content: Optional[List[str]] = None,
        thread_id: Optional[str] = None
    ) -> str:
        """
        Create a system prompt with enhanced conversation memory for the LLM.
        
        Args:
            latest_message: The most recent message from the user
            conversation_history: Processed history of the conversation
            image_content: Optional list of image URLs or data
            thread_id: Optional conversation thread identifier
            
        Returns:
            A comprehensive system prompt with conversation context
        """
        # Detect if images are present in the conversation
        has_images = image_content is not None and len(image_content) > 0
        
        # Extract the latest query content
        latest_query = ""
        if isinstance(latest_message, HumanMessage):
            latest_query = latest_message.content
        elif isinstance(latest_message, dict) and "content" in latest_message:
            latest_query = latest_message["content"]
        
        # Base system prompt
        base_prompt = f"""You are a helpful assistant that can search the web, extract information from websites, communicate via SMS/phone calls, and work with images.

When asked about date, time, or day of week:
1. DO NOT use tools - respond with the current system date/time
2. Today's date is {datetime.now().strftime('%B %d, %Y')} and it's {datetime.now().strftime('%A')}

When asked about recent events, news, or releases:
1. ALWAYS use the search_web tool first with a date-specific search (include "2025" or "this week" or "latest")
2. Then use scrape_webpage to get the full content from the most recent relevant results
3. Only provide information from what you find in the actual search results
4. If you can't find current information, say "I need to search for the most up-to-date information about this" and then search again
5. Never reference old information or make assumptions about dates
6. Always include the source and date of the information you found

Conversation thread ID: {thread_id if thread_id else 'New conversation'}
"""

        # Add image-specific instructions if images are present
        if has_images:
            image_prompt = """
When working with images:
1. If asked to describe or analyze an image, use the analyze_image tool to get detailed information about image contents
2. If asked to create an image, use the generate_image tool with a detailed prompt
3. If asked to modify an image, use the process_image tool with appropriate parameters
4. When describing images, be comprehensive and detailed
5. You can see attached images directly and provide your analysis
"""
            base_prompt += image_prompt

        # Add tool usage guidance
        tools_prompt = """
For questions about:
- Recent AI/LLM releases: specifically search AI news websites and include "2025" in the search
- Current events: always include the current month and year in the search
- Technology updates: specifically look for news from this week or this month
- When users request notifications, use send_sms or make_call to follow up.
- When users want to hear information, use speak_text to convert your response to audio
- When users want images created, use generate_image to create visuals based on descriptions
"""
        base_prompt += tools_prompt

        # Add memory-specific instructions for better context retention
        memory_prompt = """
IMPORTANT MEMORY INSTRUCTIONS:
You have access to the FULL conversation history between you and the user.
When responding, always:
1. Reference and acknowledge previous parts of the conversation when appropriate
2. Maintain continuity by connecting to earlier topics and questions
3. Use the user's previously stated preferences or concerns in your responses
4. Show that you remember details the user has shared before
5. Don't act surprised by information that was previously discussed
6. If the user previously shared images, remember what they showed and refer back to them if relevant
7. Keep track of what visual information you've already seen and analyzed
"""
        base_prompt += memory_prompt

        # Add multimodal context handling
        multimodal_context = """
In this conversation, you may encounter:
- Text messages
- Images shared by the user
- Voice transcripts
- Results from web searches

Maintain a coherent thread across all these modalities.
"""
        base_prompt += multimodal_context

        # Add conversation context with better formatting
        context_prompt = f"""
Current question: {latest_query}

IMPORTANT - Previous conversation history:
You must use this conversation history to maintain context when replying.
Reference prior exchanges when answering and maintain continuity of thought.
"""
        
        # Add a better-formatted conversation history with clear delineation
        if conversation_history and len(conversation_history) > 0:
            context_prompt += "\n--- CONVERSATION HISTORY START ---\n"
            context_prompt += "\n".join(conversation_history)
            context_prompt += "\n--- CONVERSATION HISTORY END ---\n"
            
            # Add explicit reminder to use the history
            context_prompt += "\nRemember to reference and build upon this conversation history in your response."
        else:
            context_prompt += "\nThis is the start of a new conversation."
        
        base_prompt += context_prompt

        return base_prompt

    @staticmethod
    def process_messages(messages: List) -> List[str]:
        """
        Process a list of messages into a format suitable for the system prompt.
        
        Args:
            messages: List of message objects from the conversation
            
        Returns:
            A list of formatted message strings
        """
        conversation_history = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                # Handle multimodal content in HumanMessage
                if isinstance(msg.content, list):
                    # Extract text parts only for the conversation history
                    text_parts = [item.get("text", "") for item in msg.content if isinstance(item, dict) and item.get("type") == "text"]
                    content = " ".join(text_parts)
                else:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                
                # Add timestamp if available
                timestamp = ""
                if hasattr(msg, "additional_kwargs") and "timestamp" in msg.additional_kwargs:
                    timestamp = f" [{msg.additional_kwargs['timestamp']}]"
                    
                conversation_history.append(f"Human{timestamp}: {content}")
                
            elif isinstance(msg, AIMessage):
                content = msg.content if hasattr(msg, "content") else str(msg)
                
                # Add timestamp if available
                timestamp = ""
                if hasattr(msg, "additional_kwargs") and "timestamp" in msg.additional_kwargs:
                    timestamp = f" [{msg.additional_kwargs['timestamp']}]"
                    
                conversation_history.append(f"Assistant{timestamp}: {content}")
                
            elif isinstance(msg, ToolMessage):
                content = msg.content if hasattr(msg, "content") else str(msg)
                tool_name = msg.tool_call_id if hasattr(msg, "tool_call_id") else "unknown_tool"
                conversation_history.append(f"Tool [{tool_name}]: {content}")
                
            elif isinstance(msg, SystemMessage):
                content = msg.content if hasattr(msg, "content") else str(msg)
                conversation_history.append(f"System: {content}")
                
            elif isinstance(msg, dict):
                # Handle dictionary format messages (from API/database)
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                timestamp = f" [{msg.get('timestamp', '')}]" if "timestamp" in msg else ""
                
                if isinstance(content, list):
                    # Handle multimodal content in dictionary messages
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = " ".join(text_parts)
                
                if role == "user":
                    conversation_history.append(f"Human{timestamp}: {content}")
                elif role == "assistant":
                    conversation_history.append(f"Assistant{timestamp}: {content}")
                elif role == "tool":
                    conversation_history.append(f"Tool{timestamp}: {content}")
                elif role == "system":
                    conversation_history.append(f"System{timestamp}: {content}")
                else:
                    conversation_history.append(f"{role.capitalize()}{timestamp}: {content}")
        
        return conversation_history
