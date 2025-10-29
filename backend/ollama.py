"""
Async Ollama Client for Streaming AI Responses

This is our bridge to the Ollama API. It handles:
- Streaming responses (so you see the AI "typing" in real-time)
- Regular responses (when you just want the full answer)
- Connection management (opening and closing properly)
- Error handling (because networks are unreliable!)

Built with async/await so it doesn't block your app.
"""

import json
import httpx
from typing import AsyncIterator
from backend.config import OLLAMA_BASE_URL, FIXED_MODEL


class AsyncOllamaLLM:
    """
    Talk to Ollama's AI models asynchronously.
    
    Why async? Because waiting for AI responses can take seconds, and we don't
    want your whole app frozen during that time. With async, other stuff can
    keep running while we wait for the AI.
    
    Example usage:
        # Set up the client
        llm = AsyncOllamaLLM(model="llama3.2", temperature=0.7)
        
        # Stream a response (tokens come as they're generated)
        async for chunk in llm.astream("Tell me a joke"):
            print(chunk, end="", flush=True)
        
        # Or get the full response at once
        answer = await llm.ainvoke("What's 2+2?")
        print(answer)
    """
    
    def __init__(
        self, 
        model: str = FIXED_MODEL, 
        base_url: str = OLLAMA_BASE_URL, 
        temperature: float = 0.7
    ):
        """
        Set up a connection to Ollama.
        
        Args:
            model: Which AI model to use (like "llama3.2")
            base_url: Where Ollama is running (usually localhost)
            temperature: How creative the AI should be (0 = boring, 2 = wild)
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        
        # Create a persistent connection (reused for multiple requests)
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream the AI's response word-by-word.
        
        This is what makes the "typing" effect in the UI! Instead of waiting
        for the complete response, we get chunks as they're generated.
        
        Args:
            prompt: What you're asking the AI
            
        Yields:
            Small pieces of text as they arrive
            
        Raises:
            ValueError: If something goes wrong (timeout, bad status, etc.)
        
        Example:
            response_area = st.empty()
            full_response = ""
            
            async for chunk in llm.astream("Write a poem"):
                full_response += chunk
                response_area.write(full_response + "▌")  # Blinking cursor
        """
        try:
            url = f"{self.base_url}/api/chat"
            
            # Build the request
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,  # This is the magic flag!
                "options": {"temperature": self.temperature}
            }
            
            # Open a streaming connection
            async with self.client.stream('POST', url, json=payload, timeout=180.0) as response:
                if response.status_code == 200:
                    # Process the stream line by line
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                # Each line is a JSON object
                                data = json.loads(line)
                                
                                # Extract the text chunk
                                if content := data.get("message", {}).get("content", ""):
                                    yield content
                                
                                # Check if we're done
                                if data.get("done", False):
                                    break
                                    
                            except json.JSONDecodeError:
                                # Sometimes we get malformed lines, just skip them
                                continue
                else:
                    raise ValueError(f"Ollama returned HTTP {response.status_code}")
                    
        except httpx.ReadTimeout:
            raise ValueError("The AI took too long to respond (timeout)")
        except httpx.ConnectError:
            raise ValueError("Can't connect to Ollama - is it running?")
        except Exception as e:
            raise ValueError(f"Something went wrong: {str(e)}")
    
    async def ainvoke(self, prompt: str) -> str:
        """
        Get the complete response all at once (no streaming).
        
        Use this when you need the full answer before doing anything with it.
        For example, if you need to parse the response or count words.
        
        Args:
            prompt: What you're asking the AI
            
        Returns:
            The complete response as a single string
            
        Raises:
            httpx.HTTPStatusError: If the request fails
            ValueError: If the response is malformed
        
        Example:
            summary = await llm.ainvoke("Summarize this in 3 words: ...")
            print(f"Summary: {summary}")
        """
        url = f"{self.base_url}/api/chat"
        
        # Build the request (no streaming this time)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        
        # Send and wait for complete response
        response = await self.client.post(url, json=payload, timeout=180.0)
        response.raise_for_status()
        
        # Pull out the answer
        return response.json().get("message", {}).get("content", "")
    
    async def close(self):
        """
        Close the connection properly.
        
        Always call this when you're done! It cleans up network resources
        and prevents memory leaks.
        
        Usually called during app shutdown:
            await llm.close()
        """
        await self.client.aclose()
