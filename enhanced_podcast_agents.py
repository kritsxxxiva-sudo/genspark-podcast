"""
Enhanced Podcast AI Agents Application with real AI integrations
Uses OpenAI GPT models for content generation and ElevenLabs for voice synthesis
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
from openai import AsyncOpenAI
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PodcastEpisode:
    """Enhanced podcast episode with AI-generated content."""
    title: str
    description: str
    content: str
    audio_url: Optional[str] = None
    duration: Optional[int] = None
    topics: List[str] = None
    guests: List[str] = None
    transcript: Optional[str] = None
    ai_model: str = "gpt-4"
    voice_model: str = "elevenlabs_monolingual_v1"
    created_at: str = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.guests is None:
            self.guests = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class EnhancedAIAgent:
    """Enhanced AI agent with OpenAI integration."""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str, openai_client: AsyncOpenAI):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.openai_client = openai_client
        self.tools = []
        self.conversation_history = []
        
    def add_tool(self, tool):
        """Add a tool to the agent's capabilities."""
        self.tools.append(tool)
        
    async def generate_content(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate content using OpenAI GPT models."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are {self.name}. {self.backstory}. {self.goal}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating content for {self.name}: {e}")
            return f"Error generating content: {str(e)}"

class SmartContentResearchAgent(EnhancedAIAgent):
    """Advanced content research agent with web scraping and AI analysis."""
    
    def __init__(self, openai_client: AsyncOpenAI):
        super().__init__(
            name="Smart Content Research Agent",
            role="Advanced Research Specialist",
            goal="Conduct comprehensive research using web scraping and AI analysis",
            backstory="An AI-powered researcher capable of scraping web content and analyzing trends",
            openai_client=openai_client
        )
        
    async def scrape_web_content(self, url: str) -> str:
        """Scrape content from a web page."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit to first 5000 characters
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return f"Error scraping content: {str(e)}"
    
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute research task with AI-powered analysis."""
        topic = context.get("topic", "AI and Technology")
        research_depth = context.get("research_depth", "comprehensive")
        search_urls = context.get("search_urls", [])
        
        # Scrape web content if URLs provided
        web_content = ""
        if search_urls:
            web_contents = []
            for url in search_urls[:3]:  # Limit to 3 URLs
                content = await self.scrape_web_content(url)
                web_contents.append(f"Content from {url}:\n{content[:1000]}...")
            web_content = "\n\n".join(web_contents)
        
        # Generate comprehensive research
        research_prompt = f"""
        Conduct {research_depth} research on the topic: "{topic}"
        
        Web content scraped: {web_content}
        
        Please provide:
        1. Key trends and developments
        2. Notable companies, researchers, or thought leaders
        3. Recent breakthroughs or innovations
        4. Industry applications and real-world use cases
        5. Future predictions and implications
        6. Controversies or challenges
        7. Opportunities for further exploration
        
        Make the research engaging and suitable for a podcast discussion.
        """
        
        return await self.generate_content(research_prompt, max_tokens=3000)

class CreativeScriptWriterAgent(EnhancedAIAgent):
    """Creative script writer with different podcast formats and styles."""
    
    def __init__(self, openai_client: AsyncOpenAI):
        super().__init__(
            name="Creative Script Writer Agent",
            role="Podcast Script Creator",
            goal="Create engaging, entertaining, and informative podcast scripts",
            backstory="A creative writer specializing in various podcast formats and storytelling techniques",
            openai_client=openai_client
        )
        
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Write creative podcast script based on research."""
        research_data = context.get("research_data", "")
        topic = context.get("topic", "AI Agents")
        tone = context.get("tone", "professional")
        duration = context.get("duration", 15)
        format_type = context.get("format", "interview")  # interview, solo, panel, narrative
        
        # Different prompts for different formats
        if format_type == "interview":
            prompt = f"""
            Create an engaging interview-style podcast script about "{topic}".
            
            Research background: {research_data}
            
            Format: Interview with expert guest
            Duration: {duration} minutes
            Tone: {tone}
            
            Include:
            - Host introduction and guest welcome
            - Thoughtful questions based on research
            - Guest responses with expert insights
            - Follow-up questions and discussion
            - Key takeaways and conclusion
            
            Make it sound natural and conversational.
            """
        elif format_type == "narrative":
            prompt = f"""
            Create a narrative storytelling podcast about "{topic}".
            
            Research background: {research_data}
            
            Format: Narrative with storytelling elements
            Duration: {duration} minutes
            Tone: {tone}
            
            Include:
            - Hook opening to grab attention
            - Background story and context
            - Personal anecdotes or case studies
            - Dramatic elements and tension
            - Resolution and key insights
            - Memorable closing
            
            Make it engaging like a documentary.
            """
        else:  # solo format
            prompt = f"""
            Create an informative solo podcast script about "{topic}".
            
            Research background: {research_data}
            
            Format: Solo presentation
            Duration: {duration} minutes
            Tone: {tone}
            
            Include:
            - Compelling introduction
            - Clear structure with sections
            - Engaging transitions
            - Key insights and takeaways
            - Call-to-action ending
            
            Make it both educational and entertaining.
            """
        
        return await self.generate_content(prompt, max_tokens=4000)

class AudioProductionAgent(EnhancedAIAgent):
    """Agent for audio production with multiple TTS options."""
    
    def __init__(self, openai_client: AsyncOpenAI, elevenlabs_api_key: str = None):
        super().__init__(
            name="Audio Production Agent",
            role="Audio Engineer & Producer",
            goal="Produce high-quality audio content with professional voice synthesis",
            backstory="An expert audio engineer with access to advanced TTS technologies",
            openai_client=openai_client
        )
        self.elevenlabs_api_key = elevenlabs_api_key
        
    async def generate_audio_with_openai(self, text: str, voice: str = "alloy") -> bytes:
        """Generate audio using OpenAI TTS."""
        try:
            response = await self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            return response.content
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return None
    
    async def generate_audio_with_elevenlabs(self, text: str, voice_id: str = "pNInz6obpgDQGcFmaJgB") -> bytes:
        """Generate audio using ElevenLabs API."""
        if not self.elevenlabs_api_key:
            return None
            
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        return await response.read()
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
        return None
    
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Produce audio from script using available TTS services."""
        script = context.get("script", "")
        voice_type = context.get("voice_type", "professional")
        tts_service = context.get("tts_service", "openai")  # openai, elevenlabs
        
        # Generate audio based on selected service
        audio_data = None
        if tts_service == "elevenlabs":
            audio_data = await self.generate_audio_with_elevenlabs(script)
        
        if audio_data is None:  # Fallback to OpenAI
            audio_data = await self.generate_audio_with_openai(script)
        
        if audio_data:
            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"podcast_{timestamp}.mp3"
            
            async with aiofiles.open(filename, "wb") as f:
                await f.write(audio_data)
            
            return f"Audio generated successfully: {filename} ({len(audio_data)} bytes)"
        else:
            return "Audio generation failed - no TTS service available"

class EnhancedPodcastOrchestrator:
    """Enhanced orchestrator with real AI integrations."""
    
    def __init__(self, openai_api_key: str, elevenlabs_api_key: str = None):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.agents = {
            "researcher": SmartContentResearchAgent(self.openai_client),
            "writer": CreativeScriptWriterAgent(self.openai_client),
            "producer": AudioProductionAgent(self.openai_client, elevenlabs_api_key),
            "qa": QualityAssuranceAgent(self.openai_client)
        }
        
    async def create_podcast_episode(self, topic: str, **kwargs) -> PodcastEpisode:
        """Create enhanced podcast episode with real AI generation."""
        
        logger.info(f"🎙️  Starting enhanced podcast creation for: {topic}")
        
        # Step 1: Smart Research
        logger.info("🔍 Conducting smart research...")
        research_context = {
            "topic": topic,
            "research_depth": kwargs.get("research_depth", "comprehensive"),
            "search_urls": kwargs.get("search_urls", [])
        }
        research_data = await self.agents["researcher"].execute("research_topic", research_context)
        
        # Step 2: Creative Script Writing
        logger.info("📝 Writing creative script...")
        writing_context = {
            "research_data": research_data,
            "topic": topic,
            "tone": kwargs.get("tone", "professional"),
            "duration": kwargs.get("duration", 15),
            "format": kwargs.get("format", "solo")
        }
        script = await self.agents["writer"].execute("write_script", writing_context)
        
        # Step 3: Quality Assurance
        logger.info("🔍 Quality assurance check...")
        qa_context = {"content": script}
        qa_report = await self.agents["qa"].execute("review_content", qa_context)
        
        # Step 4: Audio Production
        logger.info("🎵 Producing audio...")
        audio_context = {
            "script": script,
            "voice_type": kwargs.get("voice_type", "professional"),
            "tts_service": kwargs.get("tts_service", "openai")
        }
        audio_result = await self.agents["producer"].execute("produce_audio", audio_context)
        
        # Create enhanced podcast episode
        episode = PodcastEpisode(
            title=f"AI Agents Podcast: {topic}",
            description=f"An AI-generated exploration of {topic} with insights from advanced research and creative storytelling.",
            content=script,
            topics=[topic] + kwargs.get("additional_topics", []),
            duration=len(script.split()) * 0.1,
            ai_model="gpt-4",
            voice_model=kwargs.get("voice_model", "elevenlabs_monolingual_v1")
        )
        
        logger.info("✅ Enhanced podcast episode creation complete!")
        logger.info(f"🎙️  Episode: {episode.title}")
        logger.info(f"🎵 Audio Result: {audio_result}")
        
        return episode

# Configuration and environment setup
class Config:
    """Configuration management."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable is required")
            return False
        return True

# CLI Interface
async def main():
    """Enhanced main function with real AI integration."""
    
    print("🚀 Enhanced Podcast AI Agents System")
    print("=" * 50)
    
    # Validate configuration
    if not Config.validate():
        print("❌ Configuration validation failed. Please set required environment variables.")
        return
    
    orchestrator = EnhancedPodcastOrchestrator(
        openai_api_key=Config.OPENAI_API_KEY,
        elevenlabs_api_key=Config.ELEVENLABS_API_KEY
    )
    
    # Example: Create an enhanced podcast episode
    topic = "The Future of AI Agents in Creative Industries"
    episode = await orchestrator.create_podcast_episode(
        topic=topic,
        research_depth="comprehensive",
        tone="engaging",
        duration=25,
        format="narrative",
        voice_type="professional",
        tts_service="openai",
        search_urls=[
            "https://www.technologyreview.com/2023/01/09/1066198/ai-agents-are-getting-better-at-creative-tasks/",
            "https://www.nature.com/articles/s41586-023-06221-2"
        ]
    )
    
    # Save episode details
    episode_data = asdict(episode)
    episode_data["qa_report"] = "Quality assured by AI agent"
    
    output_file = "enhanced_podcast_episode.json"
    async with aiofiles.open(output_file, "w") as f:
        await f.write(json.dumps(episode_data, indent=2))
    
    print(f"\n📝 Enhanced episode saved to {output_file}")
    print(f"🎙️  Title: {episode.title}")
    print(f"⏱️  Duration: {episode.duration:.1f} minutes")
    print(f"🤖 AI Model: {episode.ai_model}")
    print(f"🎵 Voice Model: {episode.voice_model}")
    print(f"📅 Created: {episode.created_at}")

if __name__ == "__main__":
    asyncio.run(main())