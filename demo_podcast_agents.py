"""
Demo version of Podcast AI Agents that works without API keys
Uses mock AI responses to demonstrate the system architecture
"""

import asyncio
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MockPodcastEpisode:
    """Mock podcast episode for demonstration."""
    title: str
    description: str
    content: str
    audio_url: Optional[str] = None
    duration: Optional[int] = None
    topics: List[str] = None
    guests: List[str] = None
    transcript: Optional[str] = None
    ai_model: str = "mock-gpt-4"
    voice_model: str = "mock-elevenlabs"
    created_at: str = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.guests is None:
            self.guests = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class MockAIAgent:
    """Mock AI agent that simulates real AI responses."""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        
    def mock_generate_content(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate mock AI content based on prompt."""
        
        # Mock research content
        if "research" in prompt.lower() or "conduct" in prompt.lower():
            topic = self.extract_topic_from_prompt(prompt)
            return self.generate_mock_research(topic)
        
        # Mock script writing
        elif "script" in prompt.lower() or "podcast" in prompt.lower():
            topic = self.extract_topic_from_prompt(prompt)
            format_type = self.extract_format_from_prompt(prompt)
            return self.generate_mock_script(topic, format_type)
        
        # Mock quality assurance
        elif "quality" in prompt.lower() or "review" in prompt.lower():
            return self.generate_mock_qa_report()
        
        # Default mock response
        else:
            return f"Mock AI response from {self.name}: {prompt[:100]}..."
    
    def extract_topic_from_prompt(self, prompt: str) -> str:
        """Extract topic from prompt."""
        # Simple extraction - in real implementation would use NLP
        if "topic:" in prompt:
            start = prompt.find("topic:") + 6
            end = prompt.find("\n", start)
            return prompt[start:end].strip().strip('"')
        return "AI Technology"
    
    def extract_format_from_prompt(self, prompt: str) -> str:
        """Extract format type from prompt."""
        if "interview" in prompt.lower():
            return "interview"
        elif "narrative" in prompt.lower():
            return "narrative"
        else:
            return "solo"
    
    def generate_mock_research(self, topic: str) -> str:
        """Generate mock research content."""
        trends = [
            "Rapid advancement in multimodal AI capabilities",
            "Integration of AI agents into enterprise workflows",
            "Development of more autonomous and adaptive agents",
            "Improvements in multi-agent collaboration systems"
        ]
        
        companies = ["OpenAI", "Google DeepMind", "Anthropic", "Microsoft", "Meta AI"]
        applications = [
            "Customer service automation",
            "Content creation and editing",
            "Data analysis and insights",
            "Code generation and review"
        ]
        
        return f"""
# Research Report: {topic}

## Key Trends and Developments
{random.choice(trends)}
- Growing adoption in creative industries
- Enhanced reasoning capabilities
- Improved safety and alignment

## Notable Companies and Researchers
Leading organizations: {', '.join(random.sample(companies, 3))}
Key researchers: Dr. AI Researcher, Prof. Machine Learning, Dr. Neural Network

## Industry Applications
{chr(10).join(f"- {app}" for app in random.sample(applications, 2))}

## Future Predictions
- Widespread adoption across industries
- More sophisticated reasoning abilities
- Better human-AI collaboration
- Enhanced safety measures

## Challenges and Opportunities
- Ethical considerations and bias
- Computational requirements
- Integration complexity
- User adoption barriers

Research completed at: {datetime.now().isoformat()}
        """
    
    def generate_mock_script(self, topic: str, format_type: str) -> str:
        """Generate mock podcast script."""
        
        if format_type == "interview":
            return f"""
# Podcast Script: "Exploring {topic}"

[INTRO MUSIC]

HOST: Welcome to the AI Agents Podcast! Today we're discussing {topic} with our special guest, Dr. AI Expert.

GUEST: Thank you for having me! I'm excited to talk about {topic} and share some insights from my research.

HOST: Let's start with the basics. What exactly is {topic} and why is it important?

GUEST: Great question! {topic} represents a significant advancement in artificial intelligence because...

[SEGMENT 1: Background and Context]
- Historical development
- Current state of technology
- Key players and organizations

[SEGMENT 2: Technical Deep Dive]
- How it works under the hood
- Technical challenges and solutions
- Performance metrics and benchmarks

[SEGMENT 3: Real-World Applications]
- Industry use cases
- Success stories
- Lessons learned

[SEGMENT 4: Future Outlook]
- Upcoming developments
- Potential impact
- Research directions

[CONCLUSION]
HOST: Thank you, Dr. Expert, for sharing your insights on {topic}.

GUEST: My pleasure! It's been great discussing this exciting field.

[OUTRO MUSIC]
            """
        else:
            return f"""
# Podcast Script: "The World of {topic}"

[INTRO MUSIC]

HOST: Hello and welcome to another episode of the AI Agents Podcast! I'm your host, and today we're exploring the fascinating world of {topic}.

[SECTION 1: Introduction]
{topic} is transforming how we think about artificial intelligence and its applications. In this episode, we'll dive deep into what makes this technology so revolutionary.

[SECTION 2: Technical Overview]
At its core, {topic} involves complex algorithms and sophisticated neural networks that work together to create intelligent behavior. The key components include...

[SECTION 3: Practical Applications]
The real power of {topic} becomes apparent when we look at its applications across various industries. From healthcare to finance, from education to entertainment...

[SECTION 4: Challenges and Solutions]
Of course, implementing {topic} isn't without its challenges. Some of the key obstacles include computational requirements, data quality, and ethical considerations.

[SECTION 5: Future Prospects]
Looking ahead, the future of {topic} looks incredibly promising. Researchers are working on improving efficiency, reducing computational costs, and enhancing capabilities.

[CONCLUSION]
Thank you for joining me on this exploration of {topic}. I hope you found this episode informative and engaging.

[OUTRO MUSIC]
            """
    
    def generate_mock_qa_report(self) -> str:
        """Generate mock quality assurance report."""
        score = random.uniform(8.5, 9.8)
        return f"""
# Quality Assurance Report

## Content Analysis
- Length: {random.randint(800, 2000)} characters
- Word Count: {random.randint(200, 500)}
- Reading Level: Appropriate for target audience

## Quality Metrics
✅ Grammar and spelling: PASSED
✅ Technical accuracy: VERIFIED
✅ Content appropriateness: APPROVED
✅ Fact checking: COMPLETED
✅ Plagiarism check: ORIGINAL

## Recommendations
- Content is well-structured and engaging
- Technical details are accurate and current
- Suitable for podcast format
- Ready for publication

## Quality Score: {score:.1f}/10
Status: APPROVED FOR PUBLICATION
        """

class MockContentResearchAgent(MockAIAgent):
    """Mock content research agent."""
    
    def __init__(self):
        super().__init__(
            name="Mock Content Research Agent",
            role="Research Specialist",
            goal="Conduct comprehensive research on specified topics",
            backstory="An AI-powered researcher with access to vast knowledge"
        )
    
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute research task."""
        topic = context.get("topic", "AI and Technology")
        return self.mock_generate_content(f"research {topic}")

class MockScriptWriterAgent(MockAIAgent):
    """Mock script writer agent."""
    
    def __init__(self):
        super().__init__(
            name="Mock Script Writer Agent",
            role="Content Creator",
            goal="Create engaging and informative podcast scripts",
            backstory="A creative writer specializing in audio content"
        )
    
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Write podcast script."""
        research_data = context.get("research_data", "")
        topic = context.get("topic", "AI Technology")
        format_type = context.get("format", "solo")
        return self.mock_generate_content(f"write {format_type} script about {topic}")

class MockAudioProducerAgent(MockAIAgent):
    """Mock audio producer agent."""
    
    def __init__(self):
        super().__init__(
            name="Mock Audio Producer Agent",
            role="Audio Engineer",
            goal="Produce high-quality audio from scripts",
            backstory="An experienced audio engineer with TTS expertise"
        )
    
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Produce audio from script."""
        script = context.get("script", "")
        voice_type = context.get("voice_type", "professional")
        
        duration = len(script.split()) * 0.1
        return f"""
Audio Production Complete

Script processed: {len(script)} characters
Voice type: {voice_type}
Audio format: MP3, 44.1kHz, 128kbps
Estimated duration: {duration:.1f} minutes

Mock audio file created: podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3

Note: In production, this would integrate with:
- OpenAI Text-to-Speech API
- ElevenLabs API
- Google Cloud Text-to-Speech
- Amazon Polly
        """

class MockQualityAssuranceAgent(MockAIAgent):
    """Mock quality assurance agent."""
    
    def __init__(self):
        super().__init__(
            name="Mock Quality Assurance Agent",
            role="Content Reviewer",
            goal="Ensure content quality and accuracy",
            backstory="A meticulous reviewer ensuring high standards"
        )
    
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Review content quality."""
        content = context.get("content", "")
        return self.mock_generate_content(f"review quality of content")

class MockPodcastOrchestrator:
    """Mock orchestrator that demonstrates the workflow."""
    
    def __init__(self):
        self.agents = {
            "researcher": MockContentResearchAgent(),
            "writer": MockScriptWriterAgent(),
            "producer": MockAudioProducerAgent(),
            "qa": MockQualityAssuranceAgent()
        }
        
    async def create_podcast_episode(self, topic: str, **kwargs) -> MockPodcastEpisode:
        """Create a mock podcast episode demonstrating the workflow."""
        
        print(f"🎙️  Starting MOCK podcast creation for: {topic}")
        
        # Step 1: Research Phase
        print("🔍 MOCK Research phase initiated...")
        research_context = {
            "topic": topic,
            "research_depth": kwargs.get("research_depth", "comprehensive")
        }
        research_data = await self.agents["researcher"].execute("research_topic", research_context)
        
        # Step 2: Script Writing Phase
        print("📝 MOCK Script writing phase...")
        writing_context = {
            "research_data": research_data,
            "topic": topic,
            "format": kwargs.get("format", "solo")
        }
        script = await self.agents["writer"].execute("write_script", writing_context)
        
        # Step 3: Quality Review
        print("🔍 MOCK Quality assurance phase...")
        qa_context = {"content": script}
        qa_report = await self.agents["qa"].execute("review_content", qa_context)
        
        # Step 4: Audio Production
        print("🎵 MOCK Audio production phase...")
        audio_context = {
            "script": script,
            "voice_type": kwargs.get("voice_type", "professional")
        }
        audio_info = await self.agents["producer"].execute("produce_audio", audio_context)
        
        # Create podcast episode
        episode = MockPodcastEpisode(
            title=f"Mock Podcast: {topic}",
            description=f"A mock exploration of {topic} with simulated AI insights.",
            content=script,
            topics=[topic],
            duration=len(script.split()) * 0.1
        )
        
        print("✅ MOCK Podcast episode creation complete!")
        print(f"📊 Quality Score: Extracted from QA report")
        print(f"🎙️  Episode: {episode.title}")
        print(f"🎵 Audio Info: {audio_info[:100]}...")
        
        return episode

# Demo CLI Interface
async def demo_main():
    """Demo main function to showcase the system without API keys."""
    
    print("🎭 MOCK Podcast AI Agents System (Demo Mode)")
    print("=" * 50)
    print("This demo shows how the system works without requiring API keys.")
    print("In production, replace mock agents with real AI integrations.")
    print()
    
    orchestrator = MockPodcastOrchestrator()
    
    # Example topics for demonstration
    demo_topics = [
        "AI Agents and Multi-Agent Systems",
        "Machine Learning in Healthcare",
        "Climate Change and Technology Solutions",
        "Space Exploration Innovations",
        "Quantum Computing Breakthroughs"
    ]
    
    print("Available demo topics:")
    for i, topic in enumerate(demo_topics, 1):
        print(f"{i}. {topic}")
    
    # Use first topic for demo
    topic = demo_topics[0]
    print(f"\n🎯 Running demo with topic: {topic}")
    
    episode = await orchestrator.create_podcast_episode(
        topic=topic,
        research_depth="comprehensive",
        format="solo",
        voice_type="professional"
    )
    
    # Save episode details
    episode_data = {
        "title": episode.title,
        "description": episode.description,
        "content_preview": episode.content[:500] + "...",
        "topics": episode.topics,
        "duration": episode.duration,
        "created_at": episode.created_at,
        "ai_model": episode.ai_model,
        "voice_model": episode.voice_model,
        "demo_mode": True
    }
    
    # Save to JSON file
    with open("demo_podcast_episode.json", "w") as f:
        json.dump(episode_data, f, indent=2)
    
    print(f"\n📝 Demo episode saved to demo_podcast_episode.json")
    print(f"🎙️  Title: {episode.title}")
    print(f"⏱️  Estimated Duration: {episode.duration:.1f} minutes")
    print(f"📋 Topics: {', '.join(episode.topics)}")
    print(f"🤖 AI Model: {episode.ai_model} (Mock)")
    print(f"🎵 Voice Model: {episode.voice_model} (Mock)")
    
    print("\n🎉 Demo completed! Check the generated files to see the output.")
    print("To run with real AI, set up your API keys in the .env file.")

if __name__ == "__main__":
    asyncio.run(demo_main())