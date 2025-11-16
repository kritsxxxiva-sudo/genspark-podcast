"""
FastAPI Web Application for Podcast AI Agents
Provides REST API endpoints for podcast generation and management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import os
import json
import uuid
from datetime import datetime
import aiofiles
from pathlib import Path

# Import our enhanced agents
from enhanced_podcast_agents import (
    EnhancedPodcastOrchestrator, PodcastEpisode, Config
)

# Initialize FastAPI app
app = FastAPI(
    title="Podcast AI Agents API",
    description="AI-powered podcast generation using multi-agent systems",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class PodcastRequest(BaseModel):
    """Request model for podcast generation."""
    topic: str = Field(..., description="Main topic for the podcast episode")
    research_depth: str = Field("comprehensive", description="Research depth: basic, moderate, comprehensive")
    tone: str = Field("professional", description="Tone of the podcast: professional, casual, academic, entertaining")
    duration: int = Field(15, description="Target duration in minutes")
    format: str = Field("solo", description="Podcast format: solo, interview, narrative, panel")
    voice_type: str = Field("professional", description="Voice type for audio generation")
    tts_service: str = Field("openai", description="Text-to-speech service: openai, elevenlabs")
    search_urls: Optional[List[str]] = Field(None, description="Optional URLs for web research")
    additional_topics: Optional[List[str]] = Field(None, description="Additional related topics")

class PodcastResponse(BaseModel):
    """Response model for podcast generation."""
    episode_id: str
    title: str
    description: str
    topic: str
    duration: float
    status: str
    created_at: str
    audio_url: Optional[str] = None
    transcript_url: Optional[str] = None

class PodcastStatus(BaseModel):
    """Status model for podcast episodes."""
    episode_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    updated_at: str

# Global storage for episode status (in production, use Redis or database)
episode_status: Dict[str, PodcastStatus] = {}
episode_storage: Dict[str, PodcastEpisode] = {}

# Initialize orchestrator
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global orchestrator
    
    if Config.validate():
        orchestrator = EnhancedPodcastOrchestrator(
            openai_api_key=Config.OPENAI_API_KEY,
            elevenlabs_api_key=Config.ELEVENLABS_API_KEY
        )
        print("✅ Podcast AI Agents API initialized successfully")
    else:
        print("⚠️  Running in demo mode - some features may be limited")
        orchestrator = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Podcast AI Agents API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "generate_podcast": "/api/v1/generate",
            "get_status": "/api/v1/status/{episode_id}",
            "list_episodes": "/api/v1/episodes",
            "health_check": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "orchestrator_available": orchestrator is not None
    }

@app.post("/api/v1/generate", response_model=PodcastResponse)
async def generate_podcast(request: PodcastRequest, background_tasks: BackgroundTasks):
    """Generate a new podcast episode."""
    if not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not available. Please check API configuration."
        )
    
    episode_id = str(uuid.uuid4())
    
    # Initialize status
    episode_status[episode_id] = PodcastStatus(
        episode_id=episode_id,
        status="pending",
        progress=0,
        message="Podcast generation queued",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Start background processing
    background_tasks.add_task(process_podcast_generation, episode_id, request)
    
    return PodcastResponse(
        episode_id=episode_id,
        title=f"AI Generated: {request.topic}",
        description="Podcast episode is being generated...",
        topic=request.topic,
        duration=0,
        status="processing",
        created_at=datetime.now().isoformat()
    )

async def process_podcast_generation(episode_id: str, request: PodcastRequest):
    """Background task to generate podcast episode."""
    try:
        # Update status
        episode_status[episode_id].status = "processing"
        episode_status[episode_id].progress = 10
        episode_status[episode_id].message = "Starting podcast generation..."
        episode_status[episode_id].updated_at = datetime.now().isoformat()
        
        # Generate podcast
        episode = await orchestrator.create_podcast_episode(
            topic=request.topic,
            research_depth=request.research_depth,
            tone=request.tone,
            duration=request.duration,
            format=request.format,
            voice_type=request.voice_type,
            tts_service=request.tts_service,
            search_urls=request.search_urls or [],
            additional_topics=request.additional_topics or []
        )
        
        # Store episode
        episode_storage[episode_id] = episode
        
        # Update status
        episode_status[episode_id].status = "completed"
        episode_status[episode_id].progress = 100
        episode_status[episode_id].message = "Podcast episode generated successfully"
        episode_status[episode_id].updated_at = datetime.now().isoformat()
        
        # Save to file for persistence
        await save_episode_to_file(episode_id, episode)
        
    except Exception as e:
        logger.error(f"Error generating podcast {episode_id}: {e}")
        episode_status[episode_id].status = "failed"
        episode_status[episode_id].message = f"Generation failed: {str(e)}"
        episode_status[episode_id].updated_at = datetime.now().isoformat()

async def save_episode_to_file(episode_id: str, episode: PodcastEpisode):
    """Save episode to file for persistence."""
    episodes_dir = Path("episodes")
    episodes_dir.mkdir(exist_ok=True)
    
    episode_file = episodes_dir / f"{episode_id}.json"
    episode_data = {
        "episode_id": episode_id,
        **episode.__dict__,
        "saved_at": datetime.now().isoformat()
    }
    
    async with aiofiles.open(episode_file, "w") as f:
        await f.write(json.dumps(episode_data, indent=2, default=str))

@app.get("/api/v1/status/{episode_id}", response_model=PodcastStatus)
async def get_episode_status(episode_id: str):
    """Get the status of a podcast episode generation."""
    if episode_id not in episode_status:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    return episode_status[episode_id]

@app.get("/api/v1/episodes")
async def list_episodes():
    """List all generated podcast episodes."""
    episodes = []
    for episode_id, episode in episode_storage.items():
        episodes.append({
            "episode_id": episode_id,
            "title": episode.title,
            "topic": episode.topics[0] if episode.topics else "Unknown",
            "duration": episode.duration,
            "created_at": episode.created_at,
            "status": episode_status.get(episode_id, {}).status if episode_id in episode_status else "unknown"
        })
    
    return {
        "episodes": episodes,
        "total": len(episodes)
    }

@app.get("/api/v1/episodes/{episode_id}")
async def get_episode(episode_id: str):
    """Get detailed information about a specific episode."""
    if episode_id not in episode_storage:
        # Try to load from file
        episode = await load_episode_from_file(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        episode_storage[episode_id] = episode
    
    episode = episode_storage[episode_id]
    return {
        "episode_id": episode_id,
        "title": episode.title,
        "description": episode.description,
        "content": episode.content,
        "topics": episode.topics,
        "duration": episode.duration,
        "ai_model": episode.ai_model,
        "voice_model": episode.voice_model,
        "created_at": episode.created_at,
        "status": episode_status.get(episode_id, {}).status if episode_id in episode_status else "unknown"
    }

async def load_episode_from_file(episode_id: str) -> Optional[PodcastEpisode]:
    """Load episode from file."""
    episode_file = Path(f"episodes/{episode_id}.json")
    if not episode_file.exists():
        return None
    
    try:
        async with aiofiles.open(episode_file, "r") as f:
            data = json.loads(await f.read())
            
        return PodcastEpisode(
            title=data["title"],
            description=data["description"],
            content=data["content"],
            topics=data.get("topics", []),
            duration=data.get("duration"),
            ai_model=data.get("ai_model", "gpt-4"),
            voice_model=data.get("voice_model", "openai"),
            created_at=data.get("created_at")
        )
    except Exception as e:
        logger.error(f"Error loading episode {episode_id}: {e}")
        return None

@app.get("/api/v1/episodes/{episode_id}/download")
async def download_episode_content(episode_id: str):
    """Download episode content as a text file."""
    if episode_id not in episode_storage:
        episode = await load_episode_from_file(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        episode_storage[episode_id] = episode
    
    episode = episode_storage[episode_id]
    
    # Create temporary file with episode content
    temp_file = f"temp_episode_{episode_id}.txt"
    async with aiofiles.open(temp_file, "w") as f:
        await f.write(f"Title: {episode.title}\n")
        await f.write(f"Description: {episode.description}\n")
        await f.write(f"Created: {episode.created_at}\n")
        await f.write(f"Duration: {episode.duration} minutes\n")
        await f.write("=" * 50 + "\n")
        await f.write(episode.content)
    
    return FileResponse(
        temp_file,
        media_type="text/plain",
        filename=f"podcast_{episode_id}.txt"
    )

# Additional utility endpoints
@app.post("/api/v1/episodes/{episode_id}/regenerate")
async def regenerate_episode(episode_id: str, background_tasks: BackgroundTasks):
    """Regenerate an existing episode with new parameters."""
    if episode_id not in episode_storage:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    # Get original episode for reference
    original_episode = episode_storage[episode_id]
    
    # Create regeneration request based on original
    regenerate_request = PodcastRequest(
        topic=original_episode.topics[0] if original_episode.topics else "AI Technology",
        research_depth="comprehensive",
        tone="professional",
        duration=int(original_episode.duration) if original_episode.duration else 15
    )
    
    # Start regeneration in background
    background_tasks.add_task(process_podcast_generation, episode_id, regenerate_request)
    
    return {
        "message": "Episode regeneration started",
        "episode_id": episode_id,
        "status": "processing"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)