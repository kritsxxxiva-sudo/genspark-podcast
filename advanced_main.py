"""
Advanced FastAPI application for podcast AI agents with all sophisticated features
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from enum import Enum
import logging

# Import advanced agents
from advanced_podcast_agents import (
    AdvancedPodcastOrchestrator, AdvancedPodcastEpisode, ListenerProfile,
    PodcastFormat, ContentTone, AgentRole
)
from enhanced_podcast_agents import EnhancedPodcastOrchestrator
from podcast_ai_agents import PodcastOrchestrator
from demo_podcast_agents import DemoOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Podcast AI Agents API",
    description="Sophisticated podcast generation with multi-agent AI, personalization, and analytics",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instances
advanced_orchestrator = None
enhanced_orchestrator = None
basic_orchestrator = None
demo_orchestrator = None

# Pydantic models for API requests/responses
class PodcastRequest(BaseModel):
    """Advanced podcast request model."""
    topic: str = Field(..., description="Main topic for the podcast")
    format: PodcastFormat = Field(default=PodcastFormat.SOLO, description="Podcast format")
    tone: ContentTone = Field(default=ContentTone.PROFESSIONAL, description="Content tone")
    duration: int = Field(default=900, ge=300, le=3600, description="Duration in seconds (5-60 minutes)")
    languages: List[str] = Field(default=["en"], description="Target languages")
    research_depth: str = Field(default="comprehensive", description="Research depth level")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    audio_effects: List[str] = Field(default=["normalization"], description="Audio processing effects")
    personalization_profile: Optional[Dict[str, Any]] = Field(default=None, description="Listener personalization profile")
    search_urls: List[str] = Field(default=[], description="URLs for research")
    additional_topics: List[str] = Field(default=[], description="Additional related topics")
    
    @validator('topic')
    def validate_topic(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Topic must be at least 5 characters long')
        return v.strip()

class PodcastResponse(BaseModel):
    """Advanced podcast response model."""
    episode_id: str
    title: str
    description: str
    format: str
    tone: str
    duration: int
    topics: List[str]
    languages: List[str]
    audio_effects: List[str]
    created_at: str
    status: str
    metadata: Dict[str, Any]
    personalization_score: Optional[float] = None
    analytics: Dict[str, float] = Field(default_factory=dict)

class ListenerRequest(BaseModel):
    """Listener profile request model."""
    id: Optional[str] = Field(default=None, description="Listener ID (auto-generated if not provided)")
    favorite_topics: List[str] = Field(default=[], description="Listener's favorite topics")
    preferred_duration: int = Field(default=900, ge=300, le=3600, description="Preferred duration in seconds")
    preferred_tone: ContentTone = Field(default=ContentTone.PROFESSIONAL, description="Preferred content tone")
    language: str = Field(default="en", description="Primary language")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Additional preferences")

class ListenerResponse(BaseModel):
    """Listener profile response model."""
    id: str
    favorite_topics: List[str]
    preferred_duration: int
    preferred_tone: str
    language: str
    created_at: str
    engagement_metrics: Dict[str, float] = Field(default_factory=dict)

class AnalyticsRequest(BaseModel):
    """Analytics request model."""
    timeframe: str = Field(default="30d", regex=r"^(\d+d|all)$", description="Timeframe for analytics (e.g., '7d', '30d', 'all')")
    episode_ids: Optional[List[str]] = Field(default=None, description="Specific episode IDs to analyze")
    metrics: List[str] = Field(default=["engagement", "completion", "topics"], description="Metrics to include")

class AnalyticsResponse(BaseModel):
    """Analytics response model."""
    timeframe: str
    total_episodes: int
    average_engagement: float
    top_performing_topics: Dict[str, float]
    listener_insights: Dict[str, Any]
    generated_at: str

class TranslationRequest(BaseModel):
    """Translation request model."""
    episode_id: str
    target_languages: List[str]
    preserve_context: bool = Field(default=True, description="Preserve original context and meaning")

class TranslationResponse(BaseModel):
    """Translation response model."""
    episode_id: str
    translations: Dict[str, str]
    status: str
    completed_at: str

# Background task processing
async def process_podcast_creation(episode_id: str, request_data: Dict[str, Any]):
    """Background task for podcast creation."""
    try:
        logger.info(f"🔄 Processing podcast creation for episode: {episode_id}")
        
        # Get appropriate orchestrator based on mode
        mode = request_data.get("mode", "advanced")
        orchestrator = get_orchestrator(mode)
        
        if not orchestrator:
            logger.error(f"❌ Orchestrator not available for mode: {mode}")
            return
        
        # Create episode based on mode
        if mode == "advanced":
            episode = await orchestrator.create_advanced_episode(**request_data)
        elif mode == "enhanced":
            episode = await orchestrator.create_podcast_episode(**request_data)
        elif mode == "demo":
            episode = await orchestrator.create_podcast_episode(**request_data)
        else:
            logger.error(f"❌ Unknown mode: {mode}")
            return
        
        logger.info(f"✅ Successfully created episode: {episode.title}")
        
    except Exception as e:
        logger.error(f"❌ Error processing podcast creation for {episode_id}: {e}")

def get_orchestrator(mode: str):
    """Get appropriate orchestrator based on mode."""
    global advanced_orchestrator, enhanced_orchestrator, basic_orchestrator, demo_orchestrator
    
    if mode == "advanced":
        if not advanced_orchestrator:
            from advanced_podcast_agents import AdvancedPodcastOrchestrator
            config = {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "use_redis": False
            }
            advanced_orchestrator = AdvancedPodcastOrchestrator(config)
        return advanced_orchestrator
    
    elif mode == "enhanced":
        if not enhanced_orchestrator:
            from enhanced_podcast_agents import EnhancedPodcastOrchestrator
            enhanced_orchestrator = EnhancedPodcastOrchestrator(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY")
            )
        return enhanced_orchestrator
    
    elif mode == "demo":
        if not demo_orchestrator:
            from demo_podcast_agents import DemoOrchestrator
            demo_orchestrator = DemoOrchestrator()
        return demo_orchestrator
    
    else:
        return None

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Advanced Podcast AI Agents API",
        "version": "2.0.0",
        "description": "Sophisticated podcast generation with multi-agent AI, personalization, and analytics",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "create_podcast": "/api/podcasts/create",
            "get_episode": "/api/podcasts/{episode_id}",
            "create_listener": "/api/listeners/create",
            "get_recommendations": "/api/listeners/{listener_id}/recommendations",
            "analytics": "/api/analytics",
            "translate": "/api/translate"
        },
        "features": [
            "Multi-agent AI orchestration",
            "Advanced content personalization",
            "Real-time analytics and performance tracking",
            "Multilingual content generation",
            "Audio processing and effects",
            "Sentiment analysis",
            "CrewAI integration patterns"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "advanced_orchestrator": advanced_orchestrator is not None,
            "enhanced_orchestrator": enhanced_orchestrator is not None,
            "demo_orchestrator": demo_orchestrator is not None
        }
    }

@app.post("/api/podcasts/create", response_model=PodcastResponse)
async def create_podcast(request: PodcastRequest, background_tasks: BackgroundTasks):
    """Create a new podcast episode with advanced features."""
    try:
        episode_id = str(uuid.uuid4())
        
        # Prepare request data
        request_data = {
            "topic": request.topic,
            "format": request.format,
            "tone": request.tone,
            "duration": request.duration,
            "languages": request.languages,
            "research_depth": request.research_depth,
            "include_sentiment": request.include_sentiment,
            "audio_effects": request.audio_effects,
            "search_urls": request.search_urls,
            "additional_topics": request.additional_topics,
            "episode_id": episode_id,
            "mode": "advanced"
        }
        
        # Add personalization if provided
        if request.personalization_profile:
            request_data["listener_profile"] = ListenerProfile(**request.personalization_profile)
        
        # Add background task for processing
        background_tasks.add_task(process_podcast_creation, episode_id, request_data)
        
        # Return immediate response
        return PodcastResponse(
            episode_id=episode_id,
            title=f"Advanced Podcast: {request.topic}",
            description=f"AI-generated advanced podcast about {request.topic}",
            format=request.format.value,
            tone=request.tone.value,
            duration=request.duration,
            topics=[request.topic] + request.additional_topics,
            languages=request.languages,
            audio_effects=request.audio_effects,
            created_at=datetime.now().isoformat(),
            status="processing",
            metadata={
                "research_depth": request.research_depth,
                "include_sentiment": request.include_sentiment,
                "mode": "advanced"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating podcast: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create podcast: {str(e)}")

@app.get("/api/podcasts/{episode_id}", response_model=PodcastResponse)
async def get_episode(episode_id: str):
    """Get podcast episode details."""
    try:
        # Check if episode exists in any orchestrator
        orchestrators = [advanced_orchestrator, enhanced_orchestrator, demo_orchestrator]
        
        for orchestrator in orchestrators:
            if orchestrator and episode_id in getattr(orchestrator, 'episodes', {}):
                episode = orchestrator.episodes[episode_id]
                
                return PodcastResponse(
                    episode_id=episode.id,
                    title=episode.title,
                    description=episode.description,
                    format=episode.format.value,
                    tone=episode.tone.value,
                    duration=int(episode.duration),
                    topics=episode.topics,
                    languages=list(episode.multilingual_content.keys()) if hasattr(episode, 'multilingual_content') else ["en"],
                    audio_effects=episode.metadata.get("audio_effects", []) if hasattr(episode, 'metadata') else [],
                    created_at=episode.created_at,
                    status="completed",
                    metadata=getattr(episode, 'metadata', {}),
                    analytics=getattr(episode, 'analytics', {})
                )
        
        raise HTTPException(status_code=404, detail="Episode not found")
        
    except Exception as e:
        logger.error(f"❌ Error getting episode {episode_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get episode: {str(e)}")

@app.post("/api/listeners/create", response_model=ListenerResponse)
async def create_listener(request: ListenerRequest):
    """Create a new listener profile."""
    try:
        # Get advanced orchestrator
        orchestrator = get_orchestrator("advanced")
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
        # Create listener profile
        profile_data = {
            "id": request.id,
            "favorite_topics": request.favorite_topics,
            "preferred_duration": request.preferred_duration,
            "preferred_tone": request.preferred_tone,
            "language": request.language,
            "preferences": request.preferences
        }
        
        listener_id = orchestrator.create_listener_profile(profile_data)
        listener = orchestrator.listener_profiles[listener_id]
        
        return ListenerResponse(
            id=listener.id,
            favorite_topics=listener.favorite_topics,
            preferred_duration=listener.preferred_duration,
            preferred_tone=listener.preferred_tone.value,
            language=listener.language,
            created_at=listener.created_at,
            engagement_metrics=listener.engagement_metrics
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating listener profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create listener profile: {str(e)}")

@app.get("/api/listeners/{listener_id}/recommendations")
async def get_recommendations(listener_id: str):
    """Get personalized episode recommendations for a listener."""
    try:
        orchestrator = get_orchestrator("advanced")
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
        recommendations = await orchestrator.generate_personalized_recommendations(listener_id)
        
        return {
            "listener_id": listener_id,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
            "total_available": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations for {listener_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.post("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics(request: AnalyticsRequest):
    """Get comprehensive analytics and performance metrics."""
    try:
        orchestrator = get_orchestrator("advanced")
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
        report = await orchestrator.get_performance_report(request.timeframe)
        
        return AnalyticsResponse(
            timeframe=report["timeframe"],
            total_episodes=report["total_episodes"],
            average_engagement=report["average_engagement"],
            top_performing_topics=report["top_performing_topics"],
            listener_insights=report.get("listener_insights", {}),
            generated_at=report["generated_at"]
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics: {str(e)}")

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_episode(request: TranslationRequest):
    """Translate podcast episode to multiple languages."""
    try:
        orchestrator = get_orchestrator("advanced")
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
        # Get episode
        if request.episode_id not in orchestrator.episodes:
            raise HTTPException(status_code=404, detail="Episode not found")
        
        episode = orchestrator.episodes[request.episode_id]
        
        # Generate translations
        translations = await orchestrator.agents["translation_expert"].generate_multilingual_episode(
            episode, request.target_languages
        )
        
        # Update episode with translations
        episode.multilingual_content.update(translations)
        
        return TranslationResponse(
            episode_id=request.episode_id,
            translations=translations,
            status="completed",
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error translating episode {request.episode_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to translate episode: {str(e)}")

# WebSocket support for real-time updates (advanced feature)
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except:
                self.active_connections.discard(connection)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time commands
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            elif message.get("type") == "subscribe":
                # Handle subscription to episode updates
                episode_id = message.get("episode_id")
                if episode_id:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "subscribed",
                            "episode_id": episode_id,
                            "status": "processing"
                        }),
                        websocket
                    )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting Advanced Podcast AI Agents API")
    
    # Initialize orchestrators based on available API keys
    global advanced_orchestrator, enhanced_orchestrator, demo_orchestrator
    
    # Always initialize demo orchestrator
    from demo_podcast_agents import DemoOrchestrator
    demo_orchestrator = DemoOrchestrator()
    
    logger.info("✅ API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down Advanced Podcast AI Agents API")
    
    # Close any open connections
    if manager.active_connections:
        await manager.broadcast(json.dumps({
            "type": "shutdown",
            "message": "Server is shutting down"
        }))

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "advanced_main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )