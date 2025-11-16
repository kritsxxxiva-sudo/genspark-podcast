# Podcast AI Agents Application

## Overview
An advanced multi-agent system for automated podcast generation using AI agents. This FastAPI backend application demonstrates sophisticated multi-agent workflows for content creation, audio processing, and podcast generation.

## Project Status
- **Current State**: Successfully configured for Replit environment
- **Server**: Running on port 5000 (FastAPI + Uvicorn)
- **Mode**: Demo mode (API keys can be added for full functionality)

## Architecture
This is a **Python FastAPI backend application** with the following components:

### Core Files
- `main.py`: FastAPI web application with REST API endpoints
- `enhanced_podcast_agents.py`: Advanced AI agents with OpenAI integration
- `demo_podcast_agents.py`: Demo version without API keys
- `podcast_ai_agents.py`: Basic multi-agent system

### Multi-Agent System
1. **SmartContentResearchAgent**: Conducts web research and analysis
2. **CreativeScriptWriterAgent**: Writes engaging podcast scripts
3. **AudioProductionAgent**: Converts scripts to audio (TTS)
4. **QualityAssuranceAgent**: Reviews and validates content

### Directories
- `episodes/`: Generated podcast episodes (JSON)
- `audio/`: Audio files (MP3)
- `logs/`: Application logs
- `temp/`: Temporary files

## API Endpoints
- `GET /`: API information
- `GET /health`: Health check
- `POST /api/v1/generate`: Generate new podcast episode
- `GET /api/v1/status/{episode_id}`: Check episode status
- `GET /api/v1/episodes`: List all episodes
- `GET /api/v1/episodes/{episode_id}`: Get episode details
- `GET /api/v1/episodes/{episode_id}/download`: Download episode content
- `POST /api/v1/episodes/{episode_id}/regenerate`: Regenerate episode

## Environment Variables
Optional API keys for enhanced functionality:
- `OPENAI_API_KEY`: For GPT-4 content generation and OpenAI TTS
- `ELEVENLABS_API_KEY`: For premium voice synthesis (optional)

The application works in demo mode without API keys, but with limited functionality.

## Recent Changes (November 16, 2025)
- Configured for Replit environment
- Set up workflow to run FastAPI server on port 5000
- Installed Python 3.11 and all dependencies
- Fixed LSP errors and type issues
- Added missing QualityAssuranceAgent class
- Created necessary directories (episodes, audio, logs, temp)
- Updated .gitignore for Python project
- Server running successfully in demo mode

## Technology Stack
- **Framework**: FastAPI
- **Language**: Python 3.11
- **AI Integration**: OpenAI GPT-4, LangChain
- **Audio**: Text-to-speech (OpenAI, ElevenLabs)
- **Web Scraping**: BeautifulSoup, requests
- **Async**: aiohttp, aiofiles

## User Preferences
None specified yet.

## Next Steps
- Users can add API keys via the Secrets panel for full AI functionality
- Test the API endpoints using the `/docs` interactive documentation
- Generate podcast episodes on various topics
