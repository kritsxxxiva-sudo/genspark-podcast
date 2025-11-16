# Podcast AI Agents Application

## Overview
An advanced multi-agent system for automated podcast generation using AI agents. This full-stack application features a FastAPI backend with a modern web frontend for interactive podcast generation.

## Project Status
- **Current State**: Successfully configured for Replit environment with frontend
- **Server**: Running on port 5000 (FastAPI + Uvicorn)
- **Frontend**: Modern responsive web UI with real-time updates
- **Mode**: Demo mode (API keys can be added for full functionality)

## Architecture
This is a **full-stack application** with Python FastAPI backend and HTML/CSS/JavaScript frontend:

### Backend Files
- `main.py`: FastAPI web application with REST API endpoints
- `enhanced_podcast_agents.py`: Advanced AI agents with OpenAI integration
- `demo_podcast_agents.py`: Demo version without API keys
- `podcast_ai_agents.py`: Basic multi-agent system

### Frontend Files
- `static/index.html`: Main web interface
- `static/css/style.css`: Modern responsive styling
- `static/js/app.js`: JavaScript for API interactions and real-time updates

### Multi-Agent System
1. **SmartContentResearchAgent**: Conducts web research and analysis
2. **CreativeScriptWriterAgent**: Writes engaging podcast scripts
3. **AudioProductionAgent**: Converts scripts to audio (TTS)
4. **QualityAssuranceAgent**: Reviews and validates content

### Directories
- `static/`: Frontend static files (HTML, CSS, JS)
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
- Created necessary directories (episodes, audio, logs, temp, static)
- Updated .gitignore for Python project
- **Added modern frontend web interface**:
  - Responsive design with beautiful gradient UI
  - Interactive podcast generation form
  - Real-time episode status monitoring
  - Episodes list with details modal
  - API health indicator
  - Comprehensive form validation
- Server running successfully in demo mode with frontend

## Technology Stack
- **Framework**: FastAPI
- **Language**: Python 3.11
- **AI Integration**: OpenAI GPT-4, LangChain
- **Audio**: Text-to-speech (OpenAI, ElevenLabs)
- **Web Scraping**: BeautifulSoup, requests
- **Async**: aiohttp, aiofiles

## User Preferences
None specified yet.

## How to Use
1. **Access the Web Interface**: Open the application URL to see the frontend
2. **Generate a Podcast**: Fill in the podcast generation form and click "Generate Podcast"
3. **Monitor Progress**: Watch real-time status updates as your podcast is created
4. **View Episodes**: Check the "Recent Episodes" section to see all generated podcasts
5. **API Documentation**: Visit `/docs` for interactive API documentation

## Next Steps
- Users can add API keys via the Secrets panel for full AI functionality:
  - `OPENAI_API_KEY`: Required for GPT-4 content generation
  - `ELEVENLABS_API_KEY`: Optional for premium voice synthesis
- Test the API endpoints using the `/docs` interactive documentation
- Generate podcast episodes on various topics through the web interface
