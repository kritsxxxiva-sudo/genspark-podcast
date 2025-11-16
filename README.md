# 🎙️ Podcast AI Agents Application

An advanced multi-agent system for automated podcast generation using AI agents, inspired by the comprehensive analysis of 500+ AI agent projects. This application demonstrates sophisticated multi-agent workflows for content creation, audio processing, and podcast generation.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Architecture**: Research, writing, production, and quality assurance agents
- **AI-Powered Content Generation**: Uses OpenAI GPT models for intelligent content creation
- **Audio Production**: Text-to-speech conversion with multiple voice options
- **Web Research Integration**: Scrapes and analyzes web content for research
- **Quality Assurance**: Automated content review and validation
- **REST API**: Full-featured API for podcast generation and management

### Advanced Features
- **Multiple Podcast Formats**: Solo, interview, narrative, and panel formats
- **Voice Customization**: Various voice types and TTS services
- **Content Customization**: Adjustable tone, duration, and research depth
- **Episode Management**: Track and manage generated episodes
- **Background Processing**: Asynchronous podcast generation
- **Demo Mode**: Works without API keys for demonstration

## 🏗️ Architecture

### Multi-Agent System
```
┌─────────────────────────────────────────────────────────────┐
│                    Podcast Orchestrator                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ Research    │ │ Script      │ │ Audio       │ │ Quality  │ │
│  │ Agent       │ │ Writer      │ │ Producer    │ │ Assurance│ │
│  │             │ │ Agent       │ │ Agent       │ │ Agent    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Workflow
1. **Research Phase**: AI agent researches topics and gathers information
2. **Script Writing**: Creative agent writes engaging podcast scripts
3. **Quality Review**: QA agent reviews and validates content
4. **Audio Production**: Audio agent converts script to speech
5. **Episode Assembly**: Orchestrator combines all components

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for real AI generation)
- ElevenLabs API key (optional, for premium voices)

### Quick Setup
```bash
# Clone and navigate to the project
cd /home/user/webapp

# Run the setup script
python setup.py

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir episodes audio logs temp

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## 🎯 Usage

### Demo Mode (No API Keys Required)
```bash
# Run the demo version
python demo_podcast_agents.py
```

### Full Version (With API Keys)
```bash
# Set your API keys in .env file first
# Then run the enhanced version
python enhanced_podcast_agents.py

# Or start the web API server
python main.py
```

### Web API
```bash
# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Visit API documentation
# http://localhost:8000/docs
```

## 📋 API Endpoints

### Core Endpoints
- `POST /api/v1/generate` - Generate new podcast episode
- `GET /api/v1/status/{episode_id}` - Check episode generation status
- `GET /api/v1/episodes` - List all episodes
- `GET /api/v1/episodes/{episode_id}` - Get episode details
- `GET /health` - Health check

### Example API Usage
```bash
# Generate a podcast
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The Future of AI Agents",
    "research_depth": "comprehensive",
    "tone": "professional",
    "duration": 20,
    "format": "narrative"
  }'

# Check status
curl "http://localhost:8000/api/v1/status/{episode_id}"
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
DEBUG=false
LOG_LEVEL=INFO
```

### Customization Options
- **Research Depth**: basic, moderate, comprehensive
- **Tone**: professional, casual, academic, entertaining
- **Format**: solo, interview, narrative, panel
- **Voice Type**: professional, casual, energetic, calm
- **TTS Service**: openai, elevenlabs

## 🧪 Testing

### Run Demo
```bash
python demo_podcast_agents.py
```

### Test API
```bash
# Start the server
python main.py

# Test in browser
# Visit: http://localhost:8000/docs
```

## 📁 Project Structure

```
/home/user/webapp/
├── main.py                    # FastAPI web application
├── podcast_ai_agents.py       # Basic multi-agent system
├── enhanced_podcast_agents.py  # Advanced version with real AI
├── demo_podcast_agents.py    # Demo version without API keys
├── requirements.txt            # Python dependencies
├── setup.py                  # Installation script
├── .env.example               # Environment configuration template
├── .env                       # Environment variables (create from .env.example)
├── episodes/                  # Generated episodes storage
├── audio/                     # Audio files storage
├── logs/                      # Application logs
└── temp/                      # Temporary files
```

## 🚀 Advanced Usage

### Custom Agents
Create your own agents by extending the base classes:

```python
class MyCustomAgent(EnhancedAIAgent):
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        # Your custom logic here
        return await self.generate_content("Your custom prompt")
```

### Integration with External Services
The system supports integration with:
- **OpenAI GPT Models**: For content generation
- **ElevenLabs**: For premium voice synthesis
- **Google Cloud TTS**: Alternative text-to-speech
- **Web Scraping**: For research data collection

## 🔍 Insights from 500 AI Agents Analysis

This application incorporates insights from the comprehensive analysis of 500+ AI agent projects:

### Key Learnings Applied
- **Multi-Agent Collaboration**: Different agents with specialized roles
- **Reproducibility**: Clear workflows and documentation
- **Quality Assurance**: Built-in review and validation processes
- **Scalability**: Modular architecture for easy extension
- **Ethical Considerations**: Content review and safety measures

### Design Patterns Used
- **Agent-based Architecture**: Specialized agents for different tasks
- **Orchestrator Pattern**: Central coordination of agent workflows
- **Factory Pattern**: Dynamic agent creation and configuration
- **Observer Pattern**: Status tracking and progress updates

## 🎨 Use Cases

### Content Creation
- Automated podcast generation for various topics
- Multi-format content creation (interview, narrative, educational)
- Research-based content with web scraping
- Quality-controlled content production

### Educational Applications
- Learning material generation
- Multi-language podcast creation
- Interactive educational content
- Research project presentations

### Business Applications
- Corporate podcast production
- Marketing content creation
- Training material generation
- Automated reporting

## 🔮 Future Enhancements

### Planned Features
- **Video Podcast Generation**: Add visual elements
- **Live Streaming**: Real-time podcast generation
- **Multi-language Support**: Internationalization
- **Advanced Analytics**: Content performance tracking
- **Social Media Integration**: Automatic sharing

### Technical Improvements
- **Distributed Processing**: Scale across multiple servers
- **Caching System**: Improve performance
- **Database Integration**: Persistent storage
- **Real-time Updates**: WebSocket support
- **Mobile App**: Native mobile applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the 500 AI Agents Projects collection
- Built with modern AI technologies
- Designed for scalability and extensibility
- Focused on practical applications

---

**🎙️ Start creating amazing AI-generated podcasts today!**