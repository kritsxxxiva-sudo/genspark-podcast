const API_BASE = '';

class PodcastApp {
    constructor() {
        this.currentEpisodeId = null;
        this.init();
    }

    init() {
        this.checkAPIHealth();
        this.loadEpisodes();
        this.attachEventListeners();
        
        setInterval(() => this.checkAPIHealth(), 30000);
    }

    attachEventListeners() {
        document.getElementById('podcastForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.generatePodcast();
        });

        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.loadEpisodes();
        });

        const modal = document.getElementById('episodeModal');
        const closeModal = document.querySelector('.modal-close');
        
        closeModal.addEventListener('click', () => {
            modal.style.display = 'none';
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }

    async checkAPIHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            const data = await response.json();
            
            const statusIndicator = document.getElementById('apiStatus');
            const statusText = document.getElementById('apiStatusText');
            
            if (data.status === 'healthy') {
                statusIndicator.className = 'status-indicator healthy';
                statusText.textContent = 'API Online';
            } else {
                statusIndicator.className = 'status-indicator error';
                statusText.textContent = 'API Error';
            }
        } catch (error) {
            const statusIndicator = document.getElementById('apiStatus');
            const statusText = document.getElementById('apiStatusText');
            statusIndicator.className = 'status-indicator error';
            statusText.textContent = 'API Offline';
        }
    }

    async generatePodcast() {
        const form = document.getElementById('podcastForm');
        const formData = new FormData(form);
        const generateBtn = document.getElementById('generateBtn');
        const statusDiv = document.getElementById('generationStatus');
        
        const requestData = {
            topic: formData.get('topic'),
            duration: parseInt(formData.get('duration')),
            format: formData.get('format'),
            tone: formData.get('tone'),
            research_depth: formData.get('researchDepth'),
            voice_type: formData.get('voiceType'),
            tts_service: formData.get('ttsService')
        };

        generateBtn.disabled = true;
        generateBtn.querySelector('.btn-text').style.display = 'none';
        generateBtn.querySelector('.btn-loader').style.display = 'inline';
        
        statusDiv.style.display = 'block';
        statusDiv.className = 'status-message info';
        statusDiv.textContent = 'Starting podcast generation...';

        try {
            const response = await fetch(`${API_BASE}/api/v1/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();

            if (response.ok) {
                statusDiv.className = 'status-message success';
                statusDiv.textContent = `Podcast generation started! Episode ID: ${data.episode_id}`;
                
                this.currentEpisodeId = data.episode_id;
                this.monitorEpisodeStatus(data.episode_id);
                
                form.reset();
                
                setTimeout(() => this.loadEpisodes(), 2000);
            } else {
                throw new Error(data.error || 'Failed to generate podcast');
            }
        } catch (error) {
            statusDiv.className = 'status-message error';
            statusDiv.textContent = `Error: ${error.message}`;
        } finally {
            generateBtn.disabled = false;
            generateBtn.querySelector('.btn-text').style.display = 'inline';
            generateBtn.querySelector('.btn-loader').style.display = 'none';
        }
    }

    async monitorEpisodeStatus(episodeId) {
        const statusDiv = document.getElementById('generationStatus');
        const maxAttempts = 60;
        let attempts = 0;

        const checkStatus = async () => {
            try {
                const response = await fetch(`${API_BASE}/api/v1/status/${episodeId}`);
                const data = await response.json();

                if (data.status === 'completed') {
                    statusDiv.className = 'status-message success';
                    statusDiv.textContent = `✅ Podcast generated successfully! (${data.progress}%)`;
                    this.loadEpisodes();
                    return;
                } else if (data.status === 'failed') {
                    statusDiv.className = 'status-message error';
                    statusDiv.textContent = `❌ Generation failed: ${data.message}`;
                    return;
                } else if (data.status === 'processing') {
                    statusDiv.className = 'status-message info';
                    statusDiv.textContent = `⏳ Processing... ${data.progress}% - ${data.message}`;
                }

                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(checkStatus, 5000);
                } else {
                    statusDiv.className = 'status-message error';
                    statusDiv.textContent = 'Timeout: Generation is taking longer than expected.';
                }
            } catch (error) {
                console.error('Error checking status:', error);
            }
        };

        setTimeout(checkStatus, 2000);
    }

    async loadEpisodes() {
        const episodesList = document.getElementById('episodesList');
        episodesList.innerHTML = '<div class="loading">Loading episodes...</div>';

        try {
            const response = await fetch(`${API_BASE}/api/v1/episodes`);
            const data = await response.json();

            if (data.episodes && data.episodes.length > 0) {
                episodesList.innerHTML = '';
                data.episodes.forEach(episode => {
                    const episodeEl = this.createEpisodeElement(episode);
                    episodesList.appendChild(episodeEl);
                });
            } else {
                episodesList.innerHTML = '<div class="loading">No episodes yet. Generate your first podcast!</div>';
            }
        } catch (error) {
            episodesList.innerHTML = '<div class="loading">Error loading episodes</div>';
            console.error('Error loading episodes:', error);
        }
    }

    createEpisodeElement(episode) {
        const div = document.createElement('div');
        div.className = 'episode-item';
        
        const statusClass = episode.status || 'unknown';
        const date = episode.created_at ? new Date(episode.created_at).toLocaleDateString() : 'Unknown';
        
        div.innerHTML = `
            <div class="episode-title">${episode.title || 'Untitled Episode'}</div>
            <div class="episode-meta">
                <span>📅 ${date}</span>
                <span>⏱️ ${episode.duration || 'N/A'} min</span>
                <span class="episode-status ${statusClass}">${statusClass}</span>
            </div>
        `;

        div.addEventListener('click', () => {
            this.showEpisodeDetails(episode.episode_id);
        });

        return div;
    }

    async showEpisodeDetails(episodeId) {
        const modal = document.getElementById('episodeModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');

        modalBody.innerHTML = '<div class="loading">Loading episode details...</div>';
        modal.style.display = 'flex';

        try {
            const response = await fetch(`${API_BASE}/api/v1/episodes/${episodeId}`);
            const episode = await response.json();

            modalTitle.textContent = episode.title || 'Episode Details';
            
            const statusClass = episode.status || 'unknown';
            
            modalBody.innerHTML = `
                <div class="detail-row">
                    <div class="detail-label">Status</div>
                    <div class="detail-value">
                        <span class="episode-status ${statusClass}">${statusClass}</span>
                    </div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Description</div>
                    <div class="detail-value">${episode.description || 'No description'}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Topics</div>
                    <div class="detail-value">${(episode.topics || []).join(', ') || 'No topics'}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Duration</div>
                    <div class="detail-value">${episode.duration || 'N/A'} minutes</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">AI Model</div>
                    <div class="detail-value">${episode.ai_model || 'N/A'}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Voice Model</div>
                    <div class="detail-value">${episode.voice_model || 'N/A'}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Created</div>
                    <div class="detail-value">${episode.created_at ? new Date(episode.created_at).toLocaleString() : 'Unknown'}</div>
                </div>
                ${episode.content ? `
                <div class="detail-row">
                    <div class="detail-label">Content Preview</div>
                    <div class="detail-value" style="max-height: 200px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; background: var(--bg-primary); padding: 15px; border-radius: 8px;">
                        ${episode.content.substring(0, 1000)}${episode.content.length > 1000 ? '...' : ''}
                    </div>
                </div>
                ` : ''}
            `;
        } catch (error) {
            modalBody.innerHTML = '<div class="loading">Error loading episode details</div>';
            console.error('Error loading episode details:', error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new PodcastApp();
});
