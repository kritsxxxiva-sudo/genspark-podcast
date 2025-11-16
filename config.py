"""
Advanced configuration system for podcast AI agents
Manages API keys, feature flags, and system settings
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FeatureFlag(Enum):
    """Feature flags for enabling/disabling advanced features."""
    ADVANCED_ORCHESTRATION = "advanced_orchestration"
    PERSONALIZATION = "personalization"
    MULTILINGUAL_SUPPORT = "multilingual_support"
    REAL_TIME_ANALYTICS = "real_time_analytics"
    AUDIO_PROCESSING = "audio_processing"
    WEB_SCRAPING = "web_scraping"
    CACHING = "caching"
    BACKGROUND_TASKS = "background_tasks"
    WEBSOCKET_SUPPORT = "websocket_support"

class ServiceProvider(Enum):
    """Service providers for different functionalities."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"
    PLAYHT = "playht"

@dataclass
class ServiceConfig:
    """Configuration for external services."""
    provider: ServiceProvider
    api_key: str
    api_url: Optional[str] = None
    model: Optional[str] = None
    rate_limit: int = 100
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider.value,
            "api_key": "***" if self.api_key else None,  # Mask API key
            "api_url": self.api_url,
            "model": self.model,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay
        }

@dataclass
class CacheConfig:
    """Configuration for caching system."""
    enabled: bool = True
    provider: str = "redis"  # or "memory", "disk"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    ttl: int = 3600  # 1 hour default
    max_entries: int = 10000
    compression: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 44100
    bit_rate: str = "128k"
    channels: int = 2
    format: str = "mp3"
    quality: str = "high"
    effects: List[str] = field(default_factory=lambda: ["normalization", "compression"])
    tts_provider: ServiceProvider = ServiceProvider.OPENAI
    voice_model: str = "alloy"
    speech_rate: float = 1.0
    pitch: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "bit_rate": self.bit_rate,
            "channels": self.channels,
            "format": self.format,
            "quality": self.quality,
            "effects": self.effects,
            "tts_provider": self.tts_provider.value,
            "voice_model": self.voice_model,
            "speech_rate": self.speech_rate,
            "pitch": self.pitch
        }

@dataclass
class ResearchConfig:
    """Configuration for research and content gathering."""
    max_sources: int = 10
    timeout: int = 30
    user_agent: str = "PodcastAIAgents/2.0"
    max_content_length: int = 50000
    enable_web_scraping: bool = True
    enable_api_calls: bool = True
    enable_sentiment_analysis: bool = True
    fact_checking: bool = True
    source_credibility_threshold: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PersonalizationConfig:
    """Configuration for content personalization."""
    enabled: bool = True
    ml_model: str = "tf-idf"  # or "transformer", "custom"
    similarity_threshold: float = 0.6
    update_frequency: int = 24  # hours
    learning_rate: float = 0.01
    batch_size: int = 32
    max_history_items: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalyticsConfig:
    """Configuration for analytics and metrics."""
    enabled: bool = True
    storage_provider: str = "local"  # or "cloud", "database"
    retention_days: int = 365
    aggregation_interval: str = "hourly"  # or "daily", "weekly"
    real_time_updates: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "pdf"])
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "engagement", "completion_rate", "listening_time", "topics", "sentiment"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SecurityConfig:
    """Configuration for security and privacy."""
    encryption_enabled: bool = True
    rate_limiting: bool = True
    max_requests_per_minute: int = 60
    api_key_rotation_days: int = 90
    audit_logging: bool = True
    data_retention_days: int = 30
    anonymization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AdvancedConfig:
    """Main configuration class for advanced podcast AI agents."""
    
    # Feature flags
    feature_flags: Dict[FeatureFlag, bool] = field(default_factory=lambda: {
        FeatureFlag.ADVANCED_ORCHESTRATION: True,
        FeatureFlag.PERSONALIZATION: True,
        FeatureFlag.MULTILINGUAL_SUPPORT: True,
        FeatureFlag.REAL_TIME_ANALYTICS: True,
        FeatureFlag.AUDIO_PROCESSING: True,
        FeatureFlag.WEB_SCRAPING: True,
        FeatureFlag.CACHING: True,
        FeatureFlag.BACKGROUND_TASKS: True,
        FeatureFlag.WEBSOCKET_SUPPORT: True
    })
    
    # Service configurations
    services: Dict[str, ServiceConfig] = field(default_factory=dict)
    
    # Specialized configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # System settings
    debug: bool = False
    log_level: str = "INFO"
    max_concurrent_requests: int = 10
    request_timeout: int = 120
    health_check_interval: int = 30
    
    def __post_init__(self):
        """Initialize service configurations."""
        if not self.services:
            self.services = {
                "openai": ServiceConfig(
                    provider=ServiceProvider.OPENAI,
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    model="gpt-4",
                    rate_limit=100
                ),
                "elevenlabs": ServiceConfig(
                    provider=ServiceProvider.ELEVENLABS,
                    api_key=os.getenv("ELEVENLABS_API_KEY", ""),
                    model="eleven_monolingual_v1",
                    rate_limit=50
                )
            }
    
    def is_feature_enabled(self, feature: FeatureFlag) -> bool:
        """Check if a feature is enabled."""
        return self.feature_flags.get(feature, False)
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for a specific service."""
        return self.services.get(service_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "feature_flags": {flag.value: enabled for flag, enabled in self.feature_flags.items()},
            "services": {name: config.to_dict() for name, config in self.services.items()},
            "cache": self.cache.to_dict(),
            "audio": self.audio.to_dict(),
            "research": self.research.to_dict(),
            "personalization": self.personalization.to_dict(),
            "analytics": self.analytics.to_dict(),
            "security": self.security.to_dict(),
            "debug": self.debug,
            "log_level": self.log_level,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "health_check_interval": self.health_check_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedConfig':
        """Create configuration from dictionary."""
        # Parse feature flags
        feature_flags = {}
        for flag_name, enabled in data.get("feature_flags", {}).items():
            try:
                flag = FeatureFlag(flag_name)
                feature_flags[flag] = enabled
            except ValueError:
                logger.warning(f"Unknown feature flag: {flag_name}")
        
        # Parse service configurations
        services = {}
        for service_name, service_data in data.get("services", {}).items():
            provider = ServiceProvider(service_data.get("provider", "openai"))
            services[service_name] = ServiceConfig(
                provider=provider,
                api_key=service_data.get("api_key", ""),
                api_url=service_data.get("api_url"),
                model=service_data.get("model"),
                rate_limit=service_data.get("rate_limit", 100),
                timeout=service_data.get("timeout", 30),
                retry_attempts=service_data.get("retry_attempts", 3),
                retry_delay=service_data.get("retry_delay", 1.0)
            )
        
        # Create configuration instance
        config = cls(
            feature_flags=feature_flags,
            services=services
        )
        
        # Update specialized configurations
        if "cache" in data:
            cache_data = data["cache"]
            config.cache = CacheConfig(**cache_data)
        
        if "audio" in data:
            audio_data = data["audio"]
            config.audio = AudioConfig(**audio_data)
        
        if "research" in data:
            research_data = data["research"]
            config.research = ResearchConfig(**research_data)
        
        if "personalization" in data:
            personalization_data = data["personalization"]
            config.personalization = PersonalizationConfig(**personalization_data)
        
        if "analytics" in data:
            analytics_data = data["analytics"]
            config.analytics = AnalyticsConfig(**analytics_data)
        
        if "security" in data:
            security_data = data["security"]
            config.security = SecurityConfig(**security_data)
        
        # Update system settings
        config.debug = data.get("debug", False)
        config.log_level = data.get("log_level", "INFO")
        config.max_concurrent_requests = data.get("max_concurrent_requests", 10)
        config.request_timeout = data.get("request_timeout", 120)
        config.health_check_interval = data.get("health_check_interval", 30)
        
        return config
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'AdvancedConfig':
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate API keys
        for service_name, service_config in self.services.items():
            if not service_config.api_key and service_name in ["openai", "elevenlabs"]:
                errors.append(f"API key required for {service_name}")
        
        # Validate feature dependencies
        if self.is_feature_enabled(FeatureFlag.CACHING) and not self.cache.enabled:
            errors.append("Caching feature enabled but cache configuration disabled")
        
        if self.is_feature_enabled(FeatureFlag.PERSONALIZATION) and not self.personalization.enabled:
            errors.append("Personalization feature enabled but personalization configuration disabled")
        
        # Validate numeric ranges
        if self.max_concurrent_requests <= 0:
            errors.append("max_concurrent_requests must be positive")
        
        if self.request_timeout <= 0:
            errors.append("request_timeout must be positive")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True

# Configuration factory functions
def create_production_config() -> AdvancedConfig:
    """Create production-ready configuration."""
    config = AdvancedConfig(
        debug=False,
        log_level="INFO",
        max_concurrent_requests=50,
        request_timeout=120,
        health_check_interval=30
    )
    
    # Configure security settings
    config.security = SecurityConfig(
        encryption_enabled=True,
        rate_limiting=True,
        max_requests_per_minute=120,
        audit_logging=True
    )
    
    # Configure caching for production
    config.cache = CacheConfig(
        enabled=True,
        provider="redis",
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        ttl=7200  # 2 hours
    )
    
    # Configure analytics
    config.analytics = AnalyticsConfig(
        enabled=True,
        real_time_updates=True,
        retention_days=730  # 2 years
    )
    
    return config

def create_development_config() -> AdvancedConfig:
    """Create development configuration."""
    config = AdvancedConfig(
        debug=True,
        log_level="DEBUG",
        max_concurrent_requests=5,
        request_timeout=60,
        health_check_interval=60
    )
    
    # Use memory caching for development
    config.cache = CacheConfig(
        enabled=True,
        provider="memory",
        ttl=1800  # 30 minutes
    )
    
    # Relaxed security for development
    config.security = SecurityConfig(
        encryption_enabled=False,
        rate_limiting=False,
        max_requests_per_minute=1000,
        audit_logging=True
    )
    
    return config

def create_testing_config() -> AdvancedConfig:
    """Create testing configuration."""
    config = AdvancedConfig(
        debug=True,
        log_level="INFO",
        max_concurrent_requests=1,
        request_timeout=30,
        health_check_interval=10
    )
    
    # Disable most features for testing
    for flag in config.feature_flags:
        config.feature_flags[flag] = False
    
    # Enable only essential features
    config.feature_flags[FeatureFlag.ADVANCED_ORCHESTRATION] = True
    config.feature_flags[FeatureFlag.CACHING] = True
    
    return config

# Global configuration instance
_config_instance: Optional[AdvancedConfig] = None

def get_config() -> AdvancedConfig:
    """Get the current configuration instance."""
    global _config_instance
    if _config_instance is None:
        # Create default configuration based on environment
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            _config_instance = create_production_config()
        elif env == "testing":
            _config_instance = create_testing_config()
        else:
            _config_instance = create_development_config()
    
    return _config_instance

def set_config(config: AdvancedConfig):
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config
    logger.info("Global configuration updated")

# Example usage and testing
if __name__ == "__main__":
    # Test configuration creation
    print("Testing advanced configuration system...")
    
    # Create development config
    dev_config = create_development_config()
    print(f"Development config created: {dev_config.debug}")
    
    # Test feature flags
    print(f"Advanced orchestration enabled: {dev_config.is_feature_enabled(FeatureFlag.ADVANCED_ORCHESTRATION)}")
    
    # Test serialization
    config_dict = dev_config.to_dict()
    print(f"Config serialized successfully: {len(config_dict)} sections")
    
    # Test deserialization
    restored_config = AdvancedConfig.from_dict(config_dict)
    print(f"Config restored successfully: {restored_config.debug}")
    
    # Test validation
    is_valid = dev_config.validate()
    print(f"Config validation: {'passed' if is_valid else 'failed'}")
    
    print("Configuration system tests completed!")