"""
ConfigSystem Module - Centralized configuration management for GGF AI Framework

This module provides a unified configuration system that can load and merge configuration
from multiple sources (JSON files, YAML files, environment variables) with schema validation.
Additionally, it provides dynamic configuration based on hardware capabilities,
ensuring optimal performance on mid to high-end gaming PCs.
"""

import json
import os
import logging
import psutil
from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path
import threading
from dataclasses import dataclass, field
import jsonschema
from enum import Enum, auto

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger("GGFAI.core_framework.config")

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass

class SchemaValidationError(ConfigError):
    """Exception for schema validation failures."""
    pass

class ConfigLoadError(ConfigError):
    """Exception for configuration loading errors."""
    pass

class HardwareTier(Enum):
    """Hardware capability tiers"""
    LOW = auto()    # Basic systems
    MID = auto()    # Mid-range gaming PCs
    HIGH = auto()   # High-end gaming PCs

@dataclass
class HardwareProfile:
    """System hardware profile"""
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_memory_gb: Optional[float]
    gpu_name: Optional[str]
    tier: HardwareTier

@dataclass
class ConfigSource:
    """Information about a configuration source."""
    name: str
    type: str  # 'json', 'yaml', 'env'
    priority: int = 0
    schema: Optional[Dict] = None
    path: Optional[str] = None
    env_prefix: Optional[str] = None
    required: bool = False
    loaded_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConfigSystem:
    """
    Centralized configuration management system.
    
    Features:
    - Load config from multiple sources (JSON, YAML, env vars)
    - Schema validation
    - Hierarchical config merging
    - Environment-specific overrides
    - Hot reloading support
    - Thread-safe operations
    - Hardware-aware configuration management
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration system.
        
        Args:
            base_path: Optional base path for relative config paths
        """
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._sources: Dict[str, ConfigSource] = {}
        self._config: Dict[str, Any] = {}
        self._schemas: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._env_prefix = "GGFAI_"
        self._last_load_time = 0.0
        self.hardware_profile = self._detect_hardware()
        self.capabilities: Set[str] = set()
        self._initialize_capabilities()

    def _detect_hardware(self) -> HardwareProfile:
        """Detect system hardware capabilities"""
        try:
            # CPU info
            cpu_cores = psutil.cpu_count(logical=False) or 2
            cpu_threads = psutil.cpu_count(logical=True) or 4
            
            # RAM info
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            
            # GPU info (basic detection)
            gpu_memory_gb = None
            gpu_name = None
            try:
                # This is a simplified check - in practice you'd use a proper GPU detection library
                if os.name == 'nt':  # Windows
                    import wmi
                    w = wmi.WMI()
                    gpu = w.Win32_VideoController()[0]
                    gpu_name = gpu.Name
                    gpu_memory_gb = float(gpu.AdapterRAM) / (1024 ** 3)
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
            
            # Determine hardware tier
            tier = self._determine_tier(cpu_cores, ram_gb, gpu_memory_gb)
            
            return HardwareProfile(
                cpu_cores=cpu_cores,
                cpu_threads=cpu_threads,
                ram_gb=ram_gb,
                gpu_memory_gb=gpu_memory_gb,
                gpu_name=gpu_name,
                tier=tier
            )
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            # Return conservative defaults
            return HardwareProfile(
                cpu_cores=2,
                cpu_threads=4,
                ram_gb=4.0,
                gpu_memory_gb=None,
                gpu_name=None,
                tier=HardwareTier.LOW
            )
    
    def _determine_tier(
        self,
        cpu_cores: int,
        ram_gb: float,
        gpu_memory_gb: Optional[float]
    ) -> HardwareTier:
        """Determine hardware tier based on specifications"""
        # High-end gaming PC requirements
        if (cpu_cores >= 6 and 
            ram_gb >= 16 and 
            (gpu_memory_gb or 0) >= 6):
            return HardwareTier.HIGH
            
        # Mid-range gaming PC requirements
        elif (cpu_cores >= 4 and 
              ram_gb >= 8 and 
              (gpu_memory_gb or 0) >= 4):
            return HardwareTier.MID
            
        # Basic/Low-end systems
        else:
            return HardwareTier.LOW
    
    def _initialize_capabilities(self) -> None:
        """Initialize system capabilities based on hardware tier"""
        # Base capabilities available on all tiers
        self.capabilities = {
            "basic_processing",
            "text_input",
            "web_interface",
            "event_system",
            "error_recovery"
        }
        
        # Mid-tier capabilities
        if self.hardware_profile.tier in {HardwareTier.MID, HardwareTier.HIGH}:
            self.capabilities.update({
                "voice_processing",
                "basic_vision",
                "local_inference",
                "parallel_processing",
                "real_time_analytics"
            })
        
        # High-tier capabilities
        if self.hardware_profile.tier == HardwareTier.HIGH:
            self.capabilities.update({
                "advanced_vision",
                "multi_model_inference",
                "real_time_adaptation",
                "advanced_analytics"
            })

    def add_source(
        self,
        name: str,
        source_type: str,
        priority: int = 0,
        schema: Optional[Dict] = None,
        path: Optional[str] = None,
        env_prefix: Optional[str] = None,
        required: bool = False
    ) -> None:
        """
        Add a configuration source.
        
        Args:
            name: Unique identifier for this source
            source_type: Type of source ('json', 'yaml', 'env')
            priority: Loading priority (higher numbers load later)
            schema: Optional JSON schema for validation
            path: Path to config file (for file sources)
            env_prefix: Prefix for environment variables
            required: Whether this source must load successfully
            
        Raises:
            ConfigError: If source already exists or parameters invalid
        """
        with self._lock:
            if name in self._sources:
                raise ConfigError(f"Configuration source {name} already exists")
            
            if source_type not in ['json', 'yaml', 'env']:
                raise ConfigError(f"Invalid source type: {source_type}")
            
            if source_type == 'yaml' and not YAML_AVAILABLE:
                raise ConfigError("YAML support not available - install pyyaml")
            
            if source_type in ['json', 'yaml'] and not path:
                raise ConfigError(f"Path required for {source_type} source")
            
            source = ConfigSource(
                name=name,
                type=source_type,
                priority=priority,
                schema=schema,
                path=str(self._base_path / path) if path else None,
                env_prefix=env_prefix,
                required=required
            )
            
            self._sources[name] = source
            if schema:
                self._schemas[name] = schema
            
            logger.info(f"Added config source: {name} ({source_type})")

    def load_config(self, reload: bool = False) -> bool:
        """
        Load or reload configuration from all sources.
        
        Args:
            reload: Force reload even if config is already loaded
            
        Returns:
            bool: True if load successful
            
        Raises:
            ConfigError: If loading fails and required sources are affected
        """
        with self._lock:
            if self._config and not reload:
                return True
            
            try:
                # Sort sources by priority
                sources = sorted(
                    self._sources.values(),
                    key=lambda s: s.priority
                )
                
                config: Dict[str, Any] = {}
                load_time = time.time()
                
                # Load each source
                for source in sources:
                    try:
                        if source.type == 'json':
                            self._load_json(source, config)
                        elif source.type == 'yaml':
                            self._load_yaml(source, config)
                        elif source.type == 'env':
                            self._load_env(source, config)
                        
                        source.loaded_at = load_time
                        
                    except Exception as e:
                        msg = f"Failed to load config from {source.name}: {str(e)}"
                        if source.required:
                            raise ConfigError(msg)
                        logger.warning(msg)
                
                # Validate final config
                self._validate_config(config)
                
                self._config = config
                self._last_load_time = load_time
                logger.info("Configuration loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Configuration load failed: {str(e)}")
                raise

    def _load_json(self, source: ConfigSource, config: Dict[str, Any]) -> None:
        """Load configuration from a JSON file."""
        try:
            with open(source.path) as f:
                data = json.load(f)
            
            if source.schema:
                jsonschema.validate(instance=data, schema=source.schema)
            
            self._merge_config(config, data)
            
        except Exception as e:
            raise ConfigLoadError(f"Error loading JSON from {source.path}: {str(e)}")

    def _load_yaml(self, source: ConfigSource, config: Dict[str, Any]) -> None:
        """Load configuration from a YAML file."""
        try:
            with open(source.path) as f:
                data = yaml.safe_load(f)
            
            if source.schema:
                jsonschema.validate(instance=data, schema=source.schema)
            
            self._merge_config(config, data)
            
        except Exception as e:
            raise ConfigLoadError(f"Error loading YAML from {source.path}: {str(e)}")

    def _load_env(self, source: ConfigSource, config: Dict[str, Any]) -> None:
        """Load configuration from environment variables."""
        prefix = source.env_prefix or self._env_prefix
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert env var name to config key
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys (e.g., GGFAI_DATABASE_HOST)
                parts = config_key.split('_')
                current = env_config
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Try to parse as JSON for complex values
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                
                current[parts[-1]] = value
        
        if source.schema:
            jsonschema.validate(instance=env_config, schema=source.schema)
        
        self._merge_config(config, env_config)

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration to merge into
            override: Configuration to merge from
        """
        for key, value in override.items():
            if (
                key in base and
                isinstance(base[key], dict) and
                isinstance(value, dict)
            ):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the complete configuration against all schemas.
        
        Args:
            config: Configuration to validate
            
        Raises:
            SchemaValidationError: If validation fails
        """
        for name, schema in self._schemas.items():
            try:
                jsonschema.validate(instance=config, schema=schema)
            except jsonschema.exceptions.ValidationError as e:
                raise SchemaValidationError(
                    f"Configuration failed validation for {name}: {str(e)}"
                )

    def get_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Args:
            reload: Whether to reload configuration first
            
        Returns:
            Complete configuration dictionary
        """
        if reload:
            self.load_config(reload=True)
        return self._config.copy()

    def get_value(
        self,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            required: Whether the key must exist
            
        Returns:
            Configuration value or default
            
        Raises:
            ConfigError: If key required but not found
        """
        try:
            value = self._config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            if required:
                raise ConfigError(f"Required config key not found: {key}")
            return default

    def get_source_info(self, name: str) -> Optional[ConfigSource]:
        """
        Get information about a configuration source.
        
        Args:
            name: Source name
            
        Returns:
            ConfigSource object or None if not found
        """
        return self._sources.get(name)

    def get_schema(self, name: str) -> Optional[Dict]:
        """
        Get the schema for a configuration source.
        
        Args:
            name: Source name
            
        Returns:
            Schema dictionary or None if not found
        """
        return self._schemas.get(name)

    def get_hardware_info(self) -> Dict:
        """Get detailed hardware information"""
        return {
            "cpu_cores": self.hardware_profile.cpu_cores,
            "cpu_threads": self.hardware_profile.cpu_threads,
            "ram_gb": round(self.hardware_profile.ram_gb, 1),
            "gpu_memory_gb": round(self.hardware_profile.gpu_memory_gb, 1) if self.hardware_profile.gpu_memory_gb else None,
            "gpu_name": self.hardware_profile.gpu_name,
            "tier": self.hardware_profile.tier.name,
            "capabilities": sorted(list(self.capabilities))
        }

    def get_server_config(self) -> Dict:
        """Get optimized server configuration"""
        workers = max(2, self.hardware_profile.cpu_cores - 1)
        return {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": workers,
            "backlog": 2048 if self.hardware_profile.tier == HardwareTier.HIGH else 1024,
            "ws_max_size": 1024 * 1024 if self.hardware_profile.tier == HardwareTier.LOW else 2 * 1024 * 1024,
            "keep_alive": 5,
            "graceful_timeout": 10,
            "compression_enabled": self.hardware_profile.tier == HardwareTier.LOW
        }

    def has_capability(self, capability: str) -> bool:
        """Check if system has a specific capability"""
        return capability in self.capabilities
        
    def get_capabilities(self) -> Set[str]:
        """Get all available system capabilities"""
        return self.capabilities.copy()