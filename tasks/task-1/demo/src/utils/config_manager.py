"""
Configuration Management Module

This module provides comprehensive configuration loading and validation
for the OCR batch processing system.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ProcessingConfig moved inline since BatchProcessor is being removed
from utils.cost_analysis import CloudCosts, HardwareCosts, PowerConsumption


@dataclass
class ProcessingConfig:
    """Configuration for processing (simplified from BatchProcessor)."""
    
    device: str = "auto"  # auto, cpu, gpu, cuda, mps
    num_workers: int = None  # None = auto-detect
    batch_size: int = 10
    timeout: int = 30  # seconds per document
    max_retries: int = 3
    gpu_memory_limit: float = 0.8  # Use 80% of GPU memory
    enable_progress: bool = True
    enable_monitoring: bool = True


@dataclass 
class ImageTransformPipelineConfig:
    """Configuration for image transform pipeline."""
    
    batch_size: int = 16
    num_workers: int = None  # None = auto-detect
    use_processes: bool = False  # False = threads, True = processes
    timeout: float = 30.0
    
    # Image quality settings
    quality_enhancement: bool = False
    normalize: bool = False
    grayscale_conversion: bool = False
    resize_images: bool = False
    target_size: List[int] = None  # [width, height]
    max_image_size: List[int] = field(default_factory=lambda: [2048, 2048])
    
    # Advanced transform settings
    gaussian_blur: List[int] = None  # [kernel_x, kernel_y]
    adaptive_threshold: bool = False


@dataclass
class OCRReaderPipelineConfig:
    """Configuration for OCR reader pipeline."""
    
    device: str = "auto"  # auto, cpu, gpu, cuda, mps
    batch_size: int = 1
    confidence_threshold: float = 0.1
    timeout_per_image: float = 30.0
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCoordinationConfig:
    """Configuration for pipeline coordination."""
    
    enable_continuous_processing: bool = True
    max_buffer_refill_workers: int = 1
    stats_update_interval: float = 5.0


@dataclass
class PipelineConfiguration:
    """Complete pipeline configuration."""
    
    image_transform: ImageTransformPipelineConfig = field(default_factory=ImageTransformPipelineConfig)
    ocr_reader: OCRReaderPipelineConfig = field(default_factory=OCRReaderPipelineConfig)
    coordination: PipelineCoordinationConfig = field(default_factory=PipelineCoordinationConfig)


@dataclass
class OCRModelConfig:
    """OCR model configuration."""
    name: str = "easyocr"
    languages: List[str] = field(default_factory=lambda: ["en"])
    gpu: bool = False


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_dir: str = "data/funsd_subset"
    num_documents: int = 100
    skip_download: bool = False
    validation_enabled: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    normalize_text: bool = True
    lowercase: bool = True
    remove_punctuation: bool = False
    calculate_correlations: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    log_dir: str = "logs"
    sampling_interval: float = 1.0
    enable_gpu_monitoring: bool = True
    enable_nvml: bool = True


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    output_dir: str = "results"
    generate_html: bool = True
    generate_csv: bool = True
    generate_plots: bool = True
    generate_executive_summary: bool = True
    plot_dpi: int = 300
    plot_style: str = "seaborn-v0_8"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: Optional[str] = None
    console_output: bool = True
    file_output: bool = True
    external_lib_level: str = "WARNING"


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    error_log_retention_days: int = 30
    enable_graceful_shutdown: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    enable_multiprocessing: bool = False
    memory_limit_gb: Optional[float] = None
    disk_space_threshold_gb: float = 1.0
    cpu_threshold: float = 0.95
    memory_threshold: float = 0.9


@dataclass
class AppConfig:
    """Complete application configuration."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    pipeline: PipelineConfiguration = field(default_factory=PipelineConfiguration)
    ocr_model: OCRModelConfig = field(default_factory=OCRModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cost_analysis_hardware: HardwareCosts = field(default_factory=HardwareCosts)
    cost_analysis_power: PowerConsumption = field(default_factory=PowerConsumption)
    cost_analysis_cloud: CloudCosts = field(default_factory=CloudCosts)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


class ConfigManager:
    """Manages application configuration loading and validation."""
    
    def __init__(self):
        self.config: AppConfig = AppConfig()
        self.config_loaded = False
    
    def load_config(self, config_path: str) -> AppConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Complete application configuration
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            logging.warning(f"Config file {config_path} not found, using defaults")
            return self.config
        
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            if not yaml_config:
                logging.warning("Empty config file, using defaults")
                return self.config
            
            # Parse configuration sections
            self._load_processing_config(yaml_config.get('processing', {}))
            self._load_pipeline_config(yaml_config.get('pipeline', {}))
            self._load_ocr_model_config(yaml_config.get('ocr_model', {}))
            self._load_dataset_config(yaml_config.get('dataset', {}))
            self._load_evaluation_config(yaml_config.get('evaluation', {}))
            self._load_cost_analysis_config(yaml_config.get('cost_analysis', {}))
            self._load_monitoring_config(yaml_config.get('monitoring', {}))
            self._load_reporting_config(yaml_config.get('reporting', {}))
            self._load_logging_config(yaml_config.get('logging', {}))
            self._load_error_handling_config(yaml_config.get('error_handling', {}))
            self._load_performance_config(yaml_config.get('performance', {}))
            
            self.config_loaded = True
            logging.info(f"Configuration loaded successfully from {config_path}")
            
        except Exception as e:
            logging.error(f"Error loading config file {config_path}: {e}")
            logging.info("Using default configuration")
        
        return self.config
    
    def _load_processing_config(self, config_dict: Dict):
        """Load processing configuration."""
        self.config.processing = ProcessingConfig(
            device=config_dict.get('device', self.config.processing.device),
            num_workers=config_dict.get('num_workers', self.config.processing.num_workers),
            batch_size=config_dict.get('batch_size', self.config.processing.batch_size),
            timeout=config_dict.get('timeout', self.config.processing.timeout),
            max_retries=config_dict.get('max_retries', self.config.processing.max_retries),
            gpu_memory_limit=config_dict.get('gpu_memory_limit', self.config.processing.gpu_memory_limit),
            enable_progress=config_dict.get('enable_progress', self.config.processing.enable_progress),
            enable_monitoring=config_dict.get('enable_monitoring', self.config.processing.enable_monitoring)
        )
    
    def _load_pipeline_config(self, config_dict: Dict):
        """Load pipeline configuration."""
        # Load image transform config
        transform_config = config_dict.get('image_transform', {})
        image_transform_cfg = ImageTransformPipelineConfig(
            batch_size=transform_config.get('batch_size', self.config.pipeline.image_transform.batch_size),
            num_workers=transform_config.get('num_workers', self.config.pipeline.image_transform.num_workers),
            use_processes=transform_config.get('use_processes', self.config.pipeline.image_transform.use_processes),
            timeout=transform_config.get('timeout', self.config.pipeline.image_transform.timeout),
            quality_enhancement=transform_config.get('quality_enhancement', self.config.pipeline.image_transform.quality_enhancement),
            normalize=transform_config.get('normalize', self.config.pipeline.image_transform.normalize),
            grayscale_conversion=transform_config.get('grayscale_conversion', self.config.pipeline.image_transform.grayscale_conversion),
            resize_images=transform_config.get('resize_images', self.config.pipeline.image_transform.resize_images),
            target_size=transform_config.get('target_size', self.config.pipeline.image_transform.target_size),
            max_image_size=transform_config.get('max_image_size', self.config.pipeline.image_transform.max_image_size),
            gaussian_blur=transform_config.get('gaussian_blur', self.config.pipeline.image_transform.gaussian_blur),
            adaptive_threshold=transform_config.get('adaptive_threshold', self.config.pipeline.image_transform.adaptive_threshold)
        )
        
        # Load OCR reader config
        ocr_config = config_dict.get('ocr_reader', {})
        ocr_reader_cfg = OCRReaderPipelineConfig(
            device=ocr_config.get('device', self.config.pipeline.ocr_reader.device),
            batch_size=ocr_config.get('batch_size', self.config.pipeline.ocr_reader.batch_size),
            confidence_threshold=ocr_config.get('confidence_threshold', self.config.pipeline.ocr_reader.confidence_threshold),
            timeout_per_image=ocr_config.get('timeout_per_image', self.config.pipeline.ocr_reader.timeout_per_image),
            model_kwargs=ocr_config.get('model_kwargs', self.config.pipeline.ocr_reader.model_kwargs)
        )
        
        # Load coordination config
        coord_config = config_dict.get('coordination', {})
        coordination_cfg = PipelineCoordinationConfig(
            enable_continuous_processing=coord_config.get('enable_continuous_processing', self.config.pipeline.coordination.enable_continuous_processing),
            max_buffer_refill_workers=coord_config.get('max_buffer_refill_workers', self.config.pipeline.coordination.max_buffer_refill_workers),
            stats_update_interval=coord_config.get('stats_update_interval', self.config.pipeline.coordination.stats_update_interval)
        )
        
        # Create complete pipeline config
        self.config.pipeline = PipelineConfiguration(
            image_transform=image_transform_cfg,
            ocr_reader=ocr_reader_cfg,
            coordination=coordination_cfg
        )
    
    def _load_ocr_model_config(self, config_dict: Dict):
        """Load OCR model configuration."""
        self.config.ocr_model = OCRModelConfig(
            name=config_dict.get('name', self.config.ocr_model.name),
            languages=config_dict.get('languages', self.config.ocr_model.languages),
            gpu=config_dict.get('gpu', self.config.ocr_model.gpu)
        )
    
    def _load_dataset_config(self, config_dict: Dict):
        """Load dataset configuration."""
        self.config.dataset = DatasetConfig(
            data_dir=config_dict.get('data_dir', self.config.dataset.data_dir),
            num_documents=config_dict.get('num_documents', self.config.dataset.num_documents),
            skip_download=config_dict.get('skip_download', self.config.dataset.skip_download),
            validation_enabled=config_dict.get('validation_enabled', self.config.dataset.validation_enabled)
        )
    
    def _load_evaluation_config(self, config_dict: Dict):
        """Load evaluation configuration."""
        self.config.evaluation = EvaluationConfig(
            normalize_text=config_dict.get('normalize_text', self.config.evaluation.normalize_text),
            lowercase=config_dict.get('lowercase', self.config.evaluation.lowercase),
            remove_punctuation=config_dict.get('remove_punctuation', self.config.evaluation.remove_punctuation),
            calculate_correlations=config_dict.get('calculate_correlations', self.config.evaluation.calculate_correlations)
        )
    
    def _load_cost_analysis_config(self, config_dict: Dict):
        """Load cost analysis configuration."""
        hardware_config = config_dict.get('hardware_costs', {})
        self.config.cost_analysis_hardware = HardwareCosts(
            cpu_cost=hardware_config.get('cpu_cost', self.config.cost_analysis_hardware.cpu_cost),
            gpu_cost=hardware_config.get('gpu_cost', self.config.cost_analysis_hardware.gpu_cost),
            memory_cost=hardware_config.get('memory_cost', self.config.cost_analysis_hardware.memory_cost),
            storage_cost=hardware_config.get('storage_cost', self.config.cost_analysis_hardware.storage_cost),
            other_components=hardware_config.get('other_components', self.config.cost_analysis_hardware.other_components),
            cpu_lifespan_years=hardware_config.get('cpu_lifespan_years', self.config.cost_analysis_hardware.cpu_lifespan_years),
            gpu_lifespan_years=hardware_config.get('gpu_lifespan_years', self.config.cost_analysis_hardware.gpu_lifespan_years),
            memory_lifespan_years=hardware_config.get('memory_lifespan_years', self.config.cost_analysis_hardware.memory_lifespan_years),
            storage_lifespan_years=hardware_config.get('storage_lifespan_years', self.config.cost_analysis_hardware.storage_lifespan_years),
            other_lifespan_years=hardware_config.get('other_lifespan_years', self.config.cost_analysis_hardware.other_lifespan_years)
        )
        
        power_config = config_dict.get('power_consumption', {})
        self.config.cost_analysis_power = PowerConsumption(
            cpu_idle_watts=power_config.get('cpu_idle_watts', self.config.cost_analysis_power.cpu_idle_watts),
            cpu_load_watts=power_config.get('cpu_load_watts', self.config.cost_analysis_power.cpu_load_watts),
            gpu_idle_watts=power_config.get('gpu_idle_watts', self.config.cost_analysis_power.gpu_idle_watts),
            gpu_load_watts=power_config.get('gpu_load_watts', self.config.cost_analysis_power.gpu_load_watts),
            memory_watts=power_config.get('memory_watts', self.config.cost_analysis_power.memory_watts),
            storage_watts=power_config.get('storage_watts', self.config.cost_analysis_power.storage_watts),
            motherboard_watts=power_config.get('motherboard_watts', self.config.cost_analysis_power.motherboard_watts),
            electricity_cost_kwh=power_config.get('electricity_cost_kwh', self.config.cost_analysis_power.electricity_cost_kwh)
        )
        
        cloud_config = config_dict.get('cloud_costs', {})
        self.config.cost_analysis_cloud = CloudCosts(
            google_vision_per_1k=cloud_config.get('google_vision_per_1k', self.config.cost_analysis_cloud.google_vision_per_1k),
            aws_textract_per_1k=cloud_config.get('aws_textract_per_1k', self.config.cost_analysis_cloud.aws_textract_per_1k),
            azure_vision_per_1k=cloud_config.get('azure_vision_per_1k', self.config.cost_analysis_cloud.azure_vision_per_1k),
            azure_forms_per_1k=cloud_config.get('azure_forms_per_1k', self.config.cost_analysis_cloud.azure_forms_per_1k)
        )
    
    def _load_monitoring_config(self, config_dict: Dict):
        """Load monitoring configuration."""
        self.config.monitoring = MonitoringConfig(
            log_dir=config_dict.get('log_dir', self.config.monitoring.log_dir),
            sampling_interval=config_dict.get('sampling_interval', self.config.monitoring.sampling_interval),
            enable_gpu_monitoring=config_dict.get('enable_gpu_monitoring', self.config.monitoring.enable_gpu_monitoring),
            enable_nvml=config_dict.get('enable_nvml', self.config.monitoring.enable_nvml)
        )
    
    def _load_reporting_config(self, config_dict: Dict):
        """Load reporting configuration."""
        self.config.reporting = ReportingConfig(
            output_dir=config_dict.get('output_dir', self.config.reporting.output_dir),
            generate_html=config_dict.get('generate_html', self.config.reporting.generate_html),
            generate_csv=config_dict.get('generate_csv', self.config.reporting.generate_csv),
            generate_plots=config_dict.get('generate_plots', self.config.reporting.generate_plots),
            generate_executive_summary=config_dict.get('generate_executive_summary', self.config.reporting.generate_executive_summary),
            plot_dpi=config_dict.get('plot_dpi', self.config.reporting.plot_dpi),
            plot_style=config_dict.get('plot_style', self.config.reporting.plot_style)
        )
    
    def _load_logging_config(self, config_dict: Dict):
        """Load logging configuration."""
        self.config.logging = LoggingConfig(
            level=config_dict.get('level', self.config.logging.level),
            log_file=config_dict.get('log_file', self.config.logging.log_file),
            console_output=config_dict.get('console_output', self.config.logging.console_output),
            file_output=config_dict.get('file_output', self.config.logging.file_output),
            external_lib_level=config_dict.get('external_lib_level', self.config.logging.external_lib_level)
        )
    
    def _load_error_handling_config(self, config_dict: Dict):
        """Load error handling configuration."""
        self.config.error_handling = ErrorHandlingConfig(
            max_retries=config_dict.get('max_retries', self.config.error_handling.max_retries),
            base_delay=config_dict.get('base_delay', self.config.error_handling.base_delay),
            max_delay=config_dict.get('max_delay', self.config.error_handling.max_delay),
            backoff_factor=config_dict.get('backoff_factor', self.config.error_handling.backoff_factor),
            error_log_retention_days=config_dict.get('error_log_retention_days', self.config.error_handling.error_log_retention_days),
            enable_graceful_shutdown=config_dict.get('enable_graceful_shutdown', self.config.error_handling.enable_graceful_shutdown)
        )
    
    def _load_performance_config(self, config_dict: Dict):
        """Load performance configuration."""
        self.config.performance = PerformanceConfig(
            enable_multiprocessing=config_dict.get('enable_multiprocessing', self.config.performance.enable_multiprocessing),
            memory_limit_gb=config_dict.get('memory_limit_gb', self.config.performance.memory_limit_gb),
            disk_space_threshold_gb=config_dict.get('disk_space_threshold_gb', self.config.performance.disk_space_threshold_gb),
            cpu_threshold=config_dict.get('cpu_threshold', self.config.performance.cpu_threshold),
            memory_threshold=config_dict.get('memory_threshold', self.config.performance.memory_threshold)
        )
    
    def override_from_cli(self, **cli_args):
        """
        Override configuration with command line arguments.
        
        Args:
            **cli_args: Command line arguments to override
        """
        # Override processing config
        if 'device' in cli_args and cli_args['device'] is not None:
            self.config.processing.device = cli_args['device']
        
        if 'num_workers' in cli_args and cli_args['num_workers'] is not None:
            self.config.processing.num_workers = cli_args['num_workers']
        
        if 'batch_size' in cli_args and cli_args['batch_size'] is not None:
            self.config.processing.batch_size = cli_args['batch_size']
        
        # Override dataset config
        if 'num_documents' in cli_args and cli_args['num_documents'] is not None:
            self.config.dataset.num_documents = cli_args['num_documents']
        
        if 'dataset_dir' in cli_args and cli_args['dataset_dir'] is not None:
            self.config.dataset.data_dir = cli_args['dataset_dir']
        
        if 'skip_download' in cli_args and cli_args['skip_download'] is not None:
            self.config.dataset.skip_download = cli_args['skip_download']
        
        # Override reporting config
        if 'output_dir' in cli_args and cli_args['output_dir'] is not None:
            self.config.reporting.output_dir = cli_args['output_dir']
        
        if 'generate_reports' in cli_args and cli_args['generate_reports'] is not None:
            self.config.reporting.generate_html = cli_args['generate_reports']
            self.config.reporting.generate_csv = cli_args['generate_reports']
            self.config.reporting.generate_plots = cli_args['generate_reports']
        
        # Override OCR model config
        if 'model' in cli_args and cli_args['model'] is not None:
            self.config.ocr_model.name = cli_args['model']
        
        # Override logging config
        if 'log_level' in cli_args and cli_args['log_level'] is not None:
            self.config.logging.level = cli_args['log_level']
            
        # Override pipeline config
        if 'quality_enhancement' in cli_args and cli_args['quality_enhancement'] is not None:
            self.config.pipeline.image_transform.quality_enhancement = cli_args['quality_enhancement']
            
        if 'normalize_images' in cli_args and cli_args['normalize_images'] is not None:
            self.config.pipeline.image_transform.normalize = cli_args['normalize_images']
            
        if 'ocr_batch_size' in cli_args and cli_args['ocr_batch_size'] is not None:
            self.config.pipeline.ocr_reader.batch_size = cli_args['ocr_batch_size']
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Validate processing config
        if self.config.processing.batch_size <= 0:
            issues.append("Processing batch_size must be positive")
        
        if self.config.processing.timeout <= 0:
            issues.append("Processing timeout must be positive")
        
        if not self.config.processing.device in ['auto', 'cpu', 'gpu', 'cuda', 'mps']:
            issues.append("Processing device must be 'auto', 'cpu', 'gpu', 'cuda', or 'mps'")
        
        # Validate dataset config
        if self.config.dataset.num_documents <= 0:
            issues.append("Dataset num_documents must be positive")
        
        # Validate logging config
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.logging.level.upper() not in valid_log_levels:
            issues.append(f"Logging level must be one of: {valid_log_levels}")
        
        # Validate cost analysis config
        if self.config.cost_analysis_hardware.cpu_cost <= 0:
            issues.append("Hardware CPU cost must be positive")
        
        if self.config.cost_analysis_power.electricity_cost_kwh <= 0:
            issues.append("Electricity cost per kWh must be positive")
        
        # Validate monitoring config
        if self.config.monitoring.sampling_interval <= 0:
            issues.append("Monitoring sampling_interval must be positive")
        
        return issues
    
    def save_config(self, output_path: str):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        config_dict = {
            'processing': asdict(self.config.processing),
            'pipeline': asdict(self.config.pipeline),
            'ocr_model': asdict(self.config.ocr_model),
            'dataset': asdict(self.config.dataset),
            'evaluation': asdict(self.config.evaluation),
            'cost_analysis': {
                'hardware_costs': asdict(self.config.cost_analysis_hardware),
                'power_consumption': asdict(self.config.cost_analysis_power),
                'cloud_costs': asdict(self.config.cost_analysis_cloud)
            },
            'monitoring': asdict(self.config.monitoring),
            'reporting': asdict(self.config.reporting),
            'logging': asdict(self.config.logging),
            'error_handling': asdict(self.config.error_handling),
            'performance': asdict(self.config.performance)
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logging.info(f"Configuration saved to {output_path}")
    
    def print_config_summary(self):
        """Print configuration summary to console."""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"Processing Device: {self.config.processing.device}")
        print(f"Batch Size: {self.config.processing.batch_size}")
        print(f"OCR Model: {self.config.ocr_model.name}")
        print(f"Dataset: {self.config.dataset.num_documents} documents from {self.config.dataset.data_dir}")
        print(f"Output Directory: {self.config.reporting.output_dir}")
        print(f"Log Level: {self.config.logging.level}")
        
        # Pipeline configuration summary
        print("\nPipeline Configuration:")
        print(f"  Transform Workers: {self.config.pipeline.image_transform.num_workers or 'auto'}")
        print(f"  Transform Batch Size: {self.config.pipeline.image_transform.batch_size}")
        print(f"  Quality Enhancement: {self.config.pipeline.image_transform.quality_enhancement}")
        print(f"  Image Normalization: {self.config.pipeline.image_transform.normalize}")
        print(f"  OCR Device: {self.config.pipeline.ocr_reader.device}")
        print(f"  OCR Batch Size: {self.config.pipeline.ocr_reader.batch_size}")
        print(f"  Continuous Processing: {self.config.pipeline.coordination.enable_continuous_processing}")
        
        print(f"\nConfig Loaded: {'Yes' if self.config_loaded else 'No (using defaults)'}")
        print("=" * 60)
    
    def get_pipeline_configs(self):
        """
        Convert configuration to actual pipeline configuration objects.
        
        Returns:
            Tuple of (ImageTransformConfig, OCRConfig, PipelineConfig)
        """
        from pipeline import ImageTransformConfig, OCRConfig, PipelineConfig
        
        # Handle num_workers auto-detection
        num_workers = self.config.pipeline.image_transform.num_workers
        if num_workers is None:
            import multiprocessing as mp
            num_workers = min(self.config.pipeline.image_transform.batch_size, mp.cpu_count())
        
        # Create ImageTransformConfig
        transform_config = ImageTransformConfig(
            batch_size=self.config.pipeline.image_transform.batch_size,
            num_workers=num_workers,
            use_processes=self.config.pipeline.image_transform.use_processes,
            timeout=self.config.pipeline.image_transform.timeout,
            quality_enhancement=self.config.pipeline.image_transform.quality_enhancement,
            normalize=self.config.pipeline.image_transform.normalize,
            grayscale_conversion=self.config.pipeline.image_transform.grayscale_conversion,
            resize_images=self.config.pipeline.image_transform.resize_images,
            target_size=tuple(self.config.pipeline.image_transform.target_size) if self.config.pipeline.image_transform.target_size else None,
            max_image_size=tuple(self.config.pipeline.image_transform.max_image_size) if self.config.pipeline.image_transform.max_image_size else None,
            gaussian_blur=tuple(self.config.pipeline.image_transform.gaussian_blur) if self.config.pipeline.image_transform.gaussian_blur else None,
            adaptive_threshold=self.config.pipeline.image_transform.adaptive_threshold
        )
        
        # Determine OCR device (with fallback to processing.device)
        ocr_device = self.config.pipeline.ocr_reader.device
        if ocr_device == "auto":
            ocr_device = self.config.processing.device
        
        # Create OCRConfig
        ocr_config = OCRConfig(
            device=ocr_device,
            batch_size=self.config.pipeline.ocr_reader.batch_size,
            confidence_threshold=self.config.pipeline.ocr_reader.confidence_threshold,
            timeout_per_image=self.config.pipeline.ocr_reader.timeout_per_image,
            model_kwargs=self.config.pipeline.ocr_reader.model_kwargs
        )
        
        # Create PipelineConfig
        pipeline_config = PipelineConfig(
            transform_config=transform_config,
            ocr_config=ocr_config,
            enable_continuous_processing=self.config.pipeline.coordination.enable_continuous_processing,
            max_buffer_refill_workers=self.config.pipeline.coordination.max_buffer_refill_workers,
            stats_update_interval=self.config.pipeline.coordination.stats_update_interval
        )
        
        return transform_config, ocr_config, pipeline_config