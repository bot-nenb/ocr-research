"""
Cost Analysis Module for OCR Processing

This module provides comprehensive cost analysis comparing local OCR processing
with cloud-based alternatives, including electricity costs, hardware depreciation,
and operational efficiency metrics.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class HardwareCosts:
    """Hardware cost configuration."""
    
    cpu_cost: float = 500.0  # Initial CPU cost in USD
    gpu_cost: float = 1200.0  # Initial GPU cost in USD (if applicable)
    memory_cost: float = 200.0  # RAM cost in USD
    storage_cost: float = 100.0  # Storage cost in USD
    other_components: float = 300.0  # Motherboard, PSU, etc.
    
    cpu_lifespan_years: float = 4.0
    gpu_lifespan_years: float = 3.0
    memory_lifespan_years: float = 5.0
    storage_lifespan_years: float = 5.0
    other_lifespan_years: float = 5.0


@dataclass
class PowerConsumption:
    """Power consumption configuration."""
    
    cpu_idle_watts: float = 15.0
    cpu_load_watts: float = 65.0
    gpu_idle_watts: float = 15.0
    gpu_load_watts: float = 150.0
    memory_watts: float = 10.0
    storage_watts: float = 5.0
    motherboard_watts: float = 20.0
    
    electricity_cost_kwh: float = 0.12  # USD per kWh


@dataclass
class CloudCosts:
    """Cloud API pricing configuration."""
    
    # Google Cloud Vision API (as of 2024)
    google_vision_per_1k: float = 1.50
    
    # AWS Textract
    aws_textract_per_1k: float = 1.50
    
    # Azure Computer Vision
    azure_vision_per_1k: float = 1.00
    
    # Microsoft Form Recognizer
    azure_forms_per_1k: float = 10.00


class CostAnalyzer:
    """Analyzes and compares OCR processing costs."""
    
    def __init__(self, 
                 hardware_costs: Optional[HardwareCosts] = None,
                 power_consumption: Optional[PowerConsumption] = None,
                 cloud_costs: Optional[CloudCosts] = None):
        """
        Initialize cost analyzer.
        
        Args:
            hardware_costs: Hardware cost configuration
            power_consumption: Power consumption configuration
            cloud_costs: Cloud API pricing configuration
        """
        self.hardware_costs = hardware_costs or HardwareCosts()
        self.power_consumption = power_consumption or PowerConsumption()
        self.cloud_costs = cloud_costs or CloudCosts()
        
        self.analysis_history: List[Dict] = []
    
    def calculate_hardware_depreciation_cost(self, processing_time_hours: float,
                                           uses_gpu: bool = False) -> Dict[str, float]:
        """
        Calculate hardware depreciation cost per processing session.
        
        Args:
            processing_time_hours: Total processing time in hours
            uses_gpu: Whether GPU was used for processing
            
        Returns:
            Dictionary with depreciation costs breakdown
        """
        # Calculate total hardware cost
        total_cost = (
            self.hardware_costs.cpu_cost +
            self.hardware_costs.memory_cost +
            self.hardware_costs.storage_cost +
            self.hardware_costs.other_components
        )
        
        if uses_gpu:
            total_cost += self.hardware_costs.gpu_cost
        
        # Calculate weighted average lifespan based on component costs
        component_lifespans = [
            (self.hardware_costs.cpu_cost, self.hardware_costs.cpu_lifespan_years),
            (self.hardware_costs.memory_cost, self.hardware_costs.memory_lifespan_years),
            (self.hardware_costs.storage_cost, self.hardware_costs.storage_lifespan_years),
            (self.hardware_costs.other_components, self.hardware_costs.other_lifespan_years)
        ]
        
        if uses_gpu:
            component_lifespans.append((self.hardware_costs.gpu_cost, self.hardware_costs.gpu_lifespan_years))
        
        # Calculate weighted average lifespan in hours
        total_weighted_lifespan = sum(cost * lifespan for cost, lifespan in component_lifespans)
        total_lifespan_hours = (total_weighted_lifespan / total_cost) * 365 * 24
        
        # Calculate hourly depreciation rate
        hourly_depreciation = total_cost / total_lifespan_hours
        
        # Calculate session depreciation cost
        session_cost = hourly_depreciation * processing_time_hours
        
        breakdown = {
            "total_hardware_cost": total_cost,
            "estimated_lifespan_hours": total_lifespan_hours,
            "hourly_depreciation_rate": hourly_depreciation,
            "session_depreciation_cost": session_cost
        }
        
        return breakdown
    
    def calculate_electricity_cost(self, processing_time_hours: float,
                                 cpu_utilization: float = 0.8,
                                 uses_gpu: bool = False,
                                 gpu_utilization: float = 0.7) -> Dict[str, float]:
        """
        Calculate electricity cost for processing session.
        
        Args:
            processing_time_hours: Total processing time in hours
            cpu_utilization: Average CPU utilization (0.0-1.0)
            uses_gpu: Whether GPU was used
            gpu_utilization: Average GPU utilization (0.0-1.0)
            
        Returns:
            Dictionary with electricity cost breakdown
        """
        # Calculate power consumption
        cpu_power = (
            self.power_consumption.cpu_idle_watts * (1 - cpu_utilization) +
            self.power_consumption.cpu_load_watts * cpu_utilization
        )
        
        total_power = (
            cpu_power +
            self.power_consumption.memory_watts +
            self.power_consumption.storage_watts +
            self.power_consumption.motherboard_watts
        )
        
        if uses_gpu:
            gpu_power = (
                self.power_consumption.gpu_idle_watts * (1 - gpu_utilization) +
                self.power_consumption.gpu_load_watts * gpu_utilization
            )
            total_power += gpu_power
        
        # Convert to kWh and calculate cost
        energy_kwh = (total_power / 1000) * processing_time_hours
        electricity_cost = energy_kwh * self.power_consumption.electricity_cost_kwh
        
        breakdown = {
            "cpu_power_watts": cpu_power,
            "gpu_power_watts": gpu_power if uses_gpu else 0,
            "total_power_watts": total_power,
            "energy_consumed_kwh": energy_kwh,
            "electricity_cost_usd": electricity_cost,
            "electricity_rate_per_kwh": self.power_consumption.electricity_cost_kwh
        }
        
        return breakdown
    
    def calculate_cloud_costs(self, num_documents: int) -> Dict[str, float]:
        """
        Calculate equivalent cloud processing costs.
        
        Args:
            num_documents: Number of documents processed
            
        Returns:
            Dictionary with cloud service costs
        """
        # Convert to cost per 1000 documents
        cost_multiplier = num_documents / 1000.0
        
        costs = {
            "google_vision_api": self.cloud_costs.google_vision_per_1k * cost_multiplier,
            "aws_textract": self.cloud_costs.aws_textract_per_1k * cost_multiplier,
            "azure_computer_vision": self.cloud_costs.azure_vision_per_1k * cost_multiplier,
            "azure_form_recognizer": self.cloud_costs.azure_forms_per_1k * cost_multiplier,
            "num_documents": num_documents,
            "documents_per_1k_unit": cost_multiplier
        }
        
        # Calculate average cloud cost
        api_costs = [
            costs["google_vision_api"],
            costs["aws_textract"],
            costs["azure_computer_vision"]
        ]
        costs["average_cloud_cost"] = sum(api_costs) / len(api_costs)
        costs["premium_cloud_cost"] = costs["azure_form_recognizer"]  # Most expensive
        
        return costs
    
    def analyze_processing_session(self, 
                                 processing_time_seconds: float,
                                 num_documents: int,
                                 cpu_utilization: float = 0.8,
                                 uses_gpu: bool = False,
                                 gpu_utilization: float = 0.7,
                                 session_name: str = None) -> Dict:
        """
        Perform complete cost analysis for a processing session.
        
        Args:
            processing_time_seconds: Total processing time in seconds
            num_documents: Number of documents processed
            cpu_utilization: Average CPU utilization
            uses_gpu: Whether GPU was used
            gpu_utilization: Average GPU utilization
            session_name: Optional name for this analysis
            
        Returns:
            Complete cost analysis results
        """
        processing_time_hours = processing_time_seconds / 3600.0
        
        # Calculate local costs
        hardware_costs = self.calculate_hardware_depreciation_cost(
            processing_time_hours, uses_gpu
        )
        
        electricity_costs = self.calculate_electricity_cost(
            processing_time_hours, cpu_utilization, uses_gpu, gpu_utilization
        )
        
        total_local_cost = (
            hardware_costs["session_depreciation_cost"] +
            electricity_costs["electricity_cost_usd"]
        )
        
        # Calculate cloud costs
        cloud_costs = self.calculate_cloud_costs(num_documents)
        
        # Calculate efficiency metrics
        cost_per_document_local = total_local_cost / num_documents if num_documents > 0 else 0
        cost_per_document_cloud_avg = cloud_costs["average_cloud_cost"] / num_documents if num_documents > 0 else 0
        
        processing_speed = num_documents / processing_time_seconds if processing_time_seconds > 0 else 0
        
        # Calculate savings
        savings_vs_avg_cloud = cloud_costs["average_cloud_cost"] - total_local_cost
        savings_percentage = (savings_vs_avg_cloud / cloud_costs["average_cloud_cost"] * 100) if cloud_costs["average_cloud_cost"] > 0 else 0
        
        # Create comprehensive analysis
        analysis = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "session_name": session_name or f"Analysis_{len(self.analysis_history) + 1}",
                "processing_time_seconds": processing_time_seconds,
                "processing_time_hours": processing_time_hours,
                "num_documents": num_documents,
                "uses_gpu": uses_gpu,
                "cpu_utilization": cpu_utilization,
                "gpu_utilization": gpu_utilization if uses_gpu else 0
            },
            
            "local_costs": {
                "hardware_depreciation": hardware_costs["session_depreciation_cost"],
                "electricity": electricity_costs["electricity_cost_usd"],
                "total_local_cost": total_local_cost,
                "cost_per_document": cost_per_document_local
            },
            
            "cloud_costs": cloud_costs,
            
            "performance_metrics": {
                "documents_per_second": processing_speed,
                "documents_per_minute": processing_speed * 60,
                "documents_per_hour": processing_speed * 3600,
                "seconds_per_document": processing_time_seconds / num_documents if num_documents > 0 else 0
            },
            
            "cost_comparison": {
                "local_vs_google_vision": {
                    "local_cost": total_local_cost,
                    "cloud_cost": cloud_costs["google_vision_api"],
                    "savings": cloud_costs["google_vision_api"] - total_local_cost,
                    "savings_percentage": ((cloud_costs["google_vision_api"] - total_local_cost) / cloud_costs["google_vision_api"] * 100) if cloud_costs["google_vision_api"] > 0 else 0
                },
                "local_vs_aws_textract": {
                    "local_cost": total_local_cost,
                    "cloud_cost": cloud_costs["aws_textract"],
                    "savings": cloud_costs["aws_textract"] - total_local_cost,
                    "savings_percentage": ((cloud_costs["aws_textract"] - total_local_cost) / cloud_costs["aws_textract"] * 100) if cloud_costs["aws_textract"] > 0 else 0
                },
                "local_vs_azure_vision": {
                    "local_cost": total_local_cost,
                    "cloud_cost": cloud_costs["azure_computer_vision"],
                    "savings": cloud_costs["azure_computer_vision"] - total_local_cost,
                    "savings_percentage": ((cloud_costs["azure_computer_vision"] - total_local_cost) / cloud_costs["azure_computer_vision"] * 100) if cloud_costs["azure_computer_vision"] > 0 else 0
                },
                "local_vs_average_cloud": {
                    "local_cost": total_local_cost,
                    "cloud_cost": cloud_costs["average_cloud_cost"],
                    "savings": savings_vs_avg_cloud,
                    "savings_percentage": savings_percentage
                }
            },
            
            "cost_efficiency": {
                "local_cost_per_1k_docs": cost_per_document_local * 1000,
                "cloud_cost_per_1k_docs_avg": cloud_costs["average_cloud_cost"] * (1000 / num_documents) if num_documents > 0 else 0,
                "efficiency_ratio": cloud_costs["average_cloud_cost"] / total_local_cost if total_local_cost > 0 else float('inf')
            },
            
            "detailed_breakdown": {
                "hardware_costs": hardware_costs,
                "electricity_costs": electricity_costs
            }
        }
        
        # Store analysis
        self.analysis_history.append(analysis)
        
        return analysis
    
    def project_volume_savings(self, documents_per_month: int, 
                             months: int = 12,
                             current_session_analysis: Dict = None) -> Dict:
        """
        Project cost savings over different volume scenarios.
        
        Args:
            documents_per_month: Projected monthly document volume
            months: Number of months to project
            current_session_analysis: Reference analysis for scaling
            
        Returns:
            Volume projection analysis
        """
        if current_session_analysis is None and not self.analysis_history:
            raise ValueError("Need at least one analysis session for projection")
        
        ref_analysis = current_session_analysis or self.analysis_history[-1]
        
        # Calculate scaling factors
        ref_docs = ref_analysis["session_info"]["num_documents"]
        ref_time = ref_analysis["session_info"]["processing_time_seconds"]
        
        total_documents = documents_per_month * months
        scaling_factor = total_documents / ref_docs
        
        # Project processing time (assuming linear scaling)
        projected_time_seconds = ref_time * scaling_factor
        projected_time_hours = projected_time_seconds / 3600
        
        # Project costs
        projected_local_cost = ref_analysis["local_costs"]["total_local_cost"] * scaling_factor
        projected_cloud_cost_avg = ref_analysis["cloud_costs"]["average_cloud_cost"] * scaling_factor
        
        # Calculate total savings
        total_savings = projected_cloud_cost_avg - projected_local_cost
        monthly_savings = total_savings / months
        
        projection = {
            "projection_parameters": {
                "documents_per_month": documents_per_month,
                "total_months": months,
                "total_documents": total_documents,
                "scaling_factor": scaling_factor
            },
            
            "projected_local_processing": {
                "total_time_hours": projected_time_hours,
                "total_time_days": projected_time_hours / 24,
                "total_cost": projected_local_cost,
                "monthly_cost": projected_local_cost / months,
                "cost_per_document": projected_local_cost / total_documents
            },
            
            "projected_cloud_costs": {
                "total_cost_avg": projected_cloud_cost_avg,
                "monthly_cost_avg": projected_cloud_cost_avg / months,
                "cost_per_document": projected_cloud_cost_avg / total_documents
            },
            
            "projected_savings": {
                "total_savings": total_savings,
                "monthly_savings": monthly_savings,
                "savings_percentage": (total_savings / projected_cloud_cost_avg * 100) if projected_cloud_cost_avg > 0 else 0,
                "payback_period_months": (
                    (self.hardware_costs.cpu_cost + 
                     (self.hardware_costs.gpu_cost if ref_analysis["session_info"]["uses_gpu"] else 0) +
                     self.hardware_costs.memory_cost + 
                     self.hardware_costs.storage_cost + 
                     self.hardware_costs.other_components) / monthly_savings
                ) if monthly_savings > 0 else float('inf')
            }
        }
        
        return projection
    
    def generate_cost_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive cost analysis report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Complete cost analysis report
        """
        if not self.analysis_history:
            return {"error": "No analysis sessions available"}
        
        # Aggregate statistics
        total_documents = sum(a["session_info"]["num_documents"] for a in self.analysis_history)
        total_time = sum(a["session_info"]["processing_time_seconds"] for a in self.analysis_history)
        total_local_cost = sum(a["local_costs"]["total_local_cost"] for a in self.analysis_history)
        total_cloud_cost_avg = sum(a["cloud_costs"]["average_cloud_cost"] for a in self.analysis_history)
        
        # Calculate averages
        avg_cost_per_doc_local = total_local_cost / total_documents if total_documents > 0 else 0
        avg_cost_per_doc_cloud = total_cloud_cost_avg / total_documents if total_documents > 0 else 0
        
        report = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "total_sessions": len(self.analysis_history),
                "total_documents_processed": total_documents,
                "total_processing_time_hours": total_time / 3600
            },
            
            "aggregate_costs": {
                "total_local_cost": total_local_cost,
                "total_cloud_cost_average": total_cloud_cost_avg,
                "total_savings": total_cloud_cost_avg - total_local_cost,
                "savings_percentage": ((total_cloud_cost_avg - total_local_cost) / total_cloud_cost_avg * 100) if total_cloud_cost_avg > 0 else 0
            },
            
            "per_document_costs": {
                "local_cost_per_document": avg_cost_per_doc_local,
                "cloud_cost_per_document": avg_cost_per_doc_cloud,
                "savings_per_document": avg_cost_per_doc_cloud - avg_cost_per_doc_local
            },
            
            "sessions": self.analysis_history
        }
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Cost analysis report saved to {output_file}")
        
        return report
    
    def print_analysis_summary(self, analysis: Dict = None):
        """
        Print formatted analysis summary to console.
        
        Args:
            analysis: Analysis to print (uses latest if not provided)
        """
        if analysis is None:
            if not self.analysis_history:
                print("No analysis available")
                return
            analysis = self.analysis_history[-1]
        
        print("\n" + "=" * 80)
        print("OCR COST ANALYSIS SUMMARY")
        print("=" * 80)
        
        session = analysis["session_info"]
        local = analysis["local_costs"]
        performance = analysis["performance_metrics"]
        
        print(f"Session: {session['session_name']}")
        print(f"Documents processed: {session['num_documents']}")
        print(f"Processing time: {session['processing_time_hours']:.2f} hours")
        print(f"Device: {'GPU + CPU' if session['uses_gpu'] else 'CPU only'}")
        
        print("\nPerformance:")
        print(f"  Speed: {performance['documents_per_minute']:.1f} docs/minute")
        print(f"  Time per document: {performance['seconds_per_document']:.2f} seconds")
        
        print(f"\nLocal Processing Costs:")
        print(f"  Hardware depreciation: ${local['hardware_depreciation']:.4f}")
        print(f"  Electricity: ${local['electricity']:.4f}")
        print(f"  Total local cost: ${local['total_local_cost']:.4f}")
        print(f"  Cost per document: ${local['cost_per_document']:.4f}")
        
        print(f"\nCloud API Comparison:")
        for service, comparison in analysis["cost_comparison"].items():
            if service == "local_vs_average_cloud":
                continue
            service_name = service.replace("local_vs_", "").replace("_", " ").title()
            print(f"  {service_name}:")
            print(f"    Cloud cost: ${comparison['cloud_cost']:.4f}")
            print(f"    Savings: ${comparison['savings']:.4f} ({comparison['savings_percentage']:.1f}%)")
        
        avg_comparison = analysis["cost_comparison"]["local_vs_average_cloud"]
        print(f"\nOverall Savings vs Average Cloud:")
        print(f"  Local: ${avg_comparison['local_cost']:.4f}")
        print(f"  Average Cloud: ${avg_comparison['cloud_cost']:.4f}")
        print(f"  Savings: ${avg_comparison['savings']:.4f} ({avg_comparison['savings_percentage']:.1f}%)")
        
        efficiency = analysis["cost_efficiency"]
        print(f"\nCost Efficiency:")
        print(f"  Local cost per 1K docs: ${efficiency['local_cost_per_1k_docs']:.2f}")
        print(f"  Cloud cost per 1K docs: ${efficiency['cloud_cost_per_1k_docs_avg']:.2f}")
        print(f"  Efficiency ratio: {efficiency['efficiency_ratio']:.1f}x cheaper than cloud")
        
        print("=" * 80)