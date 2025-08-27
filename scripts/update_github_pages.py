#!/usr/bin/env python3
"""
Auto-update GitHub Pages with new OCR reports
This script scans for new reports and updates the docs structure automatically
"""

import os
import sys
import json
import glob
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
TASKS_DIR = PROJECT_ROOT / "tasks"
RESULTS_DIR = PROJECT_ROOT / "results"

def find_task_directories() -> List[Path]:
    """Find all task directories in the tasks folder"""
    task_dirs = []
    if TASKS_DIR.exists():
        for task_dir in TASKS_DIR.iterdir():
            if task_dir.is_dir() and task_dir.name.startswith('task-'):
                task_dirs.append(task_dir)
    return sorted(task_dirs)

def find_reports_in_task(task_dir: Path) -> List[Dict[str, Any]]:
    """Find all OCR reports in a task directory"""
    reports = []
    
    # Look for reports in the results directory
    results_paths = [
        task_dir / "results",
        task_dir / "demo" / "results",
        RESULTS_DIR  # Global results directory
    ]
    
    for results_path in results_paths:
        if results_path.exists():
            # Find HTML reports
            html_reports = list(results_path.glob("**/ocr_processing_report_*.html"))
            for report in html_reports:
                reports.append({
                    'type': 'html',
                    'path': report,
                    'name': report.name,
                    'task': task_dir.name,
                    'timestamp': datetime.fromtimestamp(report.stat().st_mtime)
                })
            
            # Find JSON summaries
            json_reports = list(results_path.glob("**/executive_summary.json"))
            json_reports.extend(results_path.glob("**/complete_results.json"))
            for report in json_reports:
                reports.append({
                    'type': 'json',
                    'path': report,
                    'name': report.name,
                    'task': task_dir.name,
                    'timestamp': datetime.fromtimestamp(report.stat().st_mtime)
                })
            
            # Find CSV reports
            csv_reports = list(results_path.glob("**/results_summary_*.csv"))
            for report in csv_reports:
                reports.append({
                    'type': 'csv',
                    'path': report,
                    'name': report.name,
                    'task': task_dir.name,
                    'timestamp': datetime.fromtimestamp(report.stat().st_mtime)
                })
    
    return reports

def copy_task_results_to_docs(task_dir: Path, reports: List[Dict[str, Any]]) -> None:
    """Copy task results to the appropriate docs directory"""
    task_name = task_dir.name
    docs_task_dir = DOCS_DIR / "demos" / task_name
    docs_task_dir.mkdir(parents=True, exist_ok=True)
    
    # Group reports by their parent directory to maintain structure
    source_dirs = set()
    for report in reports:
        source_dirs.add(report['path'].parent)
    
    for source_dir in source_dirs:
        # Copy entire results structure
        if source_dir.name == "results":
            dest_dir = docs_task_dir
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, dest_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)

def update_landing_page(tasks_info: Dict[str, Any]) -> None:
    """Update the main landing page with task information"""
    landing_page = DOCS_DIR / "index.html"
    
    # Read current landing page
    if landing_page.exists():
        with open(landing_page, 'r') as f:
            content = f.read()
        
        # Extract and update task cards dynamically
        # For now, we'll keep the existing structure but this could be enhanced
        # to automatically generate task cards based on discovered tasks
        
        print(f"Landing page exists at {landing_page}")
        print(f"Found {len(tasks_info)} tasks to potentially update")

def create_task_index_page(task_name: str, reports: List[Dict[str, Any]]) -> None:
    """Create or update the index page for a specific task"""
    docs_task_dir = DOCS_DIR / "demos" / task_name
    docs_task_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = docs_task_dir / "index.html"
    
    # Get the latest HTML report for the direct link
    html_reports = [r for r in reports if r['type'] == 'html']
    latest_html_report = max(html_reports, key=lambda x: x['timestamp']) if html_reports else None
    
    # Task display name
    task_display_name = task_name.replace('-', ' ').replace('task', 'Task').title()
    
    # Create index content (simplified version, could be enhanced)
    index_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{task_display_name} - OCR Research</title>
    <style>
        /* Copy styles from main template - simplified for auto-generation */
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .breadcrumb {{ margin: 1rem 0; }}
        .breadcrumb a {{ color: #667eea; text-decoration: none; }}
        h1 {{ color: #2d3748; }}
        .report-links {{ margin: 2rem 0; }}
        .report-links a {{ 
            display: inline-block; 
            margin: 0.5rem 1rem 0.5rem 0; 
            padding: 0.5rem 1rem; 
            background: #667eea; 
            color: white; 
            text-decoration: none; 
            border-radius: 5px; 
        }}
        .report-links a:hover {{ background: #5a6fd8; }}
    </style>
</head>
<body>
    <div class="breadcrumb">
        <a href="../..">← Back to OCR Research Project</a>
    </div>
    
    <h1>{task_display_name}</h1>
    
    <div class="report-links">
"""
    
    # Add links to reports
    if latest_html_report:
        index_content += f'        <a href="{latest_html_report["name"]}">Latest HTML Report</a>\n'
    
    for report in reports:
        if report['type'] == 'json' and 'executive' in report['name']:
            index_content += f'        <a href="{report["name"]}">Executive Summary (JSON)</a>\n'
        elif report['type'] == 'csv':
            index_content += f'        <a href="{report["name"]}">Results Summary (CSV)</a>\n'
    
    index_content += """    </div>
    
    <p>This page was automatically generated based on discovered reports.</p>
    
</body>
</html>"""
    
    with open(index_file, 'w') as f:
        f.write(index_content)
    
    print(f"Created/updated index for {task_name} at {index_file}")

def main():
    """Main function to update GitHub Pages structure"""
    print("🔍 Scanning for OCR tasks and reports...")
    
    # Find all tasks
    task_dirs = find_task_directories()
    print(f"Found {len(task_dirs)} task directories: {[t.name for t in task_dirs]}")
    
    # Also check for reports in the global results directory
    if RESULTS_DIR.exists():
        global_reports = find_reports_in_task(Path("results"))
        if global_reports:
            print(f"Found {len(global_reports)} reports in global results directory")
    
    tasks_info = {}
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        print(f"\n📋 Processing {task_name}...")
        
        # Find reports for this task
        reports = find_reports_in_task(task_dir)
        print(f"Found {len(reports)} reports for {task_name}")
        
        if reports:
            # Copy results to docs
            copy_task_results_to_docs(task_dir, reports)
            
            # Create/update task index page
            create_task_index_page(task_name, reports)
            
            # Store task info
            tasks_info[task_name] = {
                'reports': reports,
                'latest_report': max(reports, key=lambda x: x['timestamp']) if reports else None
            }
            
            print(f"✅ Updated docs for {task_name}")
        else:
            print(f"⚠️  No reports found for {task_name}")
    
    # Update landing page
    update_landing_page(tasks_info)
    
    print(f"\n🎉 GitHub Pages update complete!")
    print(f"📁 Docs directory: {DOCS_DIR}")
    print(f"🌐 Access your site at: https://<username>.github.io/<repo-name>/")

if __name__ == "__main__":
    main()