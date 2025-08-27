#!/usr/bin/env python3
"""
Automated link testing script for GitHub Pages
Tests all links in HTML files to ensure they resolve correctly
"""

import os
import sys
import re
import urllib.parse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
GITHUB_RAW_BASE = "https://github.com/bot-nenb/ocr-research/raw/master/"

class LinkTester:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.errors = []
        self.warnings = []
        
    def extract_links_from_html(self, html_file: Path) -> List[Dict]:
        """Extract all links from an HTML file"""
        links = []
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find href attributes
            href_pattern = r'href=["\']([^"\']*)["\']'
            matches = re.finditer(href_pattern, content)
            
            for match in matches:
                href = match.group(1)
                if href and not href.startswith('#'):  # Skip anchor links
                    links.append({
                        'url': href,
                        'file': html_file,
                        'type': self.classify_link(href)
                    })
            
            # Find src attributes for images
            src_pattern = r'src=["\']([^"\']*)["\']'
            matches = re.finditer(src_pattern, content)
            
            for match in matches:
                src = match.group(1)
                if src:
                    links.append({
                        'url': src,
                        'file': html_file,
                        'type': self.classify_link(src)
                    })
                    
        except Exception as e:
            self.errors.append(f"Error reading {html_file}: {e}")
        
        return links
    
    def classify_link(self, url: str) -> str:
        """Classify the type of link"""
        if url.startswith('http'):
            return 'external'
        elif url.startswith('//'):
            return 'protocol-relative'
        elif url.startswith('#'):
            return 'anchor'
        else:
            return 'relative'
    
    def resolve_relative_path(self, base_file: Path, relative_url: str) -> Path:
        """Resolve a relative URL to an absolute file path"""
        # Remove query parameters and fragments
        clean_url = relative_url.split('?')[0].split('#')[0]
        
        # Handle relative paths
        if clean_url.startswith('/'):
            # Absolute path from docs root
            return self.docs_dir / clean_url.lstrip('/')
        else:
            # Relative path from current file
            return (base_file.parent / clean_url).resolve()
    
    def test_relative_link(self, link_info: Dict) -> bool:
        """Test if a relative link resolves to an existing file"""
        base_file = link_info['file']
        url = link_info['url']
        
        try:
            target_path = self.resolve_relative_path(base_file, url)
            
            # Check if file exists
            if target_path.exists():
                return True
            else:
                # Check if it might be a GitHub raw URL that should exist
                relative_to_docs = target_path.relative_to(self.docs_dir.parent)
                github_url = GITHUB_RAW_BASE + str(relative_to_docs).replace('\\', '/')
                
                self.errors.append(f"❌ Broken link in {base_file.relative_to(self.docs_dir)}: {url}")
                self.errors.append(f"   Expected file: {target_path}")
                self.errors.append(f"   GitHub raw URL would be: {github_url}")
                return False
                
        except Exception as e:
            self.errors.append(f"❌ Error resolving link in {base_file.relative_to(self.docs_dir)}: {url} - {e}")
            return False
    
    def test_external_link(self, link_info: Dict) -> bool:
        """Test external links (for now, just validate they're properly formatted)"""
        url = link_info['url']
        
        # Basic URL validation
        if url.startswith('http') and '://' in url:
            return True
        else:
            self.warnings.append(f"⚠️  Malformed external URL in {link_info['file'].relative_to(self.docs_dir)}: {url}")
            return False
    
    def find_html_files(self) -> List[Path]:
        """Find all HTML files in the docs directory"""
        html_files = []
        
        for path in self.docs_dir.rglob('*.html'):
            html_files.append(path)
        
        return sorted(html_files)
    
    def test_all_links(self) -> bool:
        """Test all links in all HTML files"""
        print("🔍 Scanning for HTML files...")
        html_files = self.find_html_files()
        print(f"Found {len(html_files)} HTML files")
        
        all_links = []
        relative_links = []
        external_links = []
        
        # Extract all links
        for html_file in html_files:
            print(f"📄 Scanning {html_file.relative_to(self.docs_dir)}")
            links = self.extract_links_from_html(html_file)
            all_links.extend(links)
            
            for link in links:
                if link['type'] == 'relative':
                    relative_links.append(link)
                elif link['type'] == 'external':
                    external_links.append(link)
        
        print(f"\n📊 Link Statistics:")
        print(f"   Total links: {len(all_links)}")
        print(f"   Relative links: {len(relative_links)}")
        print(f"   External links: {len(external_links)}")
        
        # Test relative links
        print(f"\n🔗 Testing {len(relative_links)} relative links...")
        relative_success = 0
        for link in relative_links:
            if self.test_relative_link(link):
                relative_success += 1
                print(f"✅ {link['url']} in {link['file'].name}")
            else:
                print(f"❌ {link['url']} in {link['file'].name}")
        
        # Test external links
        print(f"\n🌐 Validating {len(external_links)} external links...")
        external_success = 0
        for link in external_links:
            if self.test_external_link(link):
                external_success += 1
                print(f"✅ {link['url'][:60]}{'...' if len(link['url']) > 60 else ''}")
        
        # Summary
        print(f"\n📋 Test Results:")
        print(f"   Relative links: {relative_success}/{len(relative_links)} passed")
        print(f"   External links: {external_success}/{len(external_links)} passed")
        
        if self.errors:
            print(f"\n❌ {len(self.errors)} Errors:")
            for error in self.errors:
                print(f"   {error}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} Warnings:")
            for warning in self.warnings:
                print(f"   {warning}")
        
        # Return True if no errors
        return len(self.errors) == 0
    
    def generate_report(self) -> Dict:
        """Generate a test report"""
        html_files = self.find_html_files()
        all_links = []
        
        for html_file in html_files:
            links = self.extract_links_from_html(html_file)
            all_links.extend(links)
        
        return {
            'timestamp': os.popen('date').read().strip(),
            'html_files_count': len(html_files),
            'total_links': len(all_links),
            'relative_links': len([l for l in all_links if l['type'] == 'relative']),
            'external_links': len([l for l in all_links if l['type'] == 'external']),
            'errors': self.errors,
            'warnings': self.warnings,
            'passed': len(self.errors) == 0
        }

def main():
    """Main function"""
    print("🧪 GitHub Pages Link Tester")
    print("=" * 50)
    
    if not DOCS_DIR.exists():
        print(f"❌ Docs directory not found: {DOCS_DIR}")
        sys.exit(1)
    
    tester = LinkTester(DOCS_DIR)
    
    # Test all links
    success = tester.test_all_links()
    
    # Generate report
    report = tester.generate_report()
    
    # Save report
    report_file = PROJECT_ROOT / "link_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    # Exit with appropriate code
    if success:
        print("\n✅ All tests passed! Links are working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed! Please fix the broken links before pushing.")
        sys.exit(1)

if __name__ == "__main__":
    main()