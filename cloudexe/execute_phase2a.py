#!/usr/bin/env python3
"""
CloudExe H100 Execution Script for Phase 2A
Upload and execute enhanced ParScale-VAR implementation
"""

import requests
import json
import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Execute Phase 2A scripts on H100')
    parser.add_argument('--script', default='enhanced_parscale_var.py', 
                       help='Script to upload and execute')
    parser.add_argument('--api', action='store_true', 
                       help='Use API execution (default)')
    
    args = parser.parse_args()
    
    api_key = "eyJleHBpcnlfZGF0ZSI6IjIwMjYtMDYtMTEgMDA6MDA6MDAiLCJ1c2VyaWQiOiJjOGI5NmUxZS0xODVkLTRkNDUtOTY3Mi0xYTVmZTVjYjc0NGUifQ==.EhCkcWoPFzbU0IMg2jNlHU2Z2MaQnnXQeYof9x-UrWM="
    
    print("🚀 VAR-ParScale Phase 2A H100 Execution")
    print("="*50)
    
    # Determine script path
    if args.script.startswith('phase2a/'):
        script_path = Path("..") / args.script
    else:
        script_path = Path("../phase2a") / args.script
    
    print(f"📤 Uploading {script_path.name}...")
    
    if not script_path.exists():
        print(f"❌ Script not found at {script_path}!")
        return
    
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    upload_data = {
        "command": "upload_file", 
        "path": f"/root/{script_path.name}",
        "content": script_content
    }
    
    try:
        response = requests.post(
            "https://cloudexe.io/api/execute",
            headers={"Authorization": f"Bearer {api_key}"},
            json=upload_data,
            verify=False
        )
        
        if response.status_code == 200:
            print("✅ Upload successful")
            
            # Execute Phase 2A
            print("⚡ Executing Phase 2A implementation...")
            
            exec_data = {
                "command": f"cd /root && python3 {script_path.name}"
            }
            
            exec_response = requests.post(
                "https://cloudexe.io/api/execute", 
                headers={"Authorization": f"Bearer {api_key}"},
                json=exec_data,
                verify=False
            )
            
            if exec_response.status_code == 200:
                result = exec_response.json()
                print("\n" + "="*80)
                print("PHASE 2A RESULTS")
                print("="*80)
                print(result.get("output", ""))
                
                # Save results
                results_dir = Path("../results")
                results_dir.mkdir(exist_ok=True)
                
                with open(results_dir / "phase2a_h100_results.txt", "w") as f:
                    f.write(result.get("output", ""))
                
                print(f"\n💾 Results saved to {results_dir}/phase2a_h100_results.txt")
                
            else:
                print(f"❌ Execution failed: {exec_response.status_code}")
                print(exec_response.text)
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Try: curl -k -X POST https://cloudexe.io/api/execute")

if __name__ == "__main__":
    main()