#!/usr/bin/env python3
"""
H100 Environment Setup Script
Sets up VAR codebase and dependencies before Phase 2A execution
"""

import requests
import json

def setup_h100_environment():
    api_key = "eyJleHBpcnlfZGF0ZSI6IjIwMjYtMDYtMTEgMDA6MDA6MDAiLCJ1c2VyaWQiOiJjOGI5NmUxZS0xODVkLTRkNDUtOTY3Mi0xYTVmZTVjYjc0NGUifQ==.EhCkcWoPFzbU0IMg2jNlHU2Z2MaQnnXQeYof9x-UrWM="
    
    print("🔧 Setting up H100 Environment for VAR-ParScale")
    print("="*60)
    
    setup_commands = [
        {
            "name": "Clone VAR Repository",
            "command": "cd /root && git clone https://github.com/FoundationVision/VAR.git"
        },
        {
            "name": "Install PyTorch with CUDA",
            "command": "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall"
        },
        {
            "name": "Install Dependencies", 
            "command": "pip3 install huggingface_hub scipy numpy pillow"
        },
        {
            "name": "Test VAR Import",
            "command": "cd /root/VAR && python3 -c \"from models import build_vae_var; print('✅ VAR models imported successfully')\""
        },
        {
            "name": "Check GPU",
            "command": "python3 -c \"import torch; print(f'🔥 GPU: {torch.cuda.get_device_name(0)}'); print(f'💾 Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')\""
        },
        {
            "name": "Check VAR Models",
            "command": "ls -la /root/*.pth"
        }
    ]
    
    for step in setup_commands:
        print(f"\n🔄 {step['name']}...")
        
        try:
            response = requests.post(
                "https://cloudexe.io/api/execute",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"command": step["command"]},
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get("output", "")
                error = result.get("error", "")
                
                if output:
                    print(f"✅ {output}")
                if error:
                    print(f"⚠️ {error}")
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    print("\n🎯 H100 Environment Setup Complete!")
    print("Ready for Phase 2A execution")
    return True

if __name__ == "__main__":
    setup_h100_environment()