# 🔑 H100 SSH CONNECTION DETAILS

## 📡 CloudExe Instance Access

**SSH Command:**
```bash
ssh -p 11292 root@inst-gw.cloudexe.tech
```

**Instance Details:**
- **Instance Name**: `cmu-scs-Peter`
- **SSH Port**: `11292`
- **Username**: `root`
- **Hostname**: `inst-gw.cloudexe.tech`
- **Web Access**: https://inst-gw.cloudexe.tech/inst/11293/
- **Web Password**: `6LCS45HN`

## 🚀 Direct Setup & Execution

### Option 1: Automated SSH Setup
```bash
cd /Users/peter/VAR-ParScale
./cloudexe/ssh_setup.sh "ssh -p 11292 root@inst-gw.cloudexe.tech"
```

### Option 2: Manual SSH Setup
```bash
# Connect to H100 instance
ssh -p 11292 root@inst-gw.cloudexe.tech

# On H100 instance:
cd /root
git clone https://github.com/FoundationVision/VAR.git
git clone https://github.com/peteryuqin/VAR-ParScale.git

# Install dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install huggingface_hub scipy numpy pillow

# Execute Phase 2A
cd VAR-ParScale
python3 phase2a/enhanced_parscale_var.py
```

### Option 3: One-Command Execution
```bash
# Execute Phase 2A directly via SSH
ssh -p 11292 root@inst-gw.cloudexe.tech "cd /root/VAR-ParScale && python3 phase2a/enhanced_parscale_var.py"
```

## 🎯 For CloudExe Launcher (GPU Fleet Access)

**Note**: The SSH instance has limited GPU. For full H100 access, use CloudExe launcher:

```bash
# On the SSH instance, use CloudExe launcher for full GPU:
cloudexe --gpuspec H100x1 -- /usr/bin/python3 /root/VAR-ParScale/phase2a/enhanced_parscale_var.py
```

## 📊 Instance Info

- **Created**: 2025-06-08 15:22:52
- **GPU**: Limited (setup environment)
- **Full GPU**: Use `cloudexe --gpuspec H100x1` launcher
- **Storage**: Persistent `/root` directory