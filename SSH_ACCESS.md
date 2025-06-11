# 🔑 H100 SSH ACCESS INFORMATION

## ⚠️ MISSING: SSH Connection Details

**We need SSH access information for direct H100 connection:**

### 🔍 Required Information

```bash
# SSH connection format needed:
ssh -i [KEY_FILE] [USERNAME]@[H100_HOSTNAME] -p [PORT]

# Or if using password:
ssh [USERNAME]@[H100_HOSTNAME] -p [PORT]
```

### 📋 Information Needed from CloudExe

1. **SSH Hostname/IP**: `???` 
2. **SSH Port**: `???` (usually 22)
3. **Username**: `???` (likely `root`)
4. **Authentication**:
   - SSH key file path: `???`
   - OR password: `???`

### 🚀 CloudExe Dashboard

**Check your CloudExe dashboard for:**
- Instance connection details
- SSH access information  
- Terminal/console access
- Connection strings

### ⚡ Alternative: CloudExe Web Terminal

If SSH details unavailable, use CloudExe web terminal:
1. Go to CloudExe dashboard
2. Open instance terminal/console
3. Run setup commands directly

### 🔧 Once SSH Access Available

```bash
# Direct H100 setup via SSH
ssh [CONNECTION_DETAILS]

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

## 🎯 Action Required

**Need to obtain SSH connection details from CloudExe dashboard/documentation.**