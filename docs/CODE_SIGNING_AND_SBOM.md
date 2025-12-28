# Code Signing & Release Security

## Overview

This document provides comprehensive guidance on code signing, certificate management, and secure releases for the AI Platform project.

---

## 1. Code Signing with GPG

### 1.1 GPG Key Generation

**Generate a new GPG key pair:**

```bash
# Interactive key generation
gpg --full-generate-key

# Or with preset options
gpg --batch --generate-key << EOF
%no-protection
Key-Type: RSA
Key-Length: 4096
Name-Real: AI Platform Release
Name-Email: releases@aiplatform.io
Name-Comment: Official releases
Expire-Date: 2y
EOF

# View generated keys
gpg --list-keys
gpg --list-secret-keys
```

**Key specifications:**
- Key type: RSA 4096-bit (or ED25519 for newer systems)
- Validity: 2 years (auto-rotate)
- Subkey for signing: RSA 4096-bit
- Subkey for encryption: RSA 4096-bit

### 1.2 Key Management

**Export public key to repository:**

```bash
# Export public key
gpg --armor --export releases@aiplatform.io > public.asc

# Upload to keyserver
gpg --send-keys KEY_ID
gpg --send-keys --keyserver keyserver.ubuntu.com KEY_ID

# Verify key on keyserver
gpg --recv-keys KEY_ID
```

**Secure key storage:**

```bash
# Create encrypted backup
gpg --export-secret-keys releases@aiplatform.io | gpg --armor > secret.asc.gpg
# Encrypt with strong passphrase

# Store securely:
# - Hardware security module (HSM)
# - Azure Key Vault
# - AWS Secrets Manager
# - Vault (HashiCorp)

# Never store in git or CI/CD logs
```

### 1.3 Signing Releases

**Sign tar.gz release:**

```bash
# Create release archive
tar -czf aiplatform-v1.0.0.tar.gz aiplatform/

# Sign the archive
gpg --armor --sign --detach-sig aiplatform-v1.0.0.tar.gz
# Creates: aiplatform-v1.0.0.tar.gz.asc

# Sign with fingerprint
gpg --default-key 'releases@aiplatform.io' \
    --armor --sign --detach-sig \
    aiplatform-v1.0.0.tar.gz

# Create SHA256 checksum
sha256sum aiplatform-v1.0.0.tar.gz > aiplatform-v1.0.0.tar.gz.sha256

# Sign checksum
gpg --armor --sign --detach-sig \
    aiplatform-v1.0.0.tar.gz.sha256
```

**Verify signatures:**

```bash
# Verify detached signature
gpg --verify aiplatform-v1.0.0.tar.gz.asc aiplatform-v1.0.0.tar.gz

# Verify checksum
sha256sum -c aiplatform-v1.0.0.tar.gz.sha256

# Verify signed checksum
gpg --verify aiplatform-v1.0.0.tar.gz.sha256.asc
```

### 1.4 Signed Git Commits

**Configure Git for signing:**

```bash
# Set default signing key
git config --global user.signingkey KEY_ID

# Enable signing by default
git config --global commit.gpgSign true
git config --global tag.gpgSign true

# For single repository
git config commit.gpgSign true
git config tag.gpgSign true
```

**Create signed commits:**

```bash
# Sign individual commit
git commit -S -m "Release v1.0.0"

# Force signature (if not default)
git commit --gpg-sign -m "Release v1.0.0"

# View signature
git show HEAD --show-signature
git log --pretty=fuller --show-signature
```

**Create signed tags:**

```bash
# Sign release tag
git tag -s -m "Release v1.0.0" v1.0.0

# Sign existing tag
git tag -s -f v1.0.0

# Verify tag signature
git tag -v v1.0.0

# View all signed tags
git tag -v $(git tag -l)
```

---

## 2. X.509 Certificates for Code Signing

### 2.1 Certificate Acquisition

**Options:**

1. **Extended Validation (EV) Code Signing Certificate**
   - Provider: DigiCert, Sectigo, GlobalSign
   - Cost: $200-600/year
   - Pros: High trust, browser recognition
   - Cons: Expensive, manual verification

2. **Organization Validation (OV) Code Signing Certificate**
   - Provider: Let's Encrypt (free), DigiCert, Sectigo
   - Cost: Free or $100-300/year
   - Pros: Good balance, widely accepted
   - Cons: Less prominent than EV

3. **Self-Signed Certificates** (for testing)
   - Cost: Free
   - Pros: No dependencies
   - Cons: No chain of trust, not for production

### 2.2 Generate Self-Signed Certificate

```bash
# Generate private key and certificate (valid 365 days)
openssl req -x509 -newkey rsa:4096 -keyout private.key -out certificate.crt -days 365 -nodes

# With options specified
openssl req -x509 -newkey rsa:4096 \
    -keyout aiplatform-private.key \
    -out aiplatform-certificate.crt \
    -days 365 -nodes \
    -subj "/CN=AI Platform/O=AI Platform Inc/C=US"

# Create PKCS#12 (PFX) file for Windows
openssl pkcs12 -export \
    -in aiplatform-certificate.crt \
    -inkey aiplatform-private.key \
    -out aiplatform.pfx \
    -name "AI Platform Code Signing"
```

### 2.3 Code Signing with Certificate

**Windows Authenticode (signtool):**

```bash
# Sign executable
signtool sign /f aiplatform.pfx /p PASSWORD /t http://timestamp.server.com ^
    /d "AI Platform Quantum Computing Engine" ^
    aiplatform-installer.exe

# Verify signature
signtool verify /pa aiplatform-installer.exe
```

**Java Code Signing (jarsigner):**

```bash
# Sign JAR file
jarsigner -keystore aiplatform.pfx \
    -storepass PASSWORD \
    -tsa http://timestamp.server.com \
    aiplatform.jar \
    "AI Platform"

# Verify signature
jarsigner -verify -verbose aiplatform.jar
```

**Python Wheel Signing:**

```bash
# Use GPG for Python packages (PEP 478)
gpg --armor --sign --detach-sig \
    aiplatform-1.0.0-py3-none-any.whl

# Verify wheel
gpg --verify aiplatform-1.0.0-py3-none-any.whl.asc
```

### 2.4 Timestamp Server Configuration

```bash
# Trusted timestamp servers
- http://timestamp.digicert.com
- http://timestamp.sectigo.com
- http://timestamp.globalsign.com
- http://rfc3161.globalgsign.com/
```

---

## 3. SBOM (Software Bill of Materials)

### 3.1 SBOM Generation

**Using CycloneDX (Python):**

```bash
# Install CycloneDX
pip install cyclonedx-bom

# Generate SBOM from requirements.txt
cyclonedx-bom \
    --input requirements.txt \
    --output bom.xml \
    --format xml

# Generate SBOM in JSON format
cyclonedx-bom \
    --input requirements.txt \
    --output bom.json \
    --format json

# Generate from installed packages
pip install pipdeptree
cyclonedx-bom --format json > bom.json
```

**Using SPDX (SBOM 2.3):**

```bash
# Install SPDX tools
pip install spdx-tools

# Generate SPDX SBOM
spdx-tools create \
    -n "AI Platform" \
    --version 1.0.0 \
    --output ai-platform-sbom.spdx
```

**Using Syft (any language):**

```bash
# Install Syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SBOM
syft aiplatform:latest -o spdx > sbom.spdx
syft aiplatform:latest -o cyclonedx > sbom.xml
syft . -o json > sbom.json
```

### 3.2 SBOM Content

**Minimum required fields:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<bom xmlns="http://cyclonedx.org/schema/bom/1.3" version="1">
  <metadata>
    <timestamp>2024-01-01T00:00:00Z</timestamp>
    <tools>
      <tool>
        <vendor>CycloneDX</vendor>
        <name>cyclonedx-bom</name>
        <version>3.11.0</version>
      </tool>
    </tools>
    <component>
      <name>AI Platform</name>
      <version>1.0.0</version>
      <description>AI Platform Quantum Computing Engine</description>
      <type>application</type>
    </component>
  </metadata>
  
  <components>
    <!-- Framework -->
    <component type="library">
      <name>Flask</name>
      <version>2.3.0</version>
      <purl>pkg:pypi/flask@2.3.0</purl>
      <licenses>
        <license>
          <name>BSD-3-Clause</name>
        </license>
      </licenses>
    </component>
    
    <!-- Quantum -->
    <component type="library">
      <name>Qiskit</name>
      <version>0.41.0</version>
      <purl>pkg:pypi/qiskit@0.41.0</purl>
      <licenses>
        <license>
          <name>Apache-2.0</name>
        </license>
      </licenses>
    </component>
    
    <!-- Data Science -->
    <component type="library">
      <name>NumPy</name>
      <version>1.24.0</version>
      <purl>pkg:pypi/numpy@1.24.0</purl>
      <licenses>
        <license>
          <name>BSD-3-Clause</name>
        </license>
      </licenses>
    </component>
  </components>
</bom>
```

### 3.3 SBOM Validation

```bash
# Validate CycloneDX SBOM
cyclonedx-cli validate --input-file bom.xml --input-format xml

# Validate SPDX SBOM
spdx-tools validate -i sbom.spdx

# Check for known vulnerabilities
pip install pip-audit
pip-audit --sbom bom.json

# Check licenses
pip install pip-licenses
pip-licenses --format=json > licenses.json
```

### 3.4 SBOM in CI/CD Pipeline

**GitHub Actions workflow:**

```yaml
name: Generate SBOM

on:
  release:
    types: [published]

jobs:
  sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install cyclonedx-bom
      
      - name: Generate SBOM
        run: |
          cyclonedx-bom -o bom.xml
          cyclonedx-bom -o bom.json --format json
      
      - name: Upload SBOM to release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            bom.xml
            bom.json
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 4. Release Process

### 4.1 Pre-Release Checklist

```bash
#!/bin/bash
# pre_release_checks.sh

set -e

echo "=== Pre-Release Security Checks ==="

# 1. Verify branch is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Working directory not clean"
    exit 1
fi

# 2. Check for credentials in code
if git grep -i -E 'password|secret|api.?key|token' -- '*.py' '*.js' '*.env'; then
    echo "❌ Potential credentials found in code"
    exit 1
fi

# 3. Run security scans
echo "Running Bandit (Python security)..."
bandit -r aiplatform/ -f json -o bandit-report.json

echo "Running pip-audit..."
pip-audit > pip-audit-report.txt

# 4. Verify no unresolved vulnerabilities
echo "Checking vulnerability score..."
pip-audit --output json | jq '.vulnerabilities | length'

# 5. Generate SBOM
echo "Generating SBOM..."
cyclonedx-bom -o bom.xml

# 6. Run tests
echo "Running test suite..."
pytest tests/ -v

# 7. Verify signatures
echo "Verifying GPG key..."
gpg --list-keys releases@aiplatform.io

echo "✅ All pre-release checks passed"
```

### 4.2 Release Signing

```bash
#!/bin/bash
# sign_release.sh

VERSION=$1
GPG_KEY="releases@aiplatform.io"
TIMESTAMP_SERVER="http://timestamp.digicert.com"

if [ -z "$VERSION" ]; then
    echo "Usage: ./sign_release.sh <version>"
    exit 1
fi

echo "=== Signing Release v${VERSION} ==="

# Create release artifacts
tar -czf aiplatform-v${VERSION}.tar.gz aiplatform/
zip -r aiplatform-v${VERSION}.zip aiplatform/

# Sign artifacts
echo "Signing tarball..."
gpg --default-key "$GPG_KEY" \
    --armor --sign --detach-sig \
    aiplatform-v${VERSION}.tar.gz

echo "Signing zip..."
gpg --default-key "$GPG_KEY" \
    --armor --sign --detach-sig \
    aiplatform-v${VERSION}.zip

# Create checksums
echo "Creating checksums..."
sha256sum aiplatform-v${VERSION}.tar.gz > aiplatform-v${VERSION}.tar.gz.sha256
sha256sum aiplatform-v${VERSION}.zip > aiplatform-v${VERSION}.zip.sha256

# Sign checksums
gpg --default-key "$GPG_KEY" \
    --armor --sign --detach-sig \
    aiplatform-v${VERSION}.tar.gz.sha256

gpg --default-key "$GPG_KEY" \
    --armor --sign --detach-sig \
    aiplatform-v${VERSION}.zip.sha256

# Create signed tag
git tag -s -m "Release v${VERSION}" v${VERSION}

# Sign and tag in git
git push origin v${VERSION}

echo "✅ Release signed successfully"
echo ""
echo "Artifacts:"
echo "  - aiplatform-v${VERSION}.tar.gz"
echo "  - aiplatform-v${VERSION}.tar.gz.asc"
echo "  - aiplatform-v${VERSION}.tar.gz.sha256"
echo "  - aiplatform-v${VERSION}.tar.gz.sha256.asc"
```

### 4.3 Release Verification

**User verification process:**

```bash
#!/bin/bash
# verify_release.sh

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./verify_release.sh <version>"
    exit 1
fi

echo "=== Verifying Release v${VERSION} ==="

# Download release files
FILE="aiplatform-v${VERSION}.tar.gz"

# Verify signature
echo "Verifying signature..."
gpg --verify ${FILE}.asc ${FILE} || {
    echo "❌ Signature verification failed"
    exit 1
}

# Verify checksum
echo "Verifying checksum..."
sha256sum -c ${FILE}.sha256 || {
    echo "❌ Checksum verification failed"
    exit 1
}

# Extract and verify contents
echo "Extracting release..."
tar -xzf ${FILE}

# Check SBOM
if [ -f "bom.xml" ]; then
    echo "✅ SBOM present"
    cyclonedx-cli validate --input-file bom.xml
fi

echo "✅ Release verified successfully"
```

---

## 5. Certificate Rotation

### 5.1 Rotation Schedule

- **Code signing keys**: Every 2 years
- **Certificates**: Every 1 year
- **TSA certificates**: Every 1 year
- **Keys used in CI/CD**: Every 90 days

### 5.2 Rotation Process

```bash
#!/bin/bash
# rotate_signing_key.sh

echo "=== Rotating Code Signing Key ==="

# 1. Generate new key
gpg --full-generate-key

# 2. Export new public key
gpg --list-keys
NEW_KEY_ID="<newly generated key id>"

# 3. Upload to keyservers
gpg --send-keys $NEW_KEY_ID

# 4. Update configuration
git config --global user.signingkey $NEW_KEY_ID

# 5. Create transition statement (signed by both keys)
# Sign release notes announcing new key

# 6. Archive old key securely
gpg --export-secret-keys releases@aiplatform.io | \
    gpg --encrypt -r "backup@aiplatform.io" > old-key.asc.gpg

# 7. Publish new key in repository
curl https://keybase.io/releases@aiplatform.io/pgp_keys.asc > keys/old.asc

echo "✅ Key rotation complete"
```

---

## 6. Security Best Practices

### 6.1 Secrets Management

```bash
# ❌ DON'T
git add signing_key.asc
export GPG_PASSPHRASE="password123"
echo "password" | gpg --batch --passphrase-fd 0

# ✅ DO
# Store in secure vault
vault write secret/signing-key @signing_key.asc

# Use secure key storage
gpg --pinentry-mode loopback --passphrase-file /dev/stdin

# Never log credentials
unset GPG_PASSPHRASE
```

### 6.2 CI/CD Integration

```yaml
# .github/workflows/release.yml
jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Import GPG key
        run: |
          echo "${{ secrets.GPG_PRIVATE_KEY }}" | gpg --import
      
      - name: Sign release
        run: |
          gpg --default-key "${{ secrets.GPG_KEY_ID }}" \
              --armor --sign --detach-sig \
              release.tar.gz
        env:
          GPG_TTY: $(tty)
```

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| Key not found | `gpg --recv-keys KEY_ID` |
| Signature verification fails | Verify key is trusted: `gpg --edit-key KEY_ID` |
| Wrong key used | Check `git config user.signingkey` |
| Passphrase issues | Configure GPG agent: `gpg-agent --daemon` |
| SBOM validation fails | Update tools: `pip install --upgrade cyclonedx-bom` |

---

## Conclusion

Proper code signing and SBOM generation ensure release integrity and supply chain security. Regular key rotation and verification processes maintain long-term security.

**Last Updated**: January 2024
