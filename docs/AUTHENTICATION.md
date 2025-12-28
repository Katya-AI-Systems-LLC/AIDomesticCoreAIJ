# Authentication Guide - AI Platform

## Overview

The AI Platform API uses Bearer token (JWT) authentication. All API requests must include a valid authentication token in the Authorization header.

## Getting Started

### 1. Create API Key

Navigate to your account settings and create an API key:

1. Go to https://dashboard.aiplatform.io/settings/api-keys
2. Click "Create New Key"
3. Give it a descriptive name (e.g., "Development", "Production")
4. Select scopes (see [Scopes](#scopes) section)
5. Set expiration (recommended: 90 days)
6. Copy the key (you won't see it again!)

### 2. Store API Key Securely

**Never commit API keys to version control!**

Use environment variables:

```bash
# .env (local development)
API_KEY=sk_prod_550e8400e29b41d4a716446655440000

# export in shell
export API_KEY="sk_prod_550e8400e29b41d4a716446655440000"
```

Load in your application:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
```

```javascript
require('dotenv').config();
const apiKey = process.env.API_KEY;
```

### 3. Include in Requests

Add the API key to every request header:

```bash
curl -H "Authorization: Bearer sk_prod_550e8400e29b41d4a716446655440000" \
     https://api.aiplatform.io/v1/health
```

## Token Types

### API Key Token

- **Format**: `sk_prod_*` or `sk_test_*`
- **Lifetime**: Until revoked or expiration date
- **Use case**: Server-to-server authentication
- **Scope**: Limited by configured scopes

```
Authorization: Bearer sk_prod_550e8400e29b41d4a716446655440000
```

### JWT Token

Generated from API key exchange. Used for long-lived sessions.

**Request JWT:**

```bash
curl -X POST https://api.aiplatform.io/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "sk_prod_550e8400e29b41d4a716446655440000"
  }'
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "ref_550e8400e29b41d4a716446655440000"
}
```

**Use JWT in requests:**

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
     https://api.aiplatform.io/v1/health
```

**Refresh token:**

```bash
curl -X POST https://api.aiplatform.io/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "ref_550e8400e29b41d4a716446655440000"
  }'
```

## Scopes

API keys can be limited to specific scopes:

### Available Scopes

| Scope | Permission |
|-------|-----------|
| `read:health` | Read health/status endpoints |
| `read:optimization` | Read optimization jobs |
| `write:optimization` | Submit optimization jobs |
| `read:vision` | Read vision analysis results |
| `write:vision` | Submit vision analysis jobs |
| `read:federated` | Read training job status |
| `write:federated` | Submit training jobs |
| `read:inference` | Read inference results |
| `write:inference` | Run inference |
| `read:models` | List/read models |
| `write:models` | Deploy/fine-tune models |
| `read:projects` | Read projects |
| `write:projects` | Create/update/delete projects |
| `admin` | Admin access (all scopes) |

### Scope Examples

**Minimal scope (read-only):**

```bash
curl -X POST https://api.aiplatform.io/v1/keys \
  -H "Authorization: Bearer admin-key" \
  -d '{
    "name": "Analytics Reader",
    "scopes": ["read:optimization", "read:vision", "read:projects"]
  }'
```

**Full access:**

```bash
curl -X POST https://api.aiplatform.io/v1/keys \
  -H "Authorization: Bearer admin-key" \
  -d '{
    "name": "Production Key",
    "scopes": ["admin"]
  }'
```

## Authentication Errors

### 401 Unauthorized

Missing or invalid authentication:

```json
{
  "error": "UNAUTHORIZED",
  "message": "Missing or invalid authentication token"
}
```

**Solutions:**
- Check API key is included in Authorization header
- Verify header format: `Authorization: Bearer <token>`
- Ensure API key hasn't expired
- Check key hasn't been revoked

### 403 Forbidden

Valid authentication but insufficient permissions:

```json
{
  "error": "FORBIDDEN",
  "message": "Your API key lacks permission for this operation"
}
```

**Solutions:**
- Create a new API key with required scopes
- Ask admin to grant additional permissions
- Check scopes for your API key

## Best Practices

### 1. Key Rotation

Rotate API keys regularly:

```bash
# Create new key
NEW_KEY=$(curl -X POST https://api.aiplatform.io/v1/keys \
  -H "Authorization: Bearer old-key" \
  -d '{"name": "Production Key v2"}' \
  | jq -r '.key')

# Update environment
export API_KEY=$NEW_KEY

# Revoke old key after migration
curl -X DELETE https://api.aiplatform.io/v1/keys/key-id \
  -H "Authorization: Bearer admin-key"
```

### 2. Scoped Keys

Use minimal scopes for different services:

```bash
# Frontend: read-only
FRONTEND_KEY=$(curl -X POST https://api.apiplatform.io/v1/keys \
  -d '{"name": "Frontend", "scopes": ["read:optimization", "read:vision"]}')

# Backend: full access
BACKEND_KEY=$(curl -X POST https://api.aiplatform.io/v1/keys \
  -d '{"name": "Backend", "scopes": ["admin"]}')
```

### 3. Key Expiration

Set short expiration times:

```bash
curl -X POST https://api.aiplatform.io/v1/keys \
  -d '{
    "name": "Temporary Key",
    "expires_in": 86400  # 24 hours
  }'
```

### 4. Monitoring & Auditing

Monitor API key usage:

```bash
# Get key usage
curl https://api.aiplatform.io/v1/keys/key-id/usage \
  -H "Authorization: Bearer admin-key"
```

Response:

```json
{
  "key_id": "key_550e8400e29b41d4",
  "name": "Production Key",
  "created_at": "2024-01-01T00:00:00Z",
  "last_used_at": "2024-01-15T10:30:00Z",
  "requests_count": 125000,
  "requests_this_month": 25000,
  "data_transferred_gb": 250
}
```

### 5. Separate Keys Per Environment

Use different keys for development, staging, and production:

```python
# config.py
import os

ENVIRONMENTS = {
    'development': os.getenv('DEV_API_KEY'),
    'staging': os.getenv('STAGING_API_KEY'),
    'production': os.getenv('PROD_API_KEY'),
}

API_KEY = ENVIRONMENTS[os.getenv('ENV', 'development')]
```

## Multi-Factor Authentication

For admin accounts, enable MFA:

1. Go to https://dashboard.aiplatform.io/settings/security
2. Click "Enable Multi-Factor Authentication"
3. Scan QR code with authenticator app (Google Authenticator, Authy, etc.)
4. Verify with generated code
5. Save backup codes in safe location

## OAuth 2.0 (For Applications)

If building applications for other users, use OAuth 2.0:

### Authorization Flow

```
1. Redirect user to: https://dashboard.aiplatform.io/oauth/authorize?
   client_id=app_550e8400e29b41d4&
   redirect_uri=https://example.com/callback&
   scope=read:optimization,read:vision&
   state=random_state_value

2. User authorizes application

3. Browser redirects to: https://example.com/callback?
   code=auth_550e8400e29b41d4&
   state=random_state_value

4. Exchange code for token:
   POST https://api.aiplatform.io/v1/oauth/token
   {
     "grant_type": "authorization_code",
     "code": "auth_550e8400e29b41d4",
     "client_id": "app_550e8400e29b41d4",
     "client_secret": "secret_value",
     "redirect_uri": "https://example.com/callback"
   }

5. Receive access token:
   {
     "access_token": "token_...",
     "token_type": "Bearer",
     "expires_in": 3600,
     "refresh_token": "ref_..."
   }
```

### Implementation

**Python (Flask)**

```python
from authlib.integrations.flask_client import OAuth

oauth = OAuth()
oauth.register(
    name='aiplatform',
    client_id='app_550e8400e29b41d4',
    client_secret='secret_value',
    server_metadata_url='https://api.aiplatform.io/.well-known/openid-configuration',
    client_kwargs={'scope': 'read:optimization read:vision'}
)

@app.route('/login')
def login():
    redirect_uri = url_for('auth_callback', _external=True)
    return oauth.aiplatform.authorize_redirect(redirect_uri)

@app.route('/auth/callback')
def auth_callback():
    token = oauth.aiplatform.authorize_access_token()
    session['token'] = token
    return redirect('/')
```

**JavaScript (Node.js)**

```javascript
const passport = require('passport');
const OAuthStrategy = require('passport-oauth2');

passport.use(new OAuthStrategy({
    authorizationURL: 'https://dashboard.aiplatform.io/oauth/authorize',
    tokenURL: 'https://api.aiplatform.io/v1/oauth/token',
    clientID: 'app_550e8400e29b41d4',
    clientSecret: 'secret_value',
    callbackURL: 'http://localhost:3000/auth/callback'
}, (accessToken, refreshToken, profile, done) => {
    // Store tokens
    done(null, { accessToken, refreshToken });
}));
```

## Troubleshooting

### "Invalid Token"

- Token has expired → refresh or request new token
- Token was revoked → create new API key
- Typo in token → copy from dashboard again

### "Rate Limited"

- Too many requests → wait before retrying
- Check rate limit headers: `X-RateLimit-Remaining`
- Upgrade to higher tier for more quota

### "CORS Error"

API calls from browser to different origin fail:

**Solution:** Use backend proxy or CORS headers.

Backend proxy (recommended):

```python
@app.route('/api/optimize', methods=['POST'])
def optimize_proxy():
    return requests.post(
        'https://api.aiplatform.io/v1/optimize',
        json=request.json,
        headers={'Authorization': f'Bearer {os.getenv("API_KEY")}'}
    ).json()
```

## Support

- **Documentation**: https://docs.aiplatform.io
- **API Status**: https://status.aiplatform.io
- **Help**: https://support.aiplatform.io
- **Email**: support@aiplatform.io

## Conclusion

Proper authentication is critical for secure API access. For more information, see the [API Guide](API_GUIDE.md).
