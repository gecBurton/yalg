# LLM Gateway

OpenAI-compatible API gateway for multiple AI providers with authentication and metrics.

> [!NOTE]  
> This is a just-for-fun project - most likely it wont work for you!

## Features
- **OpenAI API compatible** - Drop-in replacement
- **Multi-provider** - Azure OpenAI, AWS Bedrock (Claude), Google Gemini  
- **Authentication** - OIDC/SSO integration
- **Streaming** - Real-time responses
- **Metrics** - Usage tracking and analytics

## Quick Start

### 1. Setup
```bash
# Clone and build
git clone <repo>
cd llm-freeway
go build

# Configure providers in .env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
OIDC_CLIENT_ID=your_client_id
OIDC_CLIENT_SECRET=your_secret
```

### 2. Run
```bash
./llm-freeway
# Web UI: http://localhost:3000
```

## API Usage

All models require explicit provider prefixes:
- **Azure OpenAI**: `azure-openai/gpt-4o-mini`
- **AWS Bedrock**: `aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:0`  
- **Google AI**: `google-ai/gemini-1.5-pro`

### cURL Example
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "azure-openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Endpoints
- `POST /v1/chat/completions` - Chat completions (OpenAI compatible)
- `GET /v1/models` - List available models  
- `GET /` - Web UI for testing
- `GET /metrics` - Usage statistics
- `GET /health` - Health check

---

*Note: This README has been simplified. The service includes many more features including detailed metrics, authentication, rate limiting, and cost tracking.*