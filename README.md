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
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
OIDC_CLIENT_ID=your_client_id
OIDC_CLIENT_SECRET=your_secret
```

### 2. Run
```bash
./llm-freeway
# Web UI: http://localhost:3000
```

## Configuration

### Model Configuration (models.yaml)

The gateway uses a `models.yaml` file to configure available models with custom route names:

```yaml
models:
  # Custom route names that map to actual models
  gpt-4o:
    provider: azure-openai
    model_name: "gpt-4o"
    display_name: "GPT-4o"
    description: "OpenAI's most capable model"
    max_tokens: 128000
    supports_streaming: true
    enabled: true
    rate_limit: 100

  my-favorite-claude:
    provider: anthropic
    model_name: "claude-3-5-sonnet-20241022"
    display_name: "My Favorite Claude"
    description: "Anthropic's Claude via direct API"
    max_tokens: 200000
    supports_streaming: true
    enabled: true
    rate_limit: 30

  claude-bedrock:
    provider: aws-bedrock
    model_name: "anthropic.claude-3-sonnet-20240229-v1:0"
    display_name: "Claude via Bedrock"
    description: "Anthropic's Claude via AWS Bedrock"
    max_tokens: 200000
    supports_streaming: true
    enabled: true
    rate_limit: 50

  openai-gpt-4o:
    provider: openai
    model_name: "gpt-4o"
    display_name: "GPT-4o (OpenAI)"
    description: "Most capable GPT-4 model via direct OpenAI API"
    max_tokens: 128000
    supports_streaming: true
    enabled: true
    rate_limit: 80
```

**Key Features:**
- **Custom Route Names**: Use any name you want (e.g., `my-favorite-claude`)
- **Multiple Instances**: Configure the same model multiple times with different settings
- **Model-specific Rate Limits**: Each model can have its own rate limit
- **Flexible Configuration**: Easy to add, remove, or modify models

## API Usage

Use the custom route names defined in your `models.yaml`:

### cURL Example
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "openai-gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Endpoints
- `POST /v1/chat/completions` - Chat completions (OpenAI compatible)
- `GET /v1/models` - List available models  
- `GET /` - Web UI for testing
- `GET /metrics` - Usage statistics
- `GET /health` - Health check

