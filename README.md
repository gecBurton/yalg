# 🚀 LLM Gateway

A modern, OpenAI-compatible API gateway for multiple AI providers with enterprise-grade authentication and comprehensive analytics.

## ✨ Features

- **🔌 OpenAI API Compatible** - Drop-in replacement for existing applications
- **🌐 Multi-Provider Support** - Azure OpenAI, AWS Bedrock, Google Gemini, Anthropic Claude
- **🔐 Enterprise Authentication** - OIDC/SSO integration with secure token management
- **📊 Real-time Analytics** - Detailed usage metrics and monitoring
- **⚡ Streaming Support** - Real-time response streaming for all providers
- **🛡️ Rate Limiting** - Built-in rate limiting with per-model configuration
- **🎯 Custom Model Routes** - Flexible model naming and routing
- **📱 Modern Web UI** - Clean, responsive interface for testing and monitoring

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-gateway
cd llm-gateway

# Build the application
go build -o llm-gateway

# Make it executable
chmod +x llm-gateway
```

### 2. Configuration

Create a `.env` file with your API keys:

```env
# Provider API Keys
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key

# Authentication (OIDC/SSO)
OIDC_CLIENT_ID=your_client_id
OIDC_CLIENT_SECRET=your_client_secret
OIDC_REDIRECT_URI=http://localhost:3000/callback/
OIDC_BASE_URL=http://localhost:3000

# Optional: Database (defaults to SQLite)
DB_TYPE=sqlite
DB_PATH=./gateway.db

# Optional: Server Configuration
PORT=3000
```

### 3. Run

```bash
# Start the gateway
./llm-gateway

# Access the web interface
open http://localhost:3000
```

## 🔧 Model Configuration

Configure available models in `models.yaml`:

```yaml
models:
  # GPT Models
  gpt-4o:
    provider: azure-openai
    model_name: "gpt-4o"
    display_name: "GPT-4o (Azure)"
    description: "OpenAI's most capable model via Azure"
    max_tokens: 128000
    supports_streaming: true
    enabled: true
    rate_limit: 100

  openai-gpt-4o:
    provider: openai
    model_name: "gpt-4o"
    display_name: "GPT-4o (OpenAI)"
    description: "OpenAI's most capable model via direct API"
    max_tokens: 128000
    supports_streaming: true
    enabled: true
    rate_limit: 80

  # Claude Models
  claude-3-5-sonnet:
    provider: anthropic
    model_name: "claude-3-5-sonnet-20241022"
    display_name: "Claude 3.5 Sonnet"
    description: "Anthropic's most capable model"
    max_tokens: 200000
    supports_streaming: true
    enabled: true
    rate_limit: 50

  claude-bedrock:
    provider: aws-bedrock
    model_name: "anthropic.claude-3-sonnet-20240229-v1:0"
    display_name: "Claude 3 Sonnet (AWS)"
    description: "Claude via AWS Bedrock"
    max_tokens: 200000
    supports_streaming: true
    enabled: true
    rate_limit: 60

  # Gemini Models
  gemini-1-5-pro:
    provider: google-ai
    model_name: "gemini-1.5-pro"
    display_name: "Gemini 1.5 Pro"
    description: "Google's most capable model"
    max_tokens: 1000000
    supports_streaming: true
    enabled: true
    rate_limit: 30
```

## 🛠️ API Usage

### Authentication

All API requests require authentication via Bearer token:

```bash
# Get your token from the web interface, then:
export LLM_GATEWAY_TOKEN="your-token-here"
```

### Chat Completions

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_GATEWAY_TOKEN" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": false
  }'
```

### Streaming

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_GATEWAY_TOKEN" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [
      {"role": "user", "content": "Write a short poem about technology"}
    ],
    "stream": true
  }'
```

### List Models

```bash
curl -H "Authorization: Bearer $LLM_GATEWAY_TOKEN" \
  http://localhost:3000/v1/models
```

## 📊 Monitoring & Analytics

### Web Interface

- **Dashboard**: `http://localhost:3000`
- **Metrics**: `http://localhost:3000/metrics`
- **Health Check**: `http://localhost:3000/health`

### API Endpoints

- `GET /metrics?format=json` - Usage statistics
- `GET /v1/models` - Available models
- `GET /health` - Service health status

### Example Metrics Response

```json
{
  "usage": {
    "total_requests": 1250,
    "successful_requests": 1200,
    "failed_requests": 50,
    "average_latency": 1.2
  },
  "rate_limits": {
    "gpt-4o": {
      "limit": 100,
      "remaining": 95,
      "reset": "2024-01-01T10:00:00Z"
    }
  }
}
```

## 🔒 Security Features

- **OIDC/SSO Integration** - Enterprise-grade authentication
- **Token-based Authorization** - Secure API access
- **Rate Limiting** - Prevent abuse and manage costs
- **Request Logging** - Comprehensive audit trail
- **API Key Sanitization** - No sensitive data in logs

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   LLM Gateway   │───▶│   AI Providers  │
│                 │    │                 │    │                 │
│ • Web UI        │    │ • Auth          │    │ • OpenAI        │
│ • API Client    │    │ • Rate Limiting │    │ • Anthropic     │
│ • curl/SDK      │    │ • Metrics       │    │ • Google        │
│                 │    │ • Load Balance  │    │ • Azure         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install dependencies
go mod download

# Run tests
go test ./...

# Run with hot reload
go run main.go

# Build for production
go build -ldflags="-s -w" -o llm-gateway
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by enterprise API gateway patterns
- Built with Go for performance and reliability
- Uses modern web standards for the UI
- Supports all major AI providers

## 📞 Support

- 📖 **Documentation**: [Wiki](https://github.com/yourusername/llm-gateway/wiki)
- 💬 **Issues**: [GitHub Issues](https://github.com/yourusername/llm-gateway/issues)
- 🚀 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/llm-gateway/discussions)

---

**Made with ❤️ for the AI community**