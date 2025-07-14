# Yet Another LLM Gateway

OpenAI-compatible API gateway that provides unified access to multiple AI providers.

> [!NOTE]  
> This is a just-for-fun project - most likely it wont work for you!


## ✨ Features

### Core Features
- **🔄 OpenAI API Compatibility**: Drop-in replacement for OpenAI API clients
- **🌐 Multi-Provider Support**: Azure OpenAI, AWS Bedrock (Claude), Google Gemini
- **⚡ Streaming Support**: Real-time streaming responses for all providers


## 🚀 New API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `GET /v1/models` - List available models from all enabled providers
- `GET /metrics` - Comprehensive usage metrics and statistics
- `GET /health` - Enhanced health check with provider status
- `GET /` - Web testing interface

## 🎯 Supported Models

### Azure OpenAI Models
- GPT-4 (`gpt-4`) - Most capable for complex tasks
- GPT-4 Turbo (`gpt-4-turbo`) - Latest with improved speed
- GPT-4o (`gpt-4o`) - Omni-modal capabilities
- GPT-4o Mini (`gpt-4o-mini`) - Smaller, faster version

### Anthropic Claude Models (via AWS Bedrock)
- Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`) - Fast and efficient
- Claude 3 Sonnet (`anthropic.claude-3-sonnet-20240229-v1:0`) - Balanced performance
- Claude 3 Opus (`anthropic.claude-3-opus-20240229-v1:0`) - Most capable reasoning

### Google Gemini Models
- Gemini 1.5 Flash (`gemini-1.5-flash`) - Fast and efficient, 1M tokens
- Gemini 2.0 Flash Experimental (`gemini-2.0-flash-exp`) - Next-generation model
- Gemini 1.0 Pro (`gemini-1.0-pro`) - Original Gemini Pro

## 🛠️ Installation

### Prerequisites
- Go 1.21 or later
- Access to at least one AI provider

### Build from Source
```bash
git clone <repository-url>
cd llm-freeway
go build -o llm-freeway
```

## ⚙️ Configuration

### Core Provider Configuration

#### Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-01"  # Optional
```

#### AWS Bedrock (Anthropic Claude)
```bash
# Use AWS credentials (recommended: aws-vault or IAM roles)
aws configure
# OR set directly
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
```

#### Google Gemini
```bash
export GEMINI_API_KEY="your_gemini_api_key"
```
Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### Advanced Configuration

#### Server Settings
```bash
export PORT="8000"                           # Server port (default: 8000)
export SERVER_READ_TIMEOUT="30s"            # Request read timeout
export SERVER_WRITE_TIMEOUT="30s"           # Response write timeout
export MAX_REQUEST_SIZE="10485760"          # Max request size (10MB)
```

#### Metrics & Monitoring
```bash
export METRICS_ENABLED="true"               # Enable metrics collection
export METRICS_RETENTION_DAYS="30"          # How long to keep metrics
export METRICS_UPDATE_INTERVAL="1m"         # Metrics update frequency
export METRICS_EXPORT_PATH="/tmp/metrics.json" # Export file path
```

#### Rate Limiting (per provider)
```bash
export AZURE_OPENAI_RATE_LIMIT="60"         # Requests per minute
export AWS_BEDROCK_RATE_LIMIT="30"          # Requests per minute
export GEMINI_RATE_LIMIT="60"               # Requests per minute
```

#### Cost Tracking (per 1K tokens)
```bash
export AZURE_OPENAI_COST_PER_TOKEN="0.002"  # $0.002 per 1K tokens
export AWS_BEDROCK_COST_PER_TOKEN="0.003"   # $0.003 per 1K tokens
export GEMINI_COST_PER_TOKEN="0.001"        # $0.001 per 1K tokens
```

#### Logging (Security-First Defaults)
```bash
export LOG_LEVEL="info"                      # debug, info, warn, error
export LOG_FORMAT="text"                     # text or json
export LOG_REQUESTS="false"                  # SECURITY: Never log request content (default: false)
export LOG_RESPONSES="false"                 # SECURITY: Never log response content (default: false)
export LOG_TOKEN_USAGE="false"               # Log only token counts, not content (default: false)
```

## 🚀 Usage

### Start the Server
```bash
./llm-freeway
```

### Access the Web UI
Open `http://localhost:8000/ui` for a professional testing interface featuring:
- **Model Selection**: All providers with explicit routing options (e.g., `anthropic/claude-3-sonnet`)
- **Real-time Streaming**: Live response streaming with proper error handling
- **Accessibility**: WCAG 2.1 AA compliant with keyboard navigation and screen reader support
- **Beta Phase Banner**: GDS service identification

### Check Server Status
```bash
curl http://localhost:8000/health
```

Response includes provider status, metrics summary, and configuration details.

### View Metrics
```bash
curl http://localhost:8000/metrics?format=pretty
```

Returns comprehensive usage statistics, cost tracking, and performance metrics.

### List Available Models
```bash
curl http://localhost:8000/v1/models
```

Returns OpenAI-compatible model list with provider information and capabilities.

## 📡 API Usage

### Basic Request (Explicit Routing Required)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google-ai/gemini-1.5-flash",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": true
  }'
```

### Required Explicit Routing
All requests must use provider prefixes:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aws-bedrock/claude-3-5-sonnet-20240620-v1:0",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ]
  }'
```

**Invalid** (will be rejected):
```bash
# This will fail - no automatic detection
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model": "gpt-4", "messages": [...]}'
```

### Using with OpenAI Libraries

#### Python
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used, but required by library
)

# All models require explicit provider prefix
response = client.chat.completions.create(
    model="google/gemini-1.5-pro",  # Explicit routing required
    messages=[
        {"role": "user", "content": "Write a Python function to calculate factorial"}
    ]
)

print(response.choices[0].message.content)
```

#### Node.js
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
    baseURL: 'http://localhost:8000/v1',
    apiKey: 'dummy'
});

const response = await openai.chat.completions.create({
    model: 'aws-bedrock/claude-3-5-sonnet-20240620-v1:0', // Explicit prefix required
    messages: [{ role: 'user', content: 'Explain machine learning' }],
    stream: true
});

for await (const chunk of response) {
    console.log(chunk.choices[0]?.delta?.content || '');
}
```

## 🎯 Model Routing (Explicit Only)

### Required Provider Prefixes
All model requests **must** include an explicit provider prefix. No automatic detection is performed to avoid routing errors.

**Required Format**: `provider/model-name`

- **Azure OpenAI**: `azure-openai/model-name`
  - `azure-openai/gpt-4`
  - `azure-openai/gpt-4-turbo` 
  - `azure-openai/gpt-35-turbo`
  
- **AWS Bedrock**: `aws-bedrock/model-name`
  - `aws-bedrock/claude-3-5-sonnet-20240620-v1:0`
  - `aws-bedrock/claude-3-opus-20240229-v1:0`
  - `aws-bedrock/claude-3-haiku-20240307-v1:0`
  
- **Google AI**: `google/model-name`
  - `google-ai/gemini-1.5-pro`
  - `google-ai/gemini-1.5-flash`
  - `google-ai/gemini-2.0-flash-exp`


## 📊 Metrics & Monitoring

### Request Metrics
- Request count, success rate, error rate
- Average latency and token usage
- Cost tracking per provider and model
- Hourly usage statistics

### Provider Analytics
- Per-provider performance metrics
- Rate limiting status and utilization
- Error classification and trending
- Cost optimization insights

### Export & Integration
- JSON export for external monitoring
- Structured logging for log aggregation
- Health check endpoints for load balancers
- Metrics API for dashboard integration


## 🧪 Development

### Running Tests
```bash
make test
```

### Adding New Providers
1. Create adapter in `internal/adapter/`
2. Add provider type to `internal/config/config.go`
3. Update routing logic in `internal/router/router.go`
4. Add error handling patterns in `internal/errors/errors.go`
5. Update UI model options

### Local Development
```bash
# Start with debug logging
LOG_LEVEL=debug ./llm-freeway

# Enable all metrics
METRICS_ENABLED=true LOG_REQUESTS=true ./llm-freeway

# Test with minimal providers
AZURE_OPENAI_API_KEY=test AZURE_OPENAI_ENDPOINT=https://test.com ./llm-freeway
```

# yalg
