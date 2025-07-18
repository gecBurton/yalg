package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"llm-freeway/internal/adapter"
	"llm-freeway/internal/auth"
	"llm-freeway/internal/config"
	"llm-freeway/internal/database"
	"llm-freeway/internal/errors"
	"llm-freeway/internal/router"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/openai/openai-go"
	"github.com/pkoukk/tiktoken-go"
	"google.golang.org/genai"
)

// TokenUsage represents token usage information
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatChoice represents a chat completion choice
type ChatChoice struct {
	Index        int               `json:"index"`
	Message      map[string]string `json:"message"`
	FinishReason string            `json:"finish_reason"`
	Delta        map[string]any    `json:"delta,omitempty"`
}

// ChatResponse represents a chat completion response
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   TokenUsage   `json:"usage"`
}

// ErrorResponse represents an API error response
type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// EmbeddingRequest represents an embedding request
type EmbeddingRequest struct {
	Input          interface{} `json:"input"`          // Can be string or []string
	Model          string      `json:"model"`
	EncodingFormat string      `json:"encoding_format,omitempty"`
	Dimensions     int         `json:"dimensions,omitempty"`
	User           string      `json:"user,omitempty"`
}

// EmbeddingData represents a single embedding result
type EmbeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// EmbeddingUsage represents token usage for embeddings
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// EmbeddingResponse represents an embedding response
type EmbeddingResponse struct {
	Object string           `json:"object"`
	Data   []EmbeddingData  `json:"data"`
	Model  string           `json:"model"`
	Usage  EmbeddingUsage   `json:"usage"`
}

// ServerConfig holds configuration for the server
type ServerConfig struct {
	Config             *config.Config
	BedrockClient      *bedrockruntime.Client
	GeminiClient       *genai.Client
	AzureOpenAIClient  *openai.Client
	DirectOpenAIClient *openai.Client
	AnthropicAPIKey    string
	AWSConfig          *aws.Config
	Database           *database.DB
	ErrorHandler       *errors.ErrorHandler
	Router             *router.Router
}

// Server provides LLM gateway functionality with metrics, rate limiting, etc.
type Server struct {
	claudeAdapter      *adapter.ClaudeAdapter
	geminiAdapter      *adapter.GeminiAdapter
	azureOpenAIClient  *openai.Client
	directOpenAIClient *openai.Client
	bedrockClient      *bedrockruntime.Client
	config             *config.Config
	database           *database.DB
	errorHandler       *errors.ErrorHandler
	router             *router.Router
}

// NewServer creates a new server instance with all components
func NewServer(cfg ServerConfig) *Server {
	var geminiAdapter *adapter.GeminiAdapter
	if cfg.GeminiClient != nil {
		geminiAdapter = adapter.NewGeminiAdapter(cfg.GeminiClient)
	}

	// Create unified Claude adapter for both direct Anthropic and Bedrock
	claudeAdapter := adapter.NewClaudeAdapter(adapter.ClaudeAdapterConfig{
		AnthropicAPIKey: cfg.AnthropicAPIKey,
		AWSConfig:       cfg.AWSConfig,
	})

	return &Server{
		claudeAdapter:      claudeAdapter,
		geminiAdapter:      geminiAdapter,
		azureOpenAIClient:  cfg.AzureOpenAIClient,
		directOpenAIClient: cfg.DirectOpenAIClient,
		bedrockClient:      cfg.BedrockClient,
		config:             cfg.Config,
		database:           cfg.Database,
		errorHandler:       cfg.ErrorHandler,
		router:             cfg.Router,
	}
}

// ConvertToOpenAIMessages converts adapter Messages to OpenAI SDK format
func (s *Server) ConvertToOpenAIMessages(messages []adapter.Message) []openai.ChatCompletionMessageParamUnion {
	result := make([]openai.ChatCompletionMessageParamUnion, len(messages))
	for i, msg := range messages {
		contentStr := s.getContentAsString(msg.Content)
		switch msg.Role {
		case "assistant":
			result[i] = openai.AssistantMessage(contentStr)
		case "system":
			result[i] = openai.SystemMessage(contentStr)
		default:
			result[i] = openai.UserMessage(contentStr)
		}
	}
	return result
}

// getContentAsString converts interface{} content to string
func (s *Server) getContentAsString(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []adapter.ContentPart:
		// Extract text from content parts
		var textParts []string
		for _, part := range c {
			if part.Type == "text" {
				textParts = append(textParts, part.Text)
			}
		}
		return strings.Join(textParts, " ")
	default:
		return fmt.Sprintf("%v", content)
	}
}

// HealthHandler handles health check requests
func (s *Server) HealthHandler(w http.ResponseWriter, r *http.Request) {
	health := map[string]any{
		"status":    "healthy",
		"service":   "llm-freeway",
		"version":   "1.0.0",
		"timestamp": time.Now(),
		"providers": map[string]any{
			"azure_openai": s.config.IsProviderEnabled(config.ProviderAzureOpenAI),
			"openai":       s.config.IsProviderEnabled(config.ProviderOpenAI),
			"aws_bedrock":  s.config.IsProviderEnabled(config.ProviderAWSBedrock),
			"google_ai":    s.config.IsProviderEnabled(config.ProviderGoogleAI),
			"anthropic":    s.config.IsProviderEnabled(config.ProviderAnthropic),
		},
		"metrics_enabled": s.config.Metrics.Enabled,
	}

	// Add usage stats if metrics are enabled
	if s.config.Metrics.Enabled {
		usage := s.database.GetUsageStats()
		health["usage"] = map[string]any{
			"total_requests":      usage.TotalRequests,
			"successful_requests": usage.SuccessfulRequests,
			"failed_requests":     usage.FailedRequests,
			"average_latency":     usage.AverageLatency,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(health)
}

// UIHandler serves the UI HTML file
func (s *Server) UIHandler(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "index.html")
}

// ChatCompletionHandler handles chat completion requests
func (s *Server) ChatCompletionHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	requestID := fmt.Sprintf("req_%d", time.Now().UnixNano())

	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("Request [%s]: %s %s from %s", requestID, r.Method, r.URL.Path, r.RemoteAddr)

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	var req adapter.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		http.Error(w, "Model field is required", http.StatusBadRequest)
		return
	}

	// Validate model
	if err := s.router.ValidateModel(req.Model); err != nil {
		providerErr := s.errorHandler.CreateValidationError(
			config.ProviderType("unknown"),
			fmt.Sprintf("Invalid model: %s", err.Error()),
			map[string]any{"model": req.Model},
		)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, req.Stream)
		return
	}

	// Detect provider
	provider, err := s.router.DetectProvider(req.Model)
	if err != nil {
		providerErr := s.errorHandler.CreateValidationError("", err.Error(), nil)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, req.Stream)
		return
	}

	// Check rate limiting
	if allowed, retryAfter := s.database.CheckRate(provider); !allowed {
		providerErr := s.errorHandler.CreateRateLimitError(provider, retryAfter)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, req.Stream)
		return
	}

	// Get the actual model name for API calls
	actualModelName, err := s.router.GetModelName(req.Model)
	if err != nil {
		providerErr := s.errorHandler.CreateValidationError(
			provider,
			fmt.Sprintf("Failed to get model name: %s", err.Error()),
			map[string]any{"route": req.Model},
		)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, req.Stream)
		return
	}

	// Create a copy of the request with actual model name for API calls
	apiReq := req
	apiReq.Model = actualModelName

	// Get user information from context (if authenticated)
	var userID string
	if user, ok := auth.GetUserFromContext(r.Context()); ok {
		userID = user.ID
	}

	// Record request start
	metric := database.RequestMetrics{
		ID:        requestID,
		Timestamp: startTime,
		Provider:  provider,
		Model:     req.Model, // Keep original model name for metrics
		StartTime: startTime,
		Streaming: req.Stream,
		UserAgent: r.UserAgent(),
		ClientIP:  r.RemoteAddr,
	}

	// Route based on provider with actual model name
	var responseErr error
	switch provider {
	case config.ProviderAWSBedrock:
		// Claude adapter uses actual model name for AWS Bedrock
		responseErr = s.handleProviderRequest(apiReq, w, &metric, func(req adapter.ChatRequest, w http.ResponseWriter, metric *database.RequestMetrics) error {
			return s.handleClaude(req, w, metric, config.ProviderAWSBedrock)
		})
	case config.ProviderAnthropic:
		// Claude adapter uses actual model name for direct Anthropic
		responseErr = s.handleProviderRequest(apiReq, w, &metric, func(req adapter.ChatRequest, w http.ResponseWriter, metric *database.RequestMetrics) error {
			return s.handleClaude(req, w, metric, config.ProviderAnthropic)
		})
	case config.ProviderGoogleAI:
		responseErr = s.handleProviderRequest(apiReq, w, &metric, s.handleGemini)
	case config.ProviderAzureOpenAI:
		responseErr = s.handleProviderRequest(apiReq, w, &metric, s.handleOpenAI)
	case config.ProviderOpenAI:
		responseErr = s.handleProviderRequest(apiReq, w, &metric, s.handleOpenAI)
	default:
		providerErr := s.errorHandler.CreateValidationError(
			provider,
			fmt.Sprintf("Provider %s is not supported or enabled", provider),
			map[string]any{"provider": string(provider)},
		)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, req.Stream)
		responseErr = providerErr
	}

	// Complete metrics
	metric.EndTime = time.Now()
	metric.Duration = metric.EndTime.Sub(metric.StartTime)
	metric.Success = (responseErr == nil)

	if responseErr != nil {
		if providerErr, ok := responseErr.(*errors.ProviderError); ok {
			metric.ErrorType = string(providerErr.Type)
			metric.ErrorMessage = providerErr.Message
			metric.StatusCode = providerErr.HTTPStatus
		} else {
			metric.ErrorType = "unknown"
			metric.ErrorMessage = responseErr.Error()
			metric.StatusCode = 500
		}
	} else {
		metric.StatusCode = 200
	}

	// Record metrics
	s.database.RecordRequest(metric, userID)
}

// handleProviderRequest is a unified handler for all providers
func (s *Server) handleProviderRequest(
	req adapter.ChatRequest,
	w http.ResponseWriter,
	metric *database.RequestMetrics,
	providerHandler func(adapter.ChatRequest, http.ResponseWriter, *database.RequestMetrics) error,
) error {
	return providerHandler(req, w, metric)
}

// sendErrorResponse sends an error response in OpenAI-compatible format
func (s *Server) sendErrorResponse(w http.ResponseWriter, err *errors.ProviderError, requestID, model string, streaming bool) {
	if streaming {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		errorChunk := err.ToOpenAIStreamError(requestID, model)
		data, _ := json.Marshal(errorChunk)
		w.Write([]byte("data: " + string(data) + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))

		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(err.HTTPStatus)

		errorResp := err.ToOpenAIError(requestID)
		json.NewEncoder(w).Encode(errorResp)
	}

	// Log the error
	s.errorHandler.LogError(err, requestID, map[string]any{
		"model":     model,
		"streaming": streaming,
	})
}

// handleClaude processes requests for Anthropic models (both Bedrock and direct)
func (s *Server) handleClaude(req adapter.ChatRequest, w http.ResponseWriter, metric *database.RequestMetrics, provider config.ProviderType) error {
	log.Printf("Routing to Claude model: %s via provider: %s", req.Model, provider)

	if req.Stream {
		return s.handleStreamingRequest(req, w, metric, func() error {
			return s.claudeAdapter.HandleStreamingRequestWithProvider(req, w, string(provider))
		})
	}

	// Handle non-streaming
	openaiResp, err := s.claudeAdapter.HandleRequestWithProvider(req, string(provider))
	if err != nil {
		return s.handleNonStreamingError(err, w, metric, req.Model, provider)
	}

	s.extractTokenUsage(openaiResp, metric)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openaiResp)
	return nil
}

// handleGemini processes requests for Gemini models
func (s *Server) handleGemini(req adapter.ChatRequest, w http.ResponseWriter, metric *database.RequestMetrics) error {
	if s.geminiAdapter == nil {
		err := fmt.Errorf("Gemini client not available - missing API key configuration")
		return s.errorHandler.HandleError(err, config.ProviderGoogleAI, map[string]any{"model": req.Model})
	}

	log.Printf("Routing to Gemini model: %s", req.Model)

	if req.Stream {
		return s.handleStreamingRequest(req, w, metric, func() error {
			return s.geminiAdapter.HandleStreamingRequest(req, w)
		})
	}

	// Handle non-streaming
	openaiResp, err := s.geminiAdapter.HandleRequest(req)
	if err != nil {
		return s.handleNonStreamingError(err, w, metric, req.Model, config.ProviderGoogleAI)
	}

	s.extractTokenUsage(openaiResp, metric)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openaiResp)
	return nil
}

// getOpenAIClient returns the appropriate OpenAI client based on provider
func (s *Server) getOpenAIClient(provider config.ProviderType) (*openai.Client, error) {
	switch provider {
	case config.ProviderAzureOpenAI:
		if s.azureOpenAIClient == nil {
			return nil, fmt.Errorf("Azure OpenAI client not available - missing API key or endpoint configuration")
		}
		return s.azureOpenAIClient, nil
	case config.ProviderOpenAI:
		if s.directOpenAIClient == nil {
			return nil, fmt.Errorf("OpenAI client not available - missing API key configuration")
		}
		return s.directOpenAIClient, nil
	default:
		return nil, fmt.Errorf("unsupported OpenAI provider: %s", provider)
	}
}

// handleOpenAI processes requests for OpenAI models (both Azure and direct)
func (s *Server) handleOpenAI(req adapter.ChatRequest, w http.ResponseWriter, metric *database.RequestMetrics) error {
	// Determine provider from model route
	provider, err := s.router.DetectProvider(req.Model)
	if err != nil {
		return s.errorHandler.HandleError(err, config.ProviderOpenAI, map[string]any{"model": req.Model})
	}

	client, err := s.getOpenAIClient(provider)
	if err != nil {
		return s.errorHandler.HandleError(err, provider, map[string]any{"model": req.Model})
	}

	log.Printf("Routing to OpenAI model: %s via provider: %s", req.Model, provider)
	messages := s.ConvertToOpenAIMessages(req.Messages)

	if req.Stream {
		return s.handleStreamingRequest(req, w, metric, func() error {
			return s.handleOpenAIStreaming(messages, req.Model, w, client)
		})
	}

	// Handle non-streaming
	completion, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    req.Model,
	})

	if err != nil {
		return s.handleNonStreamingError(err, w, metric, req.Model, provider)
	}

	// Extract token usage for metrics
	if completion.Usage.TotalTokens > 0 {
		metric.TokensUsed = int(completion.Usage.TotalTokens)
		metric.PromptTokens = int(completion.Usage.PromptTokens)
		metric.ResponseTokens = int(completion.Usage.CompletionTokens)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(completion)
	return nil
}

// handleStreamingRequest handles streaming logic for all providers
func (s *Server) handleStreamingRequest(
	req adapter.ChatRequest,
	w http.ResponseWriter,
	metric *database.RequestMetrics,
	streamFunc func() error,
) error {
	log.Printf("Starting streaming for model: %s", req.Model)

	// Set streaming headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	err := streamFunc()
	if err != nil {
		log.Printf("Streaming error: %v", err)
	}

	// For streaming requests, calculate accurate token usage based on message content
	metric.TokensUsed = s.CalculateTokenUsageForMessages(req.Messages, req.Model)
	metric.PromptTokens = metric.TokensUsed * 7 / 10 // Estimate ~70% prompt, 30% response
	metric.ResponseTokens = metric.TokensUsed - metric.PromptTokens

	return err
}

// handleNonStreamingError handles errors for non-streaming requests
func (s *Server) handleNonStreamingError(err error, w http.ResponseWriter, metric *database.RequestMetrics, model string, provider config.ProviderType) error {
	log.Printf("Provider error: %v", err)

	providerErr := s.errorHandler.HandleError(err, provider, map[string]any{"model": model})

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(providerErr.HTTPStatus)
	errorResp := providerErr.ToOpenAIError(metric.ID)
	json.NewEncoder(w).Encode(errorResp)

	return providerErr
}

// handleOpenAIStreaming handles streaming for OpenAI
func (s *Server) handleOpenAIStreaming(messages []openai.ChatCompletionMessageParamUnion, model string, w http.ResponseWriter, client *openai.Client) error {
	stream := client.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    model,
	})

	for stream.Next() {
		chunk := stream.Current()
		data, _ := json.Marshal(chunk)
		w.Write([]byte("data: " + string(data) + "\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	if err := stream.Err(); err != nil {
		return err
	}

	w.Write([]byte("data: [DONE]\n\n"))
	return nil
}

// extractTokenUsage extracts token usage from response for metrics
func (s *Server) extractTokenUsage(openaiResp map[string]any, metric *database.RequestMetrics) {
	if usage, ok := openaiResp["usage"].(map[string]any); ok {
		if totalTokens, ok := usage["total_tokens"].(int); ok {
			metric.TokensUsed = totalTokens
		}
		if promptTokens, ok := usage["prompt_tokens"].(int); ok {
			metric.PromptTokens = promptTokens
		}
		if completionTokens, ok := usage["completion_tokens"].(int); ok {
			metric.ResponseTokens = completionTokens
		}
	}
}

// EmbeddingHandler handles embedding requests
func (s *Server) EmbeddingHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	requestID := fmt.Sprintf("req_%d", time.Now().UnixNano())

	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("Request [%s]: %s %s from %s", requestID, r.Method, r.URL.Path, r.RemoteAddr)

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	var req EmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		http.Error(w, "Model field is required", http.StatusBadRequest)
		return
	}

	// Validate model
	if err := s.router.ValidateModel(req.Model); err != nil {
		providerErr := s.errorHandler.CreateValidationError(
			config.ProviderType("unknown"),
			fmt.Sprintf("Invalid model: %s", err.Error()),
			map[string]any{"model": req.Model},
		)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, false)
		return
	}

	// Detect provider
	provider, err := s.router.DetectProvider(req.Model)
	if err != nil {
		providerErr := s.errorHandler.CreateValidationError("", err.Error(), nil)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, false)
		return
	}

	// Check rate limiting
	if allowed, retryAfter := s.database.CheckRate(provider); !allowed {
		providerErr := s.errorHandler.CreateRateLimitError(provider, retryAfter)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, false)
		return
	}

	// Get the actual model name for API calls
	actualModelName, err := s.router.GetModelName(req.Model)
	if err != nil {
		providerErr := s.errorHandler.CreateValidationError(
			provider,
			fmt.Sprintf("Failed to get model name: %s", err.Error()),
			map[string]any{"route": req.Model},
		)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, false)
		return
	}

	// Get user information from context (if authenticated)
	var userID string
	if user, ok := auth.GetUserFromContext(r.Context()); ok {
		userID = user.ID
	}

	// Record request start
	metric := database.RequestMetrics{
		ID:        requestID,
		Timestamp: startTime,
		Provider:  provider,
		Model:     req.Model, // Keep original model name for metrics
		StartTime: startTime,
		Streaming: false, // Embeddings don't support streaming
		UserAgent: r.UserAgent(),
		ClientIP:  r.RemoteAddr,
	}

	// Handle embeddings for supported providers
	var responseErr error
	switch provider {
	case config.ProviderAzureOpenAI, config.ProviderOpenAI:
		responseErr = s.handleEmbedding(req, actualModelName, w, &metric, provider)
	case config.ProviderGoogleAI:
		responseErr = s.handleGeminiEmbedding(req, actualModelName, w, &metric, provider)
	case config.ProviderAWSBedrock:
		responseErr = s.handleBedrockEmbedding(req, actualModelName, w, &metric, provider)
	default:
		providerErr := s.errorHandler.CreateValidationError(
			provider,
			fmt.Sprintf("Embeddings are not supported for provider %s", provider),
			map[string]any{"provider": string(provider)},
		)
		s.sendErrorResponse(w, providerErr, requestID, req.Model, false)
		responseErr = providerErr
	}

	// Complete metrics
	metric.EndTime = time.Now()
	metric.Duration = metric.EndTime.Sub(metric.StartTime)
	metric.Success = (responseErr == nil)

	if responseErr != nil {
		if providerErr, ok := responseErr.(*errors.ProviderError); ok {
			metric.ErrorType = string(providerErr.Type)
			metric.ErrorMessage = providerErr.Message
			metric.StatusCode = providerErr.HTTPStatus
		} else {
			metric.ErrorType = "unknown"
			metric.ErrorMessage = responseErr.Error()
			metric.StatusCode = 500
		}
	} else {
		metric.StatusCode = 200
	}

	// Record metrics
	s.database.RecordRequest(metric, userID)
}

// handleEmbedding processes embedding requests for OpenAI providers
func (s *Server) handleEmbedding(req EmbeddingRequest, actualModelName string, w http.ResponseWriter, metric *database.RequestMetrics, provider config.ProviderType) error {
	log.Printf("Routing to embedding model: %s via provider: %s", actualModelName, provider)

	client, err := s.getOpenAIClient(provider)
	if err != nil {
		return s.errorHandler.HandleError(err, provider, map[string]any{"model": req.Model})
	}

	// Convert input to the format expected by OpenAI SDK
	var input []string
	switch v := req.Input.(type) {
	case string:
		input = []string{v}
	case []string:
		input = v
	case []interface{}:
		input = make([]string, len(v))
		for i, item := range v {
			if str, ok := item.(string); ok {
				input[i] = str
			} else {
				return s.errorHandler.CreateValidationError(provider, "All input items must be strings", nil)
			}
		}
	default:
		return s.errorHandler.CreateValidationError(provider, "Input must be a string or array of strings", nil)
	}

	// Try using JSON marshaling/unmarshaling to work around union type issues
	requestBody := map[string]interface{}{
		"input": input,
		"model": actualModelName,
	}

	if req.EncodingFormat != "" {
		requestBody["encoding_format"] = req.EncodingFormat
	}

	if req.Dimensions > 0 {
		requestBody["dimensions"] = req.Dimensions
	}

	if req.User != "" {
		requestBody["user"] = req.User
	}

	// Marshal to JSON and back to create the correct union type
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return s.errorHandler.CreateValidationError(provider, "Failed to marshal request", nil)
	}

	var params openai.EmbeddingNewParams
	if err := json.Unmarshal(jsonData, &params); err != nil {
		return s.errorHandler.CreateValidationError(provider, "Failed to unmarshal request", nil)
	}

	// Make the request
	embedding, err := client.Embeddings.New(context.Background(), params)
	if err != nil {
		return s.handleNonStreamingError(err, w, metric, req.Model, provider)
	}

	// Convert OpenAI response to our format
	data := make([]EmbeddingData, len(embedding.Data))
	for i, item := range embedding.Data {
		// Convert []float64 to []float32
		embedding32 := make([]float32, len(item.Embedding))
		for j, val := range item.Embedding {
			embedding32[j] = float32(val)
		}
		
		data[i] = EmbeddingData{
			Object:    "embedding",
			Index:     int(item.Index),
			Embedding: embedding32,
		}
	}

	response := EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  req.Model, // Return original model name
		Usage: EmbeddingUsage{
			PromptTokens: int(embedding.Usage.PromptTokens),
			TotalTokens:  int(embedding.Usage.TotalTokens),
		},
	}

	// Update metrics with token usage
	metric.TokensUsed = int(embedding.Usage.TotalTokens)
	metric.PromptTokens = int(embedding.Usage.PromptTokens)
	metric.ResponseTokens = 0 // Embeddings don't have response tokens

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
	return nil
}

// handleGeminiEmbedding processes embedding requests for Gemini models
func (s *Server) handleGeminiEmbedding(req EmbeddingRequest, actualModelName string, w http.ResponseWriter, metric *database.RequestMetrics, provider config.ProviderType) error {
	log.Printf("Routing to Gemini embedding model: %s", actualModelName)

	if s.geminiAdapter == nil {
		return s.errorHandler.HandleError(fmt.Errorf("Gemini client not available - missing API key configuration"), provider, map[string]any{"model": req.Model})
	}

	// Convert input to the format expected by Gemini
	var input []string
	switch v := req.Input.(type) {
	case string:
		input = []string{v}
	case []string:
		input = v
	case []interface{}:
		input = make([]string, len(v))
		for i, item := range v {
			if str, ok := item.(string); ok {
				input[i] = str
			} else {
				return s.errorHandler.CreateValidationError(provider, "All input items must be strings", nil)
			}
		}
	default:
		return s.errorHandler.CreateValidationError(provider, "Input must be a string or array of strings", nil)
	}

	// Call Gemini adapter
	geminiResponse, err := s.geminiAdapter.HandleEmbeddingRequest(input, actualModelName)
	if err != nil {
		return s.handleNonStreamingError(err, w, metric, req.Model, provider)
	}

	// Convert Gemini response to our format
	data, ok := geminiResponse["data"].([]map[string]any)
	if !ok {
		return s.errorHandler.CreateValidationError(provider, "Invalid response format from Gemini", nil)
	}

	// Convert to our EmbeddingData format
	embeddings := make([]EmbeddingData, len(data))
	for i, item := range data {
		embeddingSlice, ok := item["embedding"].([]float32)
		if !ok {
			// Try to convert from []float64 to []float32
			if embeddingFloat64, ok := item["embedding"].([]float64); ok {
				embeddingSlice = make([]float32, len(embeddingFloat64))
				for j, val := range embeddingFloat64 {
					embeddingSlice[j] = float32(val)
				}
			} else {
				return s.errorHandler.CreateValidationError(provider, "Invalid embedding format", nil)
			}
		}

		embeddings[i] = EmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: embeddingSlice,
		}
	}

	// Extract usage information
	usage, ok := geminiResponse["usage"].(map[string]any)
	if !ok {
		usage = map[string]any{"prompt_tokens": 0, "total_tokens": 0}
	}

	response := EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  req.Model, // Return original model name
		Usage: EmbeddingUsage{
			PromptTokens: int(usage["prompt_tokens"].(float64)),
			TotalTokens:  int(usage["total_tokens"].(float64)),
		},
	}

	// Update metrics with token usage
	metric.TokensUsed = response.Usage.TotalTokens
	metric.PromptTokens = response.Usage.PromptTokens
	metric.ResponseTokens = 0 // Embeddings don't have response tokens

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
	return nil
}

// handleBedrockEmbedding processes embedding requests for AWS Bedrock models
func (s *Server) handleBedrockEmbedding(req EmbeddingRequest, actualModelName string, w http.ResponseWriter, metric *database.RequestMetrics, provider config.ProviderType) error {
	log.Printf("Routing to Bedrock embedding model: %s", actualModelName)

	if s.bedrockClient == nil {
		return s.errorHandler.HandleError(fmt.Errorf("Bedrock client not available - missing AWS credentials"), provider, map[string]any{"model": req.Model})
	}

	// Convert input to the format expected by Bedrock
	var input []string
	switch v := req.Input.(type) {
	case string:
		input = []string{v}
	case []string:
		input = v
	case []interface{}:
		input = make([]string, len(v))
		for i, item := range v {
			if str, ok := item.(string); ok {
				input[i] = str
			} else {
				return s.errorHandler.CreateValidationError(provider, "All input items must be strings", nil)
			}
		}
	default:
		return s.errorHandler.CreateValidationError(provider, "Input must be a string or array of strings", nil)
	}

	// Call Bedrock embedding API
	bedrockResponse, err := s.callBedrockEmbedding(input, actualModelName)
	if err != nil {
		return s.handleNonStreamingError(err, w, metric, req.Model, provider)
	}

	// Convert to our EmbeddingData format
	embeddings := make([]EmbeddingData, len(bedrockResponse.Embeddings))
	for i, embedding := range bedrockResponse.Embeddings {
		embeddings[i] = EmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: embedding,
		}
	}

	response := EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  req.Model, // Return original model name
		Usage: EmbeddingUsage{
			PromptTokens: bedrockResponse.Usage.PromptTokens,
			TotalTokens:  bedrockResponse.Usage.TotalTokens,
		},
	}

	// Update metrics with token usage
	metric.TokensUsed = response.Usage.TotalTokens
	metric.PromptTokens = response.Usage.PromptTokens
	metric.ResponseTokens = 0 // Embeddings don't have response tokens

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
	return nil
}

// BedrockEmbeddingResponse represents a response from Bedrock embedding API
type BedrockEmbeddingResponse struct {
	Embeddings [][]float32            `json:"embeddings"`
	Usage      BedrockEmbeddingUsage  `json:"usage"`
}

// BedrockEmbeddingUsage represents token usage for Bedrock embeddings
type BedrockEmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// callBedrockEmbedding calls the appropriate Bedrock embedding API based on the model
func (s *Server) callBedrockEmbedding(input []string, modelName string) (*BedrockEmbeddingResponse, error) {
	switch {
	case strings.Contains(modelName, "amazon.titan-embed"):
		return s.callTitanEmbedding(input, modelName)
	case strings.Contains(modelName, "cohere.embed"):
		return s.callCohereEmbedding(input, modelName)
	default:
		return nil, fmt.Errorf("unsupported Bedrock embedding model: %s", modelName)
	}
}

// callTitanEmbedding calls Amazon Titan embedding models
func (s *Server) callTitanEmbedding(input []string, modelName string) (*BedrockEmbeddingResponse, error) {
	var allEmbeddings [][]float32
	totalTokens := 0

	for _, text := range input {
		// Prepare request body for Titan embedding
		requestBody := map[string]interface{}{
			"inputText": text,
		}

		// Calculate accurate token count
		tokens := s.CalculateTokenUsage(text, modelName)
		totalTokens += tokens

		// Call Bedrock API
		embedding, err := s.invokeBedrockModel(modelName, requestBody)
		if err != nil {
			return nil, err
		}

		// Parse Titan response
		titanResp, ok := embedding.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid response format from Titan")
		}

		embeddingData, ok := titanResp["embedding"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("no embedding data in Titan response")
		}

		// Convert to []float32
		embeddingFloat32 := make([]float32, len(embeddingData))
		for i, val := range embeddingData {
			if floatVal, ok := val.(float64); ok {
				embeddingFloat32[i] = float32(floatVal)
			} else {
				return nil, fmt.Errorf("invalid embedding value type")
			}
		}

		allEmbeddings = append(allEmbeddings, embeddingFloat32)
	}

	return &BedrockEmbeddingResponse{
		Embeddings: allEmbeddings,
		Usage: BedrockEmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}, nil
}

// callCohereEmbedding calls Cohere embedding models
func (s *Server) callCohereEmbedding(input []string, modelName string) (*BedrockEmbeddingResponse, error) {
	// Prepare request body for Cohere embedding
	requestBody := map[string]interface{}{
		"texts":      input,
		"input_type": "search_document", // Default input type
	}

	// Calculate accurate token count
	totalTokens := 0
	for _, text := range input {
		tokens := s.CalculateTokenUsage(text, modelName)
		totalTokens += tokens
	}

	// Call Bedrock API
	embedding, err := s.invokeBedrockModel(modelName, requestBody)
	if err != nil {
		return nil, err
	}

	// Parse Cohere response
	cohereResp, ok := embedding.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response format from Cohere")
	}

	embeddingsData, ok := cohereResp["embeddings"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("no embeddings data in Cohere response")
	}

	// Convert to [][]float32
	allEmbeddings := make([][]float32, len(embeddingsData))
	for i, embeddingInterface := range embeddingsData {
		embeddingArray, ok := embeddingInterface.([]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid embedding array format")
		}

		embeddingFloat32 := make([]float32, len(embeddingArray))
		for j, val := range embeddingArray {
			if floatVal, ok := val.(float64); ok {
				embeddingFloat32[j] = float32(floatVal)
			} else {
				return nil, fmt.Errorf("invalid embedding value type")
			}
		}
		allEmbeddings[i] = embeddingFloat32
	}

	return &BedrockEmbeddingResponse{
		Embeddings: allEmbeddings,
		Usage: BedrockEmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}, nil
}

// invokeBedrockModel invokes a Bedrock model with the given request body
func (s *Server) invokeBedrockModel(modelName string, requestBody map[string]interface{}) (interface{}, error) {
	// Marshal request body to JSON
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %v", err)
	}

	// Invoke Bedrock model
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     &modelName,
		Body:        bodyBytes,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	}

	result, err := s.bedrockClient.InvokeModel(context.Background(), input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke Bedrock model: %v", err)
	}

	// Parse response
	var response interface{}
	if err := json.Unmarshal(result.Body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	return response, nil
}

// ModelsHandler returns available models
func (s *Server) ModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	models := s.router.GetAvailableModels()

	// Convert to OpenAI format
	openaiModels := make([]map[string]any, 0)

	for provider, modelRoutes := range models {
		for _, route := range modelRoutes {
			openaiModels = append(openaiModels, map[string]any{
				"id":         route.RouteName,
				"object":     "model",
				"created":    time.Now().Unix(),
				"owned_by":   string(provider),
				"permission": []any{},
				"root":       route.RouteName,
				"parent":     nil,
				"meta": map[string]any{
					"display_name":       route.DisplayName,
					"description":        route.Description,
					"max_tokens":         route.MaxTokens,
					"supports_streaming": route.SupportsStreaming,
					"provider":           string(provider),
				},
			})
		}
	}

	response := map[string]any{
		"object": "list",
		"data":   openaiModels,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// MetricsHandler returns current metrics
func (s *Server) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check if metrics are enabled
	if !s.config.Metrics.Enabled {
		http.Error(w, "Metrics are disabled", http.StatusServiceUnavailable)
		return
	}

	// Get query parameters
	format := r.URL.Query().Get("format")
	limitStr := r.URL.Query().Get("limit")

	var limit int = 100
	if limitStr != "" {
		if parsedLimit, err := strconv.Atoi(limitStr); err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	// Get usage stats
	usage := s.database.GetUsageStats()
	history := s.database.GetRequestHistory(limit)

	response := map[string]any{
		"usage":     usage,
		"history":   history,
		"timestamp": time.Now(),
	}

	// Add rate limit status
	rateLimitStatus := s.database.GetAllRateStatus()
	response["rate_limits"] = rateLimitStatus

	w.Header().Set("Content-Type", "application/json")

	if format == "pretty" {
		encoder := json.NewEncoder(w)
		encoder.SetIndent("", "  ")
		encoder.Encode(response)
	} else {
		json.NewEncoder(w).Encode(response)
	}
}

// getTokenEncoder returns the appropriate tiktoken encoder for a model
func (s *Server) getTokenEncoder(modelName string) (*tiktoken.Tiktoken, error) {
	// Map model names to appropriate encodings
	switch {
	case strings.Contains(modelName, "gpt-4"):
		return tiktoken.GetEncoding("cl100k_base")
	case strings.Contains(modelName, "gpt-3.5"):
		return tiktoken.GetEncoding("cl100k_base")
	case strings.Contains(modelName, "text-embedding"):
		return tiktoken.GetEncoding("cl100k_base")
	case strings.Contains(modelName, "claude"):
		// Claude uses a similar tokenizer to GPT-4
		return tiktoken.GetEncoding("cl100k_base")
	case strings.Contains(modelName, "gemini"):
		// Gemini uses a different tokenizer, but cl100k_base is a reasonable approximation
		return tiktoken.GetEncoding("cl100k_base")
	case strings.Contains(modelName, "titan"):
		// Amazon Titan uses a different tokenizer, but cl100k_base is a reasonable approximation
		return tiktoken.GetEncoding("cl100k_base")
	case strings.Contains(modelName, "cohere"):
		// Cohere uses a different tokenizer, but cl100k_base is a reasonable approximation
		return tiktoken.GetEncoding("cl100k_base")
	default:
		// Default to cl100k_base for unknown models
		return tiktoken.GetEncoding("cl100k_base")
	}
}

// CalculateTokenUsage accurately calculates token usage using tiktoken
func (s *Server) CalculateTokenUsage(text string, modelName string) int {
	encoder, err := s.getTokenEncoder(modelName)
	if err != nil {
		log.Printf("Failed to get encoder for model %s: %v, falling back to estimation", modelName, err)
		return s.estimateTokenUsageFromText(text)
	}

	tokens := encoder.Encode(text, nil, nil)
	return len(tokens)
}

// CalculateTokenUsageForMessages accurately calculates token usage for chat messages
func (s *Server) CalculateTokenUsageForMessages(messages []adapter.Message, modelName string) int {
	encoder, err := s.getTokenEncoder(modelName)
	if err != nil {
		log.Printf("Failed to get encoder for model %s: %v, falling back to estimation", modelName, err)
		return s.estimateTokenUsage(messages)
	}

	totalTokens := 0

	for _, msg := range messages {
		contentStr := s.getContentAsString(msg.Content)
		
		// Count tokens for the message content
		contentTokens := encoder.Encode(contentStr, nil, nil)
		totalTokens += len(contentTokens)

		// Add overhead for message structure (role, etc.)
		// Different models have different overhead patterns
		switch {
		case strings.Contains(modelName, "gpt"):
			totalTokens += 3 // GPT models add ~3 tokens per message for structure
		case strings.Contains(modelName, "claude"):
			totalTokens += 2 // Claude models have less overhead
		default:
			totalTokens += 3 // Default overhead
		}
	}

	// Minimum of 1 token for empty input
	if totalTokens < 1 {
		totalTokens = 1
	}

	return totalTokens
}

// estimateTokenUsageFromText provides a fallback estimation for a single text string
func (s *Server) estimateTokenUsageFromText(text string) int {
	charCount := len(text)
	wordCount := len(strings.Fields(text))

	// Base token estimation using character count (more accurate than word count)
	// GPT models use ~4 characters per token on average for English text
	tokenEstimate := charCount / 4

	// Alternative estimation using word count (~1.3 tokens per word)
	wordTokenEstimate := (wordCount * 13) / 10

	// Use the higher of the two estimates for better accuracy
	tokens := tokenEstimate
	if wordTokenEstimate > tokenEstimate {
		tokens = wordTokenEstimate
	}

	// Don't enforce minimum here - let the caller decide
	// Empty content should return 0 tokens
	if tokens < 0 {
		tokens = 0
	}

	return tokens
}

// estimateTokenUsage provides a fallback estimate of token usage based on word count
// This is used when tiktoken fails or for models without specific encodings
func (s *Server) estimateTokenUsage(messages []adapter.Message) int {
	totalTokens := 0

	for _, msg := range messages {
		contentStr := s.getContentAsString(msg.Content)
		msgTokens := s.estimateTokenUsageFromText(contentStr)
		
		// Add overhead for message structure (3-4 tokens per message)
		msgTokens += 3
		totalTokens += msgTokens
	}

	// Minimum of 1 token for empty input
	if totalTokens < 1 {
		totalTokens = 1
	}

	return totalTokens
}
