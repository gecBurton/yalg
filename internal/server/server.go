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
	"github.com/google/generative-ai-go/genai"
	"github.com/openai/openai-go"
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
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   TokenUsage  `json:"usage"`
}

// ErrorResponse represents an API error response
type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// ServerConfig holds configuration for the server
type ServerConfig struct {
	Config            *config.Config
	BedrockClient     *bedrockruntime.Client
	GeminiClient      *genai.Client
	AzureOpenAIClient *openai.Client
	DirectOpenAIClient *openai.Client
	AnthropicAPIKey   string
	AWSConfig         *aws.Config
	Database          *database.DB
	ErrorHandler      *errors.ErrorHandler
	Router            *router.Router
}

// Server provides LLM gateway functionality with metrics, rate limiting, etc.
type Server struct {
	claudeAdapter      *adapter.ClaudeAdapter
	geminiAdapter      *adapter.GeminiAdapter
	azureOpenAIClient  *openai.Client
	directOpenAIClient *openai.Client
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
		switch msg.Role {
		case "assistant":
			result[i] = openai.AssistantMessage(msg.Content)
		case "system":
			result[i] = openai.SystemMessage(msg.Content)
		default:
			result[i] = openai.UserMessage(msg.Content)
		}
	}
	return result
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
			"total_requests":     usage.TotalRequests,
			"successful_requests": usage.SuccessfulRequests,
			"failed_requests":    usage.FailedRequests,
			"average_latency":    usage.AverageLatency,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(health)
}

// UIHandler serves the UI HTML file
func (s *Server) UIHandler(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "ui.html")
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

	// For streaming requests, estimate token usage based on message content
	metric.TokensUsed = s.estimateTokenUsage(req.Messages)
	metric.PromptTokens = metric.TokensUsed * 7 / 10  // Estimate ~70% prompt, 30% response
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

// estimateTokenUsage provides a rough estimate of token usage based on word count
// This is used for streaming requests where exact token counts aren't available
func (s *Server) estimateTokenUsage(messages []adapter.Message) int {
	totalWords := 0
	for _, msg := range messages {
		// Count words by splitting on whitespace
		words := len(strings.Fields(msg.Content))
		totalWords += words
	}

	// More accurate estimate: ~1.3 tokens per word for English text
	// This accounts for subword tokenization (BPE) used by modern LLMs
	promptTokens := (totalWords * 13) / 10

	// Minimum of 10 tokens for very short messages
	if promptTokens < 10 {
		promptTokens = 10
	}

	// For streaming, estimate response tokens based on prompt length
	// Most LLM responses are 50-150% of prompt length, use 80% as default
	responseTokens := (promptTokens * 8) / 10
	if responseTokens < 15 {
		responseTokens = 15 // Minimum response
	}

	return promptTokens + responseTokens
}