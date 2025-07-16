package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/aws"
)

// Message represents an OpenAI chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ToolCall represents a tool call in OpenAI format
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call within a tool call
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolMessage represents a tool response message
type ToolMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents an OpenAI chat completion request
type ChatRequest struct {
	Messages       []Message                `json:"messages"`
	Model          string                   `json:"model"`
	Stream         bool                     `json:"stream"`
	Tools          []map[string]interface{} `json:"tools,omitempty"`
	ResponseFormat map[string]interface{}   `json:"response_format,omitempty"`
	Thinking       map[string]interface{}   `json:"thinking,omitempty"`
}

// ClaudeAdapter handles Claude models via both direct Anthropic API and AWS Bedrock
type ClaudeAdapter struct {
	directClient     anthropic.Client // For direct Anthropic API
	bedrockClient    anthropic.Client // For AWS Bedrock
	hasDirectClient  bool             // Track if direct client is initialized
	hasBedrockClient bool             // Track if Bedrock client is initialized
}

// ClaudeAdapterConfig holds configuration for creating Claude adapters
type ClaudeAdapterConfig struct {
	AnthropicAPIKey string
	AWSConfig       *aws.Config // For Bedrock authentication
}

// NewClaudeAdapter creates a new unified Claude adapter
func NewClaudeAdapter(config ClaudeAdapterConfig) *ClaudeAdapter {
	adapter := &ClaudeAdapter{}

	// Initialize direct Anthropic client if API key is provided
	if config.AnthropicAPIKey != "" {
		adapter.directClient = anthropic.NewClient(
			option.WithAPIKey(config.AnthropicAPIKey),
		)
		adapter.hasDirectClient = true
		log.Printf("Direct Anthropic client initialized")
	}

	// Initialize Bedrock client if AWS config is provided
	if config.AWSConfig != nil {
		adapter.bedrockClient = anthropic.NewClient(
			bedrock.WithConfig(*config.AWSConfig),
		)
		adapter.hasBedrockClient = true
		log.Printf("Anthropic Bedrock client initialized")
	}

	return adapter
}

// IsClaudeModel checks if a model name belongs to Claude/Anthropic
func (a *ClaudeAdapter) IsClaudeModel(model string) bool {
	lowerModel := strings.ToLower(model)
	return strings.Contains(lowerModel, "claude") ||
		strings.HasPrefix(model, "anthropic.")
}

// getClientForProvider returns the appropriate client based on provider type
func (a *ClaudeAdapter) getClientForProvider(provider string) (*anthropic.Client, error) {
	switch provider {
	case "anthropic":
		if !a.hasDirectClient {
			return nil, fmt.Errorf("direct Anthropic client not configured - missing API key")
		}
		return &a.directClient, nil
	case "aws-bedrock":
		if !a.hasBedrockClient {
			return nil, fmt.Errorf("bedrock client not configured - missing AWS credentials")
		}
		return &a.bedrockClient, nil
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// getClientForModel returns the appropriate client based on model prefix (deprecated - use getClientForProvider)
func (a *ClaudeAdapter) getClientForModel(model string) (*anthropic.Client, error) {
	if strings.HasPrefix(model, "anthropic/") {
		if !a.hasDirectClient {
			return nil, fmt.Errorf("direct Anthropic client not configured - missing API key")
		}
		return &a.directClient, nil
	}

	if strings.HasPrefix(model, "aws-bedrock/") {
		if !a.hasBedrockClient {
			return nil, fmt.Errorf("bedrock client not configured - missing AWS credentials")
		}
		return &a.bedrockClient, nil
	}

	return nil, fmt.Errorf("unsupported model prefix: %s", model)
}

// normalizeModelName removes provider prefixes for API calls
func (a *ClaudeAdapter) normalizeModelName(model string) string {
	if strings.HasPrefix(model, "anthropic/") {
		return strings.TrimPrefix(model, "anthropic/")
	}
	if strings.HasPrefix(model, "aws-bedrock/") {
		return strings.TrimPrefix(model, "aws-bedrock/")
	}
	return model
}

// ConvertToAnthropicMessages converts OpenAI messages to Anthropic format
func (a *ClaudeAdapter) ConvertToAnthropicMessages(messages []Message) ([]anthropic.MessageParam, string) {
	var anthropicMessages []anthropic.MessageParam
	var systemMessage string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemMessage = msg.Content
		case "user":
			anthropicMessages = append(anthropicMessages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		case "assistant":
			anthropicMessages = append(anthropicMessages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		}
	}

	return anthropicMessages, systemMessage
}

// ConvertAnthropicToOpenAI converts Anthropic response to OpenAI format with thinking/reasoning support
func (a *ClaudeAdapter) ConvertAnthropicToOpenAI(anthropicResp *anthropic.Message, model string) map[string]interface{} {
	// Extract content and thinking blocks separately
	var content string
	var reasoningContent string
	var thinkingBlocks []map[string]interface{}

	for _, block := range anthropicResp.Content {
		if block.Type == "text" {
			content += block.Text
		} else if block.Type == "thinking" {
			reasoningContent = block.Text
			thinkingBlocks = append(thinkingBlocks, map[string]interface{}{
				"type":    "thinking",
				"content": block.Text,
			})
		}
	}

	// Extract token usage
	promptTokens := int(anthropicResp.Usage.InputTokens)
	completionTokens := int(anthropicResp.Usage.OutputTokens)
	totalTokens := promptTokens + completionTokens

	// Convert stop reason
	finishReason := "stop"
	if anthropicResp.StopReason != "" {
		finishReason = string(anthropicResp.StopReason)
	}

	// Build message with thinking support
	message := map[string]interface{}{
		"role":    "assistant",
		"content": content,
	}

	// Add reasoning fields if thinking content exists
	if reasoningContent != "" {
		message["reasoning_content"] = reasoningContent
	}

	if len(thinkingBlocks) > 0 {
		message["thinking_blocks"] = thinkingBlocks
	}

	return map[string]interface{}{
		"id":      "chatcmpl-" + anthropicResp.ID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      totalTokens,
		},
	}
}

// ConvertAnthropicStreamToOpenAI converts Anthropic stream events to OpenAI format
func (a *ClaudeAdapter) ConvertAnthropicStreamToOpenAI(event anthropic.MessageStreamEventUnion, model string) map[string]interface{} {
	openaiChunk := map[string]interface{}{
		"id":      "chatcmpl-" + fmt.Sprintf("%d", time.Now().UnixNano()),
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": nil,
			},
		},
	}

	choice := openaiChunk["choices"].([]map[string]interface{})[0]
	delta := choice["delta"].(map[string]interface{})

	// Handle different types of Anthropic streaming events
	switch event.Type {
	case "message_start":
		// Start of message - add role
		delta["role"] = "assistant"

	case "content_block_start":
		// Handle tool use start events with index conversion
		if event.Index > 0 {
			// Convert Anthropic index (1-based) to OpenAI index (0-based)
			openaiIndex := int(event.Index - 1)

			// Create tool_calls delta for this block
			delta["tool_calls"] = []interface{}{
				map[string]interface{}{
					"index": openaiIndex,
					"type":  "function",
				},
			}
		}

	case "content_block_delta":
		// Content chunk - extract text from delta
		if event.Delta.Text != "" {
			if event.Index > 0 {
				// This is a tool call delta
				openaiIndex := int(event.Index - 1)
				delta["tool_calls"] = []interface{}{
					map[string]interface{}{
						"index": openaiIndex,
						"function": map[string]interface{}{
							"arguments": event.Delta.Text,
						},
					},
				}
			} else {
				// Regular text content
				delta["content"] = event.Delta.Text
			}
		}

	case "content_block_stop":
		// Skip content block stop events

	case "message_delta":
		// Message completion info
		if event.Delta.StopReason != "" {
			choice["finish_reason"] = string(event.Delta.StopReason)
		}

	case "message_stop":
		// End of message
		choice["finish_reason"] = "stop"

	default:
		// Log unknown event types
		log.Printf("Unknown stream event type: %s", event.Type)
	}

	return openaiChunk
}

// HandleRequestWithProvider processes non-streaming requests for Claude models with explicit provider
func (a *ClaudeAdapter) HandleRequestWithProvider(req ChatRequest, provider string) (map[string]interface{}, error) {
	log.Printf("Processing non-streaming request for Claude model: %s via provider: %s", req.Model, provider)

	client, err := a.getClientForProvider(provider)
	if err != nil {
		return nil, err
	}

	// Model name is already the actual model name (no prefix to remove)
	anthropicMessages, systemMessage := a.ConvertToAnthropicMessages(req.Messages)

	// Build the message params
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: 1000,
		Messages:  anthropicMessages,
	}

	if systemMessage != "" {
		params.System = []anthropic.TextBlockParam{
			{
				Text: systemMessage,
			},
		}
	}

	log.Printf("Calling Anthropic API (%s) with %d messages", req.Model, len(anthropicMessages))

	// Call Anthropic API (works for both direct and Bedrock)
	result, err := client.Messages.New(context.Background(), params)
	if err != nil {
		log.Printf("Anthropic API error for model %s: %v", req.Model, err)
		return nil, err
	}

	log.Printf("Anthropic API response received, content blocks: %d", len(result.Content))

	// Convert to OpenAI format
	response := a.ConvertAnthropicToOpenAI(result, req.Model)

	// Log the response structure for debugging
	if data, err := json.Marshal(response); err == nil {
		log.Printf("Converted response: %s", string(data))
	} else {
		log.Printf("Error marshaling response: %v", err)
	}

	return response, nil
}

// HandleStreamingRequestWithProvider processes streaming requests for Claude models with explicit provider
func (a *ClaudeAdapter) HandleStreamingRequestWithProvider(req ChatRequest, w http.ResponseWriter, provider string) error {
	log.Printf("Processing streaming request for Claude model: %s via provider: %s", req.Model, provider)

	client, err := a.getClientForProvider(provider)
	if err != nil {
		return err
	}

	// Model name is already the actual model name (no prefix to remove)
	anthropicMessages, systemMessage := a.ConvertToAnthropicMessages(req.Messages)

	// Build the message params
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: 1000,
		Messages:  anthropicMessages,
	}

	if systemMessage != "" {
		params.System = []anthropic.TextBlockParam{
			{
				Text: systemMessage,
			},
		}
	}

	log.Printf("Calling Anthropic streaming API (%s) with %d messages", req.Model, len(anthropicMessages))

	// Call Anthropic streaming API (works for both direct and Bedrock)
	stream := client.Messages.NewStreaming(context.Background(), params)

	// Process the stream
	eventCount := 0
	for stream.Next() {
		event := stream.Current()
		eventCount++

		log.Printf("Received stream event %d: type=%s", eventCount, event.Type)

		// Convert to OpenAI format
		openaiChunk := a.ConvertAnthropicStreamToOpenAI(event, req.Model)

		// Skip empty chunks that don't add value
		choices := openaiChunk["choices"].([]map[string]interface{})
		if len(choices) > 0 {
			delta := choices[0]["delta"].(map[string]interface{})
			finishReason := choices[0]["finish_reason"]

			// Only send chunk if it has content, role, or finish_reason
			if len(delta) > 0 || finishReason != nil {
				// Send as SSE
				data, err := json.Marshal(openaiChunk)
				if err != nil {
					log.Printf("Error marshaling chunk: %v", err)
					continue
				}

				w.Write([]byte("data: " + string(data) + "\n\n"))
			}
		}

		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	if err := stream.Err(); err != nil {
		log.Printf("Anthropic streaming error for model %s: %v", req.Model, err)
		return err
	}

	log.Printf("Stream completed after %d events", eventCount)

	// Send completion marker
	w.Write([]byte("data: [DONE]\n\n"))

	return nil
}

// HandleRequest processes non-streaming requests for Claude models (backward compatibility)
func (a *ClaudeAdapter) HandleRequest(req ChatRequest) (map[string]interface{}, error) {
	// Try to detect provider from model name for backward compatibility
	var provider string
	if strings.HasPrefix(req.Model, "anthropic/") {
		provider = "anthropic"
		// Remove prefix for API call
		req.Model = strings.TrimPrefix(req.Model, "anthropic/")
	} else if strings.HasPrefix(req.Model, "aws-bedrock/") {
		provider = "aws-bedrock"
		// Remove prefix for API call
		req.Model = strings.TrimPrefix(req.Model, "aws-bedrock/")
	} else {
		// Default to aws-bedrock for backward compatibility
		provider = "aws-bedrock"
	}

	return a.HandleRequestWithProvider(req, provider)
}

// HandleStreamingRequest processes streaming requests for Claude models (backward compatibility)
func (a *ClaudeAdapter) HandleStreamingRequest(req ChatRequest, w http.ResponseWriter) error {
	// Try to detect provider from model name for backward compatibility
	var provider string
	if strings.HasPrefix(req.Model, "anthropic/") {
		provider = "anthropic"
		// Remove prefix for API call
		req.Model = strings.TrimPrefix(req.Model, "anthropic/")
	} else if strings.HasPrefix(req.Model, "aws-bedrock/") {
		provider = "aws-bedrock"
		// Remove prefix for API call
		req.Model = strings.TrimPrefix(req.Model, "aws-bedrock/")
	} else {
		// Default to aws-bedrock for backward compatibility
		provider = "aws-bedrock"
	}

	return a.HandleStreamingRequestWithProvider(req, w, provider)
}

// ProcessAnthropicHeaders processes Anthropic response headers and converts them to OpenAI-compatible format
func (a *ClaudeAdapter) ProcessAnthropicHeaders(headers map[string]string) map[string]string {
	result := make(map[string]string)

	// Map of Anthropic rate limit headers to OpenAI format
	anthropicToOpenAI := map[string]string{
		"anthropic-ratelimit-requests-limit":     "x-ratelimit-limit-requests",
		"anthropic-ratelimit-requests-remaining": "x-ratelimit-remaining-requests",
		"anthropic-ratelimit-tokens-limit":       "x-ratelimit-limit-tokens",
		"anthropic-ratelimit-tokens-remaining":   "x-ratelimit-remaining-tokens",
	}

	// Process all headers
	for key, value := range headers {
		// Check if it's an Anthropic rate limit header that needs conversion
		if openaiKey, exists := anthropicToOpenAI[key]; exists {
			// Add both OpenAI format and LLM provider format
			result[openaiKey] = value
			result["llm_provider-"+key] = value
		} else {
			// Add other headers with llm_provider prefix
			result["llm_provider-"+key] = value
		}
	}

	return result
}

// AnthropicHeaderConfig holds configuration for generating Anthropic headers
type AnthropicHeaderConfig struct {
	APIKey           string
	ComputerToolUsed bool
	PromptCachingSet bool
}

// GetAnthropicHeaders generates appropriate headers for Anthropic API requests
func (a *ClaudeAdapter) GetAnthropicHeaders(config *AnthropicHeaderConfig) map[string]string {
	headers := make(map[string]string)

	// Always add the API key header
	if config.APIKey != "" {
		headers["x-api-key"] = config.APIKey
	}

	// Add beta headers if special features are used
	var betaFeatures []string

	if config.ComputerToolUsed {
		betaFeatures = append(betaFeatures, "computer-use-2024-10-22")
	}

	if config.PromptCachingSet {
		betaFeatures = append(betaFeatures, "prompt-caching-2024-07-31")
	}

	// Add anthropic-beta header if any beta features are used
	if len(betaFeatures) > 0 {
		headers["anthropic-beta"] = strings.Join(betaFeatures, ",")
	}

	return headers
}

// MapToolHelper transforms OpenAI tool format to Anthropic format with cache control support
func (a *ClaudeAdapter) MapToolHelper(tool map[string]interface{}) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Extract function information
	function, ok := tool["function"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("tool missing function field")
	}

	// Set the tool name from function name
	if name, exists := function["name"]; exists {
		result["name"] = name
	}

	// Set the description from function description
	if description, exists := function["description"]; exists {
		result["description"] = description
	}

	// Transform parameters to input_schema
	if parameters, exists := function["parameters"]; exists {
		result["input_schema"] = parameters
	}

	// Handle cache control - check both inside function and outside function
	var cacheControl map[string]interface{}

	// First check if cache_control is inside the function
	if functionCacheControl, exists := function["cache_control"]; exists {
		if cc, ok := functionCacheControl.(map[string]interface{}); ok {
			cacheControl = cc
		}
	}

	// Then check if cache_control is at the tool level (outside function)
	if toolCacheControl, exists := tool["cache_control"]; exists {
		if cc, ok := toolCacheControl.(map[string]interface{}); ok {
			cacheControl = cc
		}
	}

	// Add cache control if found
	if cacheControl != nil {
		result["cache_control"] = cacheControl
	}

	return result, nil
}

// CreateJSONToolCallForResponseFormat creates a tool call for JSON response format
func (a *ClaudeAdapter) CreateJSONToolCallForResponseFormat(schema map[string]interface{}) map[string]interface{} {
	tool := map[string]interface{}{
		"name": "json_tool_call",
	}

	var inputSchema map[string]interface{}

	if schema != nil {
		// Use the provided schema
		inputSchema = schema
	} else {
		// Default schema for no constraints
		inputSchema = map[string]interface{}{
			"type":                 "object",
			"additionalProperties": true,
			"properties":           map[string]interface{}{},
		}
	}

	tool["input_schema"] = inputSchema

	return tool
}

// ConvertToolResponseToMessage converts tool call responses to a message format
func (a *ClaudeAdapter) ConvertToolResponseToMessage(toolCalls []ToolCall) *ToolMessage {
	if len(toolCalls) == 0 {
		return nil
	}

	// Find the json_tool_call
	for _, toolCall := range toolCalls {
		if toolCall.Function.Name == "json_tool_call" {
			if toolCall.Function.Arguments == "" {
				return nil
			}

			// Try to parse the arguments as JSON to extract values
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err == nil {
				// Check if there's a "values" key
				if values, exists := args["values"]; exists {
					if valuesJSON, err := json.Marshal(values); err == nil {
						return &ToolMessage{
							Role:    "assistant",
							Content: string(valuesJSON),
						}
					}
				}
			}

			// If no values key or JSON parsing failed, return the arguments as-is
			return &ToolMessage{
				Role:    "assistant",
				Content: toolCall.Function.Arguments,
			}
		}
	}

	return nil
}

// TransformTools transforms a list of tools, preserving computer tools and transforming others
func (a *ClaudeAdapter) TransformTools(tools []map[string]interface{}) ([]map[string]interface{}, error) {
	var result []map[string]interface{}

	for _, tool := range tools {
		// Check if it's a computer tool - preserve as-is
		if toolType, exists := tool["type"]; exists && toolType == "computer_20241022" {
			result = append(result, tool)
			continue
		}

		// For other tools, use MapToolHelper to transform them
		transformedTool, err := a.MapToolHelper(tool)
		if err != nil {
			return nil, fmt.Errorf("failed to transform tool: %w", err)
		}

		result = append(result, transformedTool)
	}

	return result, nil
}

// ConvertAnthropicToOpenAIWithCitations converts Anthropic response to OpenAI format with citations support
func (a *ClaudeAdapter) ConvertAnthropicToOpenAIWithCitations(anthropicResp *anthropic.Message, model string, citations []map[string]interface{}) map[string]interface{} {
	// Start with the standard conversion
	result := a.ConvertAnthropicToOpenAI(anthropicResp, model)

	// Add citations to provider_specific_fields if provided
	if len(citations) > 0 {
		choices, ok := result["choices"].([]map[string]interface{})
		if ok && len(choices) > 0 {
			message, ok := choices[0]["message"].(map[string]interface{})
			if ok {
				// Add provider_specific_fields to existing message
				message["provider_specific_fields"] = map[string]interface{}{
					"citations": citations,
				}
			}
		}
	}

	return result
}
