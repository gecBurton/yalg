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

// ChatRequest represents an OpenAI chat completion request
type ChatRequest struct {
	Messages []Message `json:"messages"`
	Model    string    `json:"model"`
	Stream   bool      `json:"stream"`
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

// ConvertAnthropicToOpenAI converts Anthropic response to OpenAI format
func (a *ClaudeAdapter) ConvertAnthropicToOpenAI(anthropicResp *anthropic.Message, model string) map[string]interface{} {
	// Extract content
	var content string
	for _, block := range anthropicResp.Content {
		if block.Type == "text" {
			content += block.Text
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

	return map[string]interface{}{
		"id":      "chatcmpl-" + anthropicResp.ID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": content,
				},
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
		// Skip content block start events

	case "content_block_delta":
		// Content chunk - extract text from delta
		if event.Delta.Text != "" {
			delta["content"] = event.Delta.Text
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
