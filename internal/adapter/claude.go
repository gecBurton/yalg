package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/aws"
)

// ContentPart represents a part of multi-modal content
type ContentPart struct {
	Type     string                 `json:"type"`
	Text     string                 `json:"text,omitempty"`
	ImageURL *ImageURL              `json:"image_url,omitempty"`
	CacheControl *CacheControl      `json:"cache_control,omitempty"`
}

// ImageURL represents an image in a message
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// CacheControl represents cache control settings
type CacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// Message represents an OpenAI chat message
type Message struct {
	Role      string        `json:"role"`
	Content   interface{}   `json:"content"`  // Can be string or []ContentPart
	ToolCalls []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
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

// Tool represents a tool available to the model
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents a function that can be called
type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ChatRequest represents an OpenAI chat completion request
type ChatRequest struct {
	Messages         []Message                `json:"messages"`
	Model            string                   `json:"model"`
	Stream           bool                     `json:"stream"`
	Tools            []Tool                   `json:"tools,omitempty"`
	ToolChoice       interface{}              `json:"tool_choice,omitempty"`
	ResponseFormat   map[string]interface{}   `json:"response_format,omitempty"`
	Thinking         map[string]interface{}   `json:"thinking,omitempty"`
	MaxTokens        int                      `json:"max_tokens,omitempty"`
	Temperature      float64                  `json:"temperature,omitempty"`
	TopP             float64                  `json:"top_p,omitempty"`
	Stop             []string                 `json:"stop,omitempty"`
	FrequencyPenalty float64                  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64                  `json:"presence_penalty,omitempty"`
	N                int                      `json:"n,omitempty"`
	Modalities       []string                 `json:"modalities,omitempty"`
	WebSearchOptions map[string]interface{}   `json:"web_search_options,omitempty"`
	URLContext       map[string]interface{}   `json:"url_context,omitempty"`
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

// getContentAsString extracts content as string from Message.Content
func getContentAsString(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []ContentPart:
		// Extract text from content parts
		var textParts []string
		for _, part := range c {
			if part.Type == "text" {
				textParts = append(textParts, part.Text)
			}
		}
		return strings.Join(textParts, " ")
	default:
		return ""
	}
}

// extractURLsFromContent extracts URLs from content string
func extractURLsFromContent(content string) []string {
	// Regular expression to match URLs
	urlRegex := regexp.MustCompile(`https?://[^\s]+`)
	return urlRegex.FindAllString(content, -1)
}

// isValidURL checks if a string is a valid URL
func isValidURL(str string) bool {
	u, err := url.Parse(str)
	return err == nil && u.Scheme != "" && u.Host != ""
}

// fetchURLContent fetches content from a URL (basic implementation)
func fetchURLContent(urlStr string) (string, error) {
	// Basic URL validation
	if !isValidURL(urlStr) {
		return "", fmt.Errorf("invalid URL: %s", urlStr)
	}

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Make GET request
	resp, err := client.Get(urlStr)
	if err != nil {
		return "", fmt.Errorf("failed to fetch URL %s: %w", urlStr, err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP error %d for URL %s", resp.StatusCode, urlStr)
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body from %s: %w", urlStr, err)
	}

	// Basic HTML stripping (very basic implementation)
	content := string(body)
	content = stripBasicHTML(content)

	// Limit content length to prevent excessive context
	if len(content) > 10000 {
		content = content[:10000] + "... [content truncated]"
	}

	return content, nil
}

// stripBasicHTML removes basic HTML tags (very basic implementation)
func stripBasicHTML(content string) string {
	// Remove script and style tags and their contents
	scriptRegex := regexp.MustCompile(`(?i)<script[^>]*>.*?</script>`)
	content = scriptRegex.ReplaceAllString(content, "\n\n")
	
	styleRegex := regexp.MustCompile(`(?i)<style[^>]*>.*?</style>`)
	content = styleRegex.ReplaceAllString(content, "\n\n")
	
	// Remove title tags and their contents
	titleRegex := regexp.MustCompile(`(?i)<title[^>]*>.*?</title>`)
	content = titleRegex.ReplaceAllString(content, "")
	
	// Remove head tags and their contents
	headRegex := regexp.MustCompile(`(?i)<head[^>]*>.*?</head>`)
	content = headRegex.ReplaceAllString(content, "")
	
	// Replace block-level elements with newlines to preserve spacing
	blockRegex := regexp.MustCompile(`(?i)</(div|p|h[1-6]|li|ul|ol|blockquote|section|article|header|footer|main|nav|aside)>`)
	content = blockRegex.ReplaceAllString(content, "\n\n")
	
	// Remove remaining HTML tags
	htmlRegex := regexp.MustCompile(`<[^>]*>`)
	content = htmlRegex.ReplaceAllString(content, "")
	
	// Clean up excessive whitespace while preserving intentional spacing
	content = regexp.MustCompile(`\n\s*\n`).ReplaceAllString(content, "\n\n")
	content = regexp.MustCompile(`[ \t]+`).ReplaceAllString(content, " ")
	content = strings.TrimSpace(content)
	
	return content
}

// processContentWithURLContext processes content by fetching URL contexts
func (a *ClaudeAdapter) processContentWithURLContext(content string) string {
	// Extract URLs from content
	urls := extractURLsFromContent(content)
	
	if len(urls) == 0 {
		return content
	}
	
	// Build enhanced content with URL contexts
	var enhancedContent strings.Builder
	enhancedContent.WriteString(content)
	
	// Process each URL
	for _, urlStr := range urls {
		log.Printf("Processing URL context: %s", urlStr)
		
		// Fetch URL content
		urlContent, err := fetchURLContent(urlStr)
		if err != nil {
			log.Printf("Failed to fetch URL %s: %v", urlStr, err)
			// Continue processing other URLs
			continue
		}
		
		// Add URL context to content
		enhancedContent.WriteString(fmt.Sprintf("\n\n[URL Context from %s]\n%s\n[End URL Context]", urlStr, urlContent))
	}
	
	return enhancedContent.String()
}

// ConvertToAnthropicMessages converts OpenAI messages to Anthropic format with cache control support
func (a *ClaudeAdapter) ConvertToAnthropicMessages(messages []Message) ([]anthropic.MessageParam, string) {
	return a.ConvertToAnthropicMessagesWithURLContext(messages, false)
}

// ConvertToAnthropicMessagesWithURLContext converts OpenAI messages to Anthropic format with URL context processing
func (a *ClaudeAdapter) ConvertToAnthropicMessagesWithURLContext(messages []Message, enableURLContext bool) ([]anthropic.MessageParam, string) {
	var anthropicMessages []anthropic.MessageParam
	var systemMessage string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemMessage = getContentAsString(msg.Content)
			if enableURLContext {
				systemMessage = a.processContentWithURLContext(systemMessage)
			}
		case "user":
			// Process URL context for user messages if enabled
			if enableURLContext {
				msg = a.processMessageURLContext(msg)
			}
			anthropicMessages = append(anthropicMessages, a.convertMessageToAnthropicWithCacheControl(msg, "user"))
		case "assistant":
			anthropicMessages = append(anthropicMessages, a.convertMessageToAnthropicWithCacheControl(msg, "assistant"))
		}
	}

	return anthropicMessages, systemMessage
}

// processMessageURLContext processes URLs in a message to add context
func (a *ClaudeAdapter) processMessageURLContext(msg Message) Message {
	// Process string content
	if content, ok := msg.Content.(string); ok {
		msg.Content = a.processContentWithURLContext(content)
		return msg
	}
	
	// Process structured content
	if contentParts, ok := msg.Content.([]ContentPart); ok {
		processedParts := make([]ContentPart, len(contentParts))
		for i, part := range contentParts {
			processedParts[i] = part
			if part.Type == "text" {
				processedParts[i].Text = a.processContentWithURLContext(part.Text)
			}
		}
		msg.Content = processedParts
	}
	
	return msg
}


// ConvertToAnthropicMessagesWithCacheControl converts OpenAI messages to Anthropic format with full cache control support
func (a *ClaudeAdapter) ConvertToAnthropicMessagesWithCacheControl(messages []Message) ([]anthropic.MessageParam, []anthropic.TextBlockParam, bool) {
	var anthropicMessages []anthropic.MessageParam
	var systemBlocks []anthropic.TextBlockParam
	var hasCacheControl bool

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// Handle system messages with cache control
			if contentParts, ok := msg.Content.([]ContentPart); ok {
				for _, part := range contentParts {
					if part.Type == "text" {
						textBlock := anthropic.TextBlockParam{
							Type: "text",
							Text: part.Text,
						}
						
						// Check for cache control
						if part.CacheControl != nil {
							hasCacheControl = true
							// Cache control would be handled through headers
						}
						
						systemBlocks = append(systemBlocks, textBlock)
					}
				}
			} else {
				// Simple string system message
				systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
					Type: "text",
					Text: getContentAsString(msg.Content),
				})
			}
		case "user":
			anthropicMessages = append(anthropicMessages, a.convertMessageToAnthropicWithCacheControl(msg, "user"))
		case "assistant":
			anthropicMessages = append(anthropicMessages, a.convertMessageToAnthropicWithCacheControl(msg, "assistant"))
		}
	}

	return anthropicMessages, systemBlocks, hasCacheControl
}

// convertMessageToAnthropicWithCacheControl converts a message to Anthropic format with cache control support
func (a *ClaudeAdapter) convertMessageToAnthropicWithCacheControl(msg Message, role string) anthropic.MessageParam {
	// Check if content is structured with cache control
	if contentParts, ok := msg.Content.([]ContentPart); ok {
		var blocks []anthropic.ContentBlockParamUnion
		
		for _, part := range contentParts {
			if part.Type == "text" {
				// Create text block with cache control support
				textBlock := anthropic.NewTextBlock(part.Text)
				
				// For now, we'll handle cache control at the header level
				// Future enhancement would add proper cache control support
				blocks = append(blocks, textBlock)
			}
		}
		
		if role == "user" {
			return anthropic.MessageParam{
				Role:    "user",
				Content: blocks,
			}
		} else {
			return anthropic.MessageParam{
				Role:    "assistant", 
				Content: blocks,
			}
		}
	}
	
	// Fallback to simple string content
	contentStr := getContentAsString(msg.Content)
	if role == "user" {
		return anthropic.NewUserMessage(anthropic.NewTextBlock(contentStr))
	} else {
		return anthropic.NewAssistantMessage(anthropic.NewTextBlock(contentStr))
	}
}

// HasCacheControl checks if any message content has cache control directives
func (a *ClaudeAdapter) HasCacheControl(messages []Message) bool {
	for _, msg := range messages {
		if contentParts, ok := msg.Content.([]ContentPart); ok {
			for _, part := range contentParts {
				if part.CacheControl != nil {
					return true
				}
			}
		}
	}
	return false
}

// ConvertAnthropicToOpenAI converts Anthropic response to OpenAI format with thinking/reasoning support
func (a *ClaudeAdapter) ConvertAnthropicToOpenAI(anthropicResp *anthropic.Message, model string) map[string]interface{} {
	return a.ConvertAnthropicToOpenAIWithBudget(anthropicResp, model, nil)
}

// ConvertAnthropicToOpenAIWithBudget converts Anthropic response to OpenAI format with budget token tracking
func (a *ClaudeAdapter) ConvertAnthropicToOpenAIWithBudget(anthropicResp *anthropic.Message, model string, budgetInfo map[string]interface{}) map[string]interface{} {
	// Extract content and thinking blocks separately
	var content string
	var reasoningContent string
	var thinkingBlocks []map[string]interface{}
	var thinkingTokens int

	for _, block := range anthropicResp.Content {
		if block.Type == "text" {
			content += block.Text
		} else if block.Type == "thinking" {
			reasoningContent = block.Text
			thinkingBlocks = append(thinkingBlocks, map[string]interface{}{
				"type":    "thinking",
				"content": block.Text,
			})
			// Estimate thinking tokens (rough approximation)
			thinkingTokens += len(block.Text) / 4
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

	// Build usage with budget tracking
	usage := map[string]interface{}{
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"total_tokens":      totalTokens,
	}

	// Add thinking token usage if applicable
	if thinkingTokens > 0 {
		usage["thinking_tokens"] = thinkingTokens
	}

	// Add budget information if provided
	if budgetInfo != nil {
		if budgetTokens, exists := budgetInfo["budget_tokens"]; exists {
			usage["budget_tokens"] = budgetTokens
			if budget, ok := budgetTokens.(int); ok {
				remaining := budget - totalTokens
				if remaining < 0 {
					remaining = 0
				}
				usage["budget_remaining"] = remaining
			}
		}
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
		"usage": usage,
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

	// Check if prompt caching is needed
	hasCacheControl := a.HasCacheControl(req.Messages) || a.HasToolCacheControl(req.Tools)
	
	// Check if URL context processing is enabled
	enableURLContext := req.URLContext != nil
	if enableURLContext {
		log.Printf("URL context processing enabled for request")
	}
	
	// Model name is already the actual model name (no prefix to remove)
	anthropicMessages, systemMessage := a.ConvertToAnthropicMessagesWithURLContext(req.Messages, enableURLContext)

	// Build the message params
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: 1000,
		Messages:  anthropicMessages,
	}

	// Set max tokens if provided
	if req.MaxTokens > 0 {
		params.MaxTokens = int64(req.MaxTokens)
	}

	// Handle thinking mode with budget tokens
	if req.Thinking != nil {
		log.Printf("Thinking mode detected in request")
		
		// Extract budget tokens if specified
		if budgetTokens, exists := req.Thinking["budget_tokens"]; exists {
			if budget, ok := budgetTokens.(int); ok && budget > 0 {
				log.Printf("Setting thinking mode budget tokens: %d", budget)
				// For now, we'll use this as a hint for max tokens
				// Future enhancement would include proper budget token tracking
				budgetInt64 := int64(budget)
				if budgetInt64 < params.MaxTokens {
					params.MaxTokens = budgetInt64
				}
			}
		}
	}

	if systemMessage != "" {
		params.System = []anthropic.TextBlockParam{
			{
				Text: systemMessage,
			},
		}
	}

	// Add cache control headers if needed
	if hasCacheControl {
		log.Printf("Cache control detected, adding prompt caching headers")
		// For now, we'll handle this at the HTTP request level
		// Future enhancement would include proper cache control in the SDK
	}

	log.Printf("Calling Anthropic API (%s) with %d messages", req.Model, len(anthropicMessages))

	// Call Anthropic API (works for both direct and Bedrock)
	result, err := client.Messages.New(context.Background(), params)
	if err != nil {
		log.Printf("Anthropic API error for model %s: %v", req.Model, err)
		return nil, err
	}

	log.Printf("Anthropic API response received, content blocks: %d", len(result.Content))

	// Convert to OpenAI format with budget tracking
	var response map[string]interface{}
	if req.Thinking != nil {
		response = a.ConvertAnthropicToOpenAIWithBudget(result, req.Model, req.Thinking)
	} else {
		response = a.ConvertAnthropicToOpenAI(result, req.Model)
	}

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

	// Check if prompt caching is needed
	hasCacheControl := a.HasCacheControl(req.Messages) || a.HasToolCacheControl(req.Tools)
	
	// Check if URL context processing is enabled
	enableURLContext := req.URLContext != nil
	if enableURLContext {
		log.Printf("URL context processing enabled for streaming request")
	}
	
	// Model name is already the actual model name (no prefix to remove)
	anthropicMessages, systemMessage := a.ConvertToAnthropicMessagesWithURLContext(req.Messages, enableURLContext)

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

	// Add cache control headers if needed
	if hasCacheControl {
		log.Printf("Cache control detected for streaming request, adding prompt caching headers")
		// For now, we'll handle this at the HTTP request level
		// Future enhancement would include proper cache control in the SDK
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

// HasToolCacheControl checks if any tool has cache control directives
func (a *ClaudeAdapter) HasToolCacheControl(tools []Tool) bool {
	for _, tool := range tools {
		if tool.Type == "function" {
			// Check if there's cache control in the function definition
			if tool.Function.Parameters != nil {
				if _, exists := tool.Function.Parameters["cache_control"]; exists {
					return true
				}
			}
		}
	}
	return false
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
