package adapter

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"google.golang.org/genai"
)

// sanitizeError removes API keys from error messages for secure logging
func sanitizeError(err error) string {
	if err == nil {
		return ""
	}

	errStr := err.Error()
	if apiKey := os.Getenv("GEMINI_API_KEY"); apiKey != "" {
		errStr = strings.ReplaceAll(errStr, "&key="+apiKey, "&key=***")
		errStr = strings.ReplaceAll(errStr, "key="+apiKey, "key=***")
	}
	return errStr
}

// GeminiAdapter handles conversion between OpenAI and Gemini formats
type GeminiAdapter struct {
	client *genai.Client
}

// NewGeminiAdapter creates a new Gemini adapter
func NewGeminiAdapter(client *genai.Client) *GeminiAdapter {
	return &GeminiAdapter{
		client: client,
	}
}

// IsGeminiModel checks if a model name belongs to Gemini
func (a *GeminiAdapter) IsGeminiModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "gemini") ||
		strings.HasPrefix(model, "models/gemini")
}

// convertContentToGeminiParts converts OpenAI content to Gemini parts
func (a *GeminiAdapter) convertContentToGeminiParts(content interface{}) ([]*genai.Part, error) {
	switch c := content.(type) {
	case string:
		return []*genai.Part{genai.NewPartFromText(c)}, nil
	case []ContentPart:
		var parts []*genai.Part
		for _, part := range c {
			switch part.Type {
			case "text":
				parts = append(parts, genai.NewPartFromText(part.Text))
			case "image_url":
				if part.ImageURL != nil {
					// Handle base64 encoded images
					if strings.HasPrefix(part.ImageURL.URL, "data:") {
						// Parse data URL
						imagePart, err := a.convertDataURLToGeminiPart(part.ImageURL.URL)
						if err != nil {
							parts = append(parts, genai.NewPartFromText("Error processing image"))
						} else {
							parts = append(parts, imagePart)
						}
					} else {
						// Handle regular URLs - in real implementation you'd fetch and convert
						parts = append(parts, genai.NewPartFromText(fmt.Sprintf("Image URL: %s", part.ImageURL.URL)))
					}
				}
			}
		}
		return parts, nil
	default:
		return []*genai.Part{genai.NewPartFromText(fmt.Sprintf("%v", content))}, nil
	}
}

// convertDataURLToGeminiPart converts a data URL to a Gemini part
func (a *GeminiAdapter) convertDataURLToGeminiPart(dataURL string) (*genai.Part, error) {
	// Parse data URL format: data:image/jpeg;base64,{base64data}
	re := regexp.MustCompile(`^data:([^;]+);base64,(.+)$`)
	matches := re.FindStringSubmatch(dataURL)

	if len(matches) == 3 {
		mimeType := matches[1]
		base64Data := matches[2]

		// Decode base64
		data, err := base64.StdEncoding.DecodeString(base64Data)
		if err != nil {
			return nil, fmt.Errorf("failed to decode base64 data: %v", err)
		}

		// Create image part
		return genai.NewPartFromBytes(data, mimeType), nil
	}

	return nil, fmt.Errorf("invalid data URL format")
}

// ConvertToGeminiMessages converts OpenAI messages to Gemini format with multi-modal support
func (a *GeminiAdapter) ConvertToGeminiMessages(messages []Message) ([]*genai.Content, error) {
	return a.ConvertToGeminiMessagesWithURLContext(messages, false)
}

// ConvertToGeminiMessagesWithURLContext converts OpenAI messages to Gemini format with URL context processing
func (a *GeminiAdapter) ConvertToGeminiMessagesWithURLContext(messages []Message, enableURLContext bool) ([]*genai.Content, error) {
	var geminiMessages []*genai.Content
	var systemMessages []string

	// First pass: collect system messages
	for _, msg := range messages {
		if msg.Role == "system" {
			contentStr := getContentAsString(msg.Content)
			if enableURLContext {
				// Use Claude adapter's URL processing (shared function)
				claudeAdapter := &ClaudeAdapter{}
				contentStr = claudeAdapter.processContentWithURLContext(contentStr)
			}
			systemMessages = append(systemMessages, contentStr)
		}
	}

	// Second pass: convert non-system messages
	for _, msg := range messages {
		if msg.Role == "system" {
			continue // Skip system messages, they're handled separately
		}

		var role genai.Role
		switch msg.Role {
		case "user":
			role = genai.RoleUser
		case "assistant":
			role = genai.RoleModel
		case "tool":
			role = genai.RoleUser // Tool responses are treated as user messages
		default:
			role = genai.RoleUser
		}

		// Process URL context for user messages if enabled
		if enableURLContext && msg.Role == "user" {
			claudeAdapter := &ClaudeAdapter{}
			msg = claudeAdapter.processMessageURLContext(msg)
		}

		parts, err := a.convertContentToGeminiParts(msg.Content)
		if err != nil {
			return nil, err
		}

		// Add tool call information if present
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				toolText := fmt.Sprintf("Tool Call: %s(%s)", toolCall.Function.Name, toolCall.Function.Arguments)
				parts = append(parts, genai.NewPartFromText(toolText))
			}
		}

		geminiMessages = append(geminiMessages, genai.NewContentFromParts(parts, role))
	}

	// Prepend system messages to first user message
	if len(systemMessages) > 0 && len(geminiMessages) > 0 {
		for i, msg := range geminiMessages {
			if msg.Role == genai.RoleUser {
				systemPrompt := strings.Join(systemMessages, "\n\n")

				// Get the user content
				var userContent string
				if len(msg.Parts) > 0 {
					if msg.Parts[0].Text != "" {
						userContent = msg.Parts[0].Text
					}
				}

				// Create single combined part
				combinedContent := fmt.Sprintf("System: %s\n\n%s", systemPrompt, userContent)
				geminiMessages[i] = genai.NewContentFromText(combinedContent, genai.RoleUser)
				break
			}
		}
	}

	return geminiMessages, nil
}

// convertOpenAIToolsToGemini converts OpenAI tools to Gemini function declarations
func (a *GeminiAdapter) convertOpenAIToolsToGemini(tools []Tool) []*genai.Tool {
	var geminiTools []*genai.Tool

	for _, tool := range tools {
		if tool.Type == "function" {
			// Convert OpenAI parameters to Gemini schema
			geminiSchema := &genai.Schema{
				Type: genai.TypeObject,
			}

			// Convert parameters if they exist
			if tool.Function.Parameters != nil {
				if properties, ok := tool.Function.Parameters["properties"].(map[string]interface{}); ok {
					geminiSchema.Properties = make(map[string]*genai.Schema)
					for propName, propSchema := range properties {
						if propMap, ok := propSchema.(map[string]interface{}); ok {
							geminiProp := &genai.Schema{}
							if propType, ok := propMap["type"].(string); ok {
								switch propType {
								case "string":
									geminiProp.Type = genai.TypeString
								case "integer":
									geminiProp.Type = genai.TypeInteger
								case "number":
									geminiProp.Type = genai.TypeNumber
								case "boolean":
									geminiProp.Type = genai.TypeBoolean
								case "array":
									geminiProp.Type = genai.TypeArray
								case "object":
									geminiProp.Type = genai.TypeObject
								}
							}
							if description, ok := propMap["description"].(string); ok {
								geminiProp.Description = description
							}
							geminiSchema.Properties[propName] = geminiProp
						}
					}
				}

				// Handle required fields
				if required, ok := tool.Function.Parameters["required"].([]interface{}); ok {
					geminiSchema.Required = make([]string, len(required))
					for i, req := range required {
						if reqStr, ok := req.(string); ok {
							geminiSchema.Required[i] = reqStr
						}
					}
				} else if required, ok := tool.Function.Parameters["required"].([]string); ok {
					geminiSchema.Required = required
				}
			}

			// Create function declaration
			funcDecl := &genai.FunctionDeclaration{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  geminiSchema,
			}

			// Create tool
			geminiTool := &genai.Tool{
				FunctionDeclarations: []*genai.FunctionDeclaration{funcDecl},
			}

			geminiTools = append(geminiTools, geminiTool)
		}
	}

	return geminiTools
}

// generateToolCallID generates a unique ID for a tool call
func generateToolCallID() string {
	return "call_" + strconv.FormatInt(time.Now().UnixNano(), 36)
}

// parseToolCalls extracts tool calls from Gemini response parts
func (a *GeminiAdapter) parseToolCalls(parts []*genai.Part) []ToolCall {
	var toolCalls []ToolCall

	for _, part := range parts {
		if part.FunctionCall != nil {
			// Convert function call arguments to JSON string
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				argsJSON = []byte("{}")
			}

			toolCall := ToolCall{
				ID:   generateToolCallID(),
				Type: "function",
				Function: FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}

	return toolCalls
}

// ConvertGeminiToOpenAI converts Gemini response to OpenAI format with tool call support
func (a *GeminiAdapter) ConvertGeminiToOpenAI(geminiResp *genai.GenerateContentResponse, model string) map[string]interface{} {
	var content string
	var toolCalls []ToolCall
	finishReason := "stop"

	if len(geminiResp.Candidates) > 0 {
		candidate := geminiResp.Candidates[0]

		// Extract content and tool calls from parts
		var textParts []string
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		}
		content = strings.Join(textParts, "")

		// Extract tool calls
		toolCalls = a.parseToolCalls(candidate.Content.Parts)

		// Set finish reason based on tool calls
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		}

		// Handle other finish reasons
		switch candidate.FinishReason {
		case genai.FinishReasonSafety:
			finishReason = "content_filter"
		case genai.FinishReasonMaxTokens:
			finishReason = "stop" // LiteLLM expects "stop" for max tokens
		case genai.FinishReasonRecitation:
			finishReason = "stop" // For non-streaming, use "stop" as fallback
		}
	}

	// Extract token usage from Gemini response
	var promptTokens, completionTokens, totalTokens int
	if geminiResp.UsageMetadata != nil {
		promptTokens = int(geminiResp.UsageMetadata.PromptTokenCount)
		completionTokens = int(geminiResp.UsageMetadata.CandidatesTokenCount)
		totalTokens = int(geminiResp.UsageMetadata.TotalTokenCount)
	}

	// Build message - use different format depending on whether there are tool calls
	var message interface{}

	if len(toolCalls) > 0 {
		// Complex message with tool calls
		message = map[string]interface{}{
			"role":       "assistant",
			"content":    nil, // Content is nil when there are tool calls
			"tool_calls": toolCalls,
		}
	} else {
		// Simple message format for backward compatibility
		message = map[string]string{
			"role":    "assistant",
			"content": content,
		}
	}

	return map[string]interface{}{
		"id":      "chatcmpl-" + strings.ReplaceAll(strings.ToLower(model), "/", "-"),
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

// ConvertErrorToOpenAIStream converts an error to OpenAI streaming format
func (a *GeminiAdapter) ConvertErrorToOpenAIStream(err error, model string) map[string]interface{} {
	return map[string]interface{}{
		"id":      "chatcmpl-error-" + strings.ReplaceAll(strings.ToLower(model), "/", "-"),
		"object":  "chat.completion.chunk",
		"created": 1699999999,
		"model":   model,
		"error": map[string]interface{}{
			"message": err.Error(),
			"type":    "invalid_request_error",
			"code":    "model_error",
		},
	}
}

// ConvertErrorToOpenAI converts an error to OpenAI non-streaming format
func (a *GeminiAdapter) ConvertErrorToOpenAI(err error, model string) map[string]interface{} {
	return map[string]interface{}{
		"error": map[string]interface{}{
			"message": err.Error(),
			"type":    "invalid_request_error",
			"code":    "model_error",
		},
	}
}

// ConvertGeminiStreamToOpenAI converts Gemini stream chunks to OpenAI format
func (a *GeminiAdapter) ConvertGeminiStreamToOpenAI(resp *genai.GenerateContentResponse, model string, isFirst bool) map[string]interface{} {
	openaiChunk := map[string]interface{}{
		"id":      "chatcmpl-" + strings.ReplaceAll(strings.ToLower(model), "/", "-"),
		"object":  "chat.completion.chunk",
		"created": 1699999999,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": nil,
			},
		},
	}

	choice := openaiChunk["choices"].([]map[string]interface{})[0]
	delta := choice["delta"].(map[string]interface{})

	// Add role for first chunk
	if isFirst {
		delta["role"] = "assistant"
	}

	// Extract content and tool calls from Gemini response
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		candidate := resp.Candidates[0]

		// Handle text content
		var textContent string
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				textContent += part.Text
			}
		}

		if textContent != "" {
			delta["content"] = textContent
		}

		// Handle tool calls
		toolCalls := a.parseToolCalls(candidate.Content.Parts)
		if len(toolCalls) > 0 {
			delta["tool_calls"] = toolCalls
			delta["content"] = nil // Content is nil when there are tool calls
		}
	}

	// Check for finish reason
	if len(resp.Candidates) > 0 {
		candidate := resp.Candidates[0]

		// Check if there are tool calls to set appropriate finish reason
		if len(a.parseToolCalls(candidate.Content.Parts)) > 0 {
			choice["finish_reason"] = "tool_calls"
		} else {
			switch candidate.FinishReason {
			case genai.FinishReasonStop:
				choice["finish_reason"] = "stop"
			case genai.FinishReasonMaxTokens:
				choice["finish_reason"] = "stop" // LiteLLM expects "stop" for max tokens
			case genai.FinishReasonSafety:
				choice["finish_reason"] = "content_filter"
			case genai.FinishReasonRecitation:
				// LiteLLM expects nil for recitation
				choice["finish_reason"] = nil
			}
		}
	}

	return openaiChunk
}

// HandleRequest processes requests for Gemini models
func (a *GeminiAdapter) HandleRequest(req ChatRequest) (map[string]interface{}, error) {
	log.Printf("Processing non-streaming request for Gemini model: %s", req.Model)
	log.Printf("Request messages: %+v", req.Messages)

	// Get the model - ensure we're using the correct model name format
	modelName := req.Model
	if !strings.HasPrefix(modelName, "models/") {
		modelName = "models/" + modelName
	}
	log.Printf("Using Gemini model ID: %s", modelName)

	// Create generate content config
	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr(float32(0.7)),
		TopK:            genai.Ptr(float32(40)),
		TopP:            genai.Ptr(float32(0.9)),
		MaxOutputTokens: 2048,
	}

	// Configure safety settings to be more permissive
	config.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
	}

	// Configure tools if provided
	if len(req.Tools) > 0 {
		geminiTools := a.convertOpenAIToolsToGemini(req.Tools)
		config.Tools = geminiTools
		log.Printf("Configured %d tools for Gemini model", len(geminiTools))
	}

	log.Printf("Model configuration: %+v", config)

	// Check if URL context processing is enabled
	enableURLContext := req.URLContext != nil
	if enableURLContext {
		log.Printf("URL context processing enabled for Gemini request")
	}

	// Convert messages
	geminiMessages, err := a.ConvertToGeminiMessagesWithURLContext(req.Messages, enableURLContext)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %v", err)
	}

	log.Printf("Converted to %d Gemini messages", len(geminiMessages))

	// Handle case where we have no messages
	if len(geminiMessages) == 0 {
		return nil, fmt.Errorf("no valid messages to process")
	}

	log.Printf("Using generate content with %d messages", len(geminiMessages))
	resp, err := a.client.Models.GenerateContent(context.Background(), modelName, geminiMessages, config)

	if err != nil {
		log.Printf("Gemini error for model %s: %s", req.Model, sanitizeError(err))
		return nil, fmt.Errorf("gemini API error: %s", sanitizeError(err))
	}

	// Convert to OpenAI format
	return a.ConvertGeminiToOpenAI(resp, req.Model), nil
}

// HandleStreamingRequest processes streaming requests for Gemini models
func (a *GeminiAdapter) HandleStreamingRequest(req ChatRequest, w http.ResponseWriter) error {
	log.Printf("Processing streaming request for Gemini model: %s", req.Model)
	log.Printf("Request messages: %+v", req.Messages)

	// Get the model - ensure we're using the correct model name format
	modelName := req.Model
	if !strings.HasPrefix(modelName, "models/") {
		modelName = "models/" + modelName
	}
	log.Printf("Using Gemini model ID: %s", modelName)

	// Create generate content config
	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr(float32(0.7)),
		TopK:            genai.Ptr(float32(40)),
		TopP:            genai.Ptr(float32(0.9)),
		MaxOutputTokens: 2048,
	}

	// Configure tools if provided
	if len(req.Tools) > 0 {
		geminiTools := a.convertOpenAIToolsToGemini(req.Tools)
		config.Tools = geminiTools
		log.Printf("Configured %d tools for streaming Gemini model", len(geminiTools))
	}

	// Configure safety settings to be more permissive
	config.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
		},
	}

	log.Printf("Model configuration: %+v", config)

	// Check if URL context processing is enabled
	enableURLContext := req.URLContext != nil
	if enableURLContext {
		log.Printf("URL context processing enabled for Gemini streaming request")
	}

	// Convert messages
	geminiMessages, err := a.ConvertToGeminiMessagesWithURLContext(req.Messages, enableURLContext)
	if err != nil {
		return fmt.Errorf("failed to convert messages: %v", err)
	}

	log.Printf("Converted to %d Gemini messages", len(geminiMessages))

	// Handle case where we have no messages
	if len(geminiMessages) == 0 {
		return fmt.Errorf("no valid messages to process")
	}

	log.Printf("Using streaming generation with %d messages", len(geminiMessages))

	// Use the new streaming API
	stream := a.client.Models.GenerateContentStream(context.Background(), modelName, geminiMessages, config)

	isFirst := true
	for resp, err := range stream {
		if err != nil {
			log.Printf("Gemini streaming error for model %s: %s", req.Model, sanitizeError(err))
			return fmt.Errorf("gemini API error: %s", sanitizeError(err))
		}

		// Convert to OpenAI format
		openaiChunk := a.ConvertGeminiStreamToOpenAI(resp, req.Model, isFirst)
		isFirst = false

		// Send as SSE
		data, _ := json.Marshal(openaiChunk)
		w.Write([]byte("data: " + string(data) + "\n\n"))

		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	// Send completion marker
	w.Write([]byte("data: [DONE]\n\n"))

	return nil
}
