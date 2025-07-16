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

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/iterator"
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
func (a *GeminiAdapter) convertContentToGeminiParts(content interface{}) ([]genai.Part, error) {
	switch c := content.(type) {
	case string:
		return []genai.Part{genai.Text(c)}, nil
	case []ContentPart:
		var parts []genai.Part
		for _, part := range c {
			switch part.Type {
			case "text":
				parts = append(parts, genai.Text(part.Text))
			case "image_url":
				if part.ImageURL != nil {
					// Handle base64 encoded images
					if strings.HasPrefix(part.ImageURL.URL, "data:") {
						// Parse data URL
						parts = append(parts, a.convertDataURLToGeminiPart(part.ImageURL.URL))
					} else {
						// Handle regular URLs - in real implementation you'd fetch and convert
						parts = append(parts, genai.Text(fmt.Sprintf("Image URL: %s", part.ImageURL.URL)))
					}
				}
			}
		}
		return parts, nil
	default:
		return []genai.Part{genai.Text(fmt.Sprintf("%v", content))}, nil
	}
}

// convertDataURLToGeminiPart converts a data URL to a Gemini part
func (a *GeminiAdapter) convertDataURLToGeminiPart(dataURL string) genai.Part {
	// Parse data URL format: data:image/jpeg;base64,{base64data}
	re := regexp.MustCompile(`^data:([^;]+);base64,(.+)$`)
	matches := re.FindStringSubmatch(dataURL)
	
	if len(matches) == 3 {
		mimeType := matches[1]
		base64Data := matches[2]
		
		// Decode base64
		data, err := base64.StdEncoding.DecodeString(base64Data)
		if err != nil {
			return genai.Text("Error decoding image data")
		}
		
		// Create image part
		return genai.ImageData(mimeType, data)
	}
	
	return genai.Text("Invalid image data")
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

		var role string
		switch msg.Role {
		case "user":
			role = "user"
		case "assistant":
			role = "model"
		case "tool":
			role = "user" // Tool responses are treated as user messages
		default:
			role = "user"
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
				parts = append(parts, genai.Text(toolText))
			}
		}

		geminiMessages = append(geminiMessages, &genai.Content{
			Parts: parts,
			Role:  role,
		})
	}

	// Prepend system messages to first user message
	if len(systemMessages) > 0 && len(geminiMessages) > 0 {
		for i, msg := range geminiMessages {
			if msg.Role == "user" {
				systemPrompt := strings.Join(systemMessages, "\n\n")
				
				// Get the user content
				var userContent string
				if len(msg.Parts) > 0 {
					if text, ok := msg.Parts[0].(genai.Text); ok {
						userContent = string(text)
					}
				}
				
				// Create single combined part
				combinedContent := fmt.Sprintf("System: %s\n\n%s", systemPrompt, userContent)
				geminiMessages[i].Parts = []genai.Part{genai.Text(combinedContent)}
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
func (a *GeminiAdapter) parseToolCalls(parts []genai.Part) []ToolCall {
	var toolCalls []ToolCall
	
	for _, part := range parts {
		if fc, ok := part.(genai.FunctionCall); ok {
			// Convert function call arguments to JSON string
			argsJSON, err := json.Marshal(fc.Args)
			if err != nil {
				argsJSON = []byte("{}")
			}
			
			toolCall := ToolCall{
				ID:   generateToolCallID(),
				Type: "function",
				Function: FunctionCall{
					Name:      fc.Name,
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
			if text, ok := part.(genai.Text); ok {
				textParts = append(textParts, string(text))
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
			if text, ok := part.(genai.Text); ok {
				textContent += string(text)
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

	model := a.client.GenerativeModel(modelName)

	// Configure model settings
	model.GenerationConfig = genai.GenerationConfig{
		Temperature:     genai.Ptr(float32(0.7)),
		TopK:            genai.Ptr(int32(40)),
		TopP:            genai.Ptr(float32(0.9)),
		MaxOutputTokens: genai.Ptr(int32(2048)),
	}

	// Configure safety settings to be more permissive
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
	}

	// Configure tools if provided
	if len(req.Tools) > 0 {
		geminiTools := a.convertOpenAIToolsToGemini(req.Tools)
		model.Tools = geminiTools
		log.Printf("Configured %d tools for Gemini model", len(geminiTools))
	}

	log.Printf("Model configuration - Generation: %+v, Safety: %+v", model.GenerationConfig, model.SafetySettings)

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

	var resp *genai.GenerateContentResponse

	if len(geminiMessages) == 1 {
		// Single message, use direct generation instead of chat
		var prompt string
		if len(geminiMessages[0].Parts) > 0 {
			if text, ok := geminiMessages[0].Parts[0].(genai.Text); ok {
				prompt = string(text)
			}
		}

		if prompt == "" {
			return nil, fmt.Errorf("empty prompt for single message request")
		}

		log.Printf("Using direct generation for single message")
		log.Printf("Prompt length: %d characters", len(prompt))
		resp, err = model.GenerateContent(context.Background(), genai.Text(prompt))
	} else {
		// Multiple messages, use chat session
		cs := model.StartChat()
		cs.History = geminiMessages[:len(geminiMessages)-1] // All but last message as history

		// Get the last message content
		var prompt string
		if len(geminiMessages) > 0 {
			lastMsg := geminiMessages[len(geminiMessages)-1]
			if len(lastMsg.Parts) > 0 {
				if text, ok := lastMsg.Parts[0].(genai.Text); ok {
					prompt = string(text)
				}
			}
		}

		if prompt == "" {
			return nil, fmt.Errorf("empty prompt for chat session request")
		}

		log.Printf("Using chat session")
		log.Printf("Prompt length: %d characters", len(prompt))
		resp, err = cs.SendMessage(context.Background(), genai.Text(prompt))
	}

	if err != nil {
		log.Printf("Gemini error for model %s: %s", req.Model, sanitizeError(err))
		if gerr, ok := err.(*googleapi.Error); ok {
			log.Printf("Google API Error - Code: %d, Message: %s", gerr.Code, gerr.Message)
		}
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

	model := a.client.GenerativeModel(modelName)

	// Configure model settings
	model.GenerationConfig = genai.GenerationConfig{
		Temperature:     genai.Ptr(float32(0.7)),
		TopK:            genai.Ptr(int32(40)),
		TopP:            genai.Ptr(float32(0.9)),
		MaxOutputTokens: genai.Ptr(int32(2048)),
	}

	// Configure tools if provided
	if len(req.Tools) > 0 {
		geminiTools := a.convertOpenAIToolsToGemini(req.Tools)
		model.Tools = geminiTools
		log.Printf("Configured %d tools for streaming Gemini model", len(geminiTools))
	}

	// Configure safety settings to be more permissive
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockMediumAndAbove,
		},
	}

	log.Printf("Model configuration - Generation: %+v, Safety: %+v", model.GenerationConfig, model.SafetySettings)

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

	// Handle case where we have no messages or only one message
	if len(geminiMessages) == 0 {
		return fmt.Errorf("no valid messages to process")
	}

	if len(geminiMessages) == 1 {
		// Single message, use direct generation instead of chat
		var prompt string
		if len(geminiMessages[0].Parts) > 0 {
			if text, ok := geminiMessages[0].Parts[0].(genai.Text); ok {
				prompt = string(text)
			}
		}

		if prompt == "" {
			return fmt.Errorf("empty prompt for single message request")
		}

		log.Printf("Using direct generation for single message: %s", prompt)
		log.Printf("Prompt length: %d characters", len(prompt))

		// Try a more explicit approach with Content structure
		content := &genai.Content{
			Parts: []genai.Part{genai.Text(prompt)},
			Role:  "user",
		}

		log.Printf("Creating stream with content: %+v", content)
		iter := model.GenerateContentStream(context.Background(), content.Parts...)

		isFirst := true
		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				log.Printf("Gemini streaming error for model %s: %s", req.Model, sanitizeError(err))

				if gerr, ok := err.(*googleapi.Error); ok {
					log.Printf("Google API Error - Code: %d, Message: '%s'", gerr.Code, gerr.Message)
				}

				// Already logged with sanitizeError above

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
	} else {
		// Multiple messages, use chat session
		cs := model.StartChat()
		cs.History = geminiMessages[:len(geminiMessages)-1] // All but last message as history

		// Get the last message content
		var prompt string
		if len(geminiMessages) > 0 {
			lastMsg := geminiMessages[len(geminiMessages)-1]
			if len(lastMsg.Parts) > 0 {
				if text, ok := lastMsg.Parts[0].(genai.Text); ok {
					prompt = string(text)
				}
			}
		}

		if prompt == "" {
			return fmt.Errorf("empty prompt for chat session request")
		}

		log.Printf("Using chat session with prompt: %s", prompt)
		log.Printf("Prompt length: %d characters", len(prompt))
		iter := cs.SendMessageStream(context.Background(), genai.Text(prompt))

		isFirst := true
		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				break
			}
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
	}

	// Send completion marker
	w.Write([]byte("data: [DONE]\n\n"))

	return nil
}
