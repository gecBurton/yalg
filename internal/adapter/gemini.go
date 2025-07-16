package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

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

// ConvertToGeminiMessages converts OpenAI messages to Gemini format
func (a *GeminiAdapter) ConvertToGeminiMessages(messages []Message) ([]*genai.Content, error) {
	var geminiMessages []*genai.Content

	for _, msg := range messages {
		var role string
		switch msg.Role {
		case "user":
			role = "user"
		case "assistant":
			role = "model"
		case "system":
			// Gemini doesn't have a system role, prepend to first user message
			if len(geminiMessages) == 0 {
				// If this is the first message and it's system, we'll handle it specially
				continue
			}
			// Skip system messages that aren't first
			continue
		default:
			role = "user"
		}

		geminiMessages = append(geminiMessages, &genai.Content{
			Parts: []genai.Part{genai.Text(msg.Content)},
			Role:  role,
		})
	}

	// Handle system message by prepending to first user message
	for i, msg := range messages {
		if msg.Role == "system" && len(geminiMessages) > 0 {
			// Find first user message and prepend system content
			for j := i + 1; j < len(messages); j++ {
				if messages[j].Role == "user" {
					systemPrompt := fmt.Sprintf("System: %s\n\nUser: %s", msg.Content, messages[j].Content)
					// Update the corresponding gemini message
					for k, gMsg := range geminiMessages {
						if gMsg.Role == "user" {
							geminiMessages[k].Parts = []genai.Part{genai.Text(systemPrompt)}
							break
						}
					}
					break
				}
			}
			break
		}
	}

	return geminiMessages, nil
}

// ConvertGeminiToOpenAI converts Gemini response to OpenAI format
func (a *GeminiAdapter) ConvertGeminiToOpenAI(geminiResp *genai.GenerateContentResponse, model string) map[string]interface{} {
	content := ""
	if len(geminiResp.Candidates) > 0 && len(geminiResp.Candidates[0].Content.Parts) > 0 {
		if text, ok := geminiResp.Candidates[0].Content.Parts[0].(genai.Text); ok {
			content = string(text)
		}
	}

	// Extract token usage from Gemini response
	var promptTokens, completionTokens, totalTokens int
	if geminiResp.UsageMetadata != nil {
		promptTokens = int(geminiResp.UsageMetadata.PromptTokenCount)
		completionTokens = int(geminiResp.UsageMetadata.CandidatesTokenCount)
		totalTokens = int(geminiResp.UsageMetadata.TotalTokenCount)
	}

	return map[string]interface{}{
		"id":      "chatcmpl-" + strings.ReplaceAll(strings.ToLower(model), "/", "-"),
		"object":  "chat.completion",
		"created": 1699999999,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": content,
				},
				"finish_reason": "stop",
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

	// Extract content from Gemini response
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		if text, ok := resp.Candidates[0].Content.Parts[0].(genai.Text); ok {
			content := string(text)
			if content != "" {
				delta["content"] = content
			}
		}
	}

	// Check for finish reason
	if len(resp.Candidates) > 0 {
		switch resp.Candidates[0].FinishReason {
		case genai.FinishReasonStop, genai.FinishReasonMaxTokens:
			choice["finish_reason"] = "stop"
		case genai.FinishReasonSafety:
			choice["finish_reason"] = "content_filter"
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

	log.Printf("Model configuration - Generation: %+v, Safety: %+v", model.GenerationConfig, model.SafetySettings)

	// Convert messages
	geminiMessages, err := a.ConvertToGeminiMessages(req.Messages)
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

	// Convert messages
	geminiMessages, err := a.ConvertToGeminiMessages(req.Messages)
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
