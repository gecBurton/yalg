package adapter

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
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

// AnthropicMessage represents a message in Anthropic format
type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ClaudeAdapter handles conversion between OpenAI and Claude formats
type ClaudeAdapter struct {
	bedrockClient *bedrockruntime.Client
}

// NewClaudeAdapter creates a new Claude adapter
func NewClaudeAdapter(bedrockClient *bedrockruntime.Client) *ClaudeAdapter {
	return &ClaudeAdapter{
		bedrockClient: bedrockClient,
	}
}

// IsAnthropicModel checks if a model name belongs to Anthropic
func (a *ClaudeAdapter) IsAnthropicModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "claude") ||
		strings.HasPrefix(model, "anthropic.")
}

// ConvertToAnthropicMessages converts OpenAI messages to Anthropic format
func (a *ClaudeAdapter) ConvertToAnthropicMessages(messages []Message) ([]AnthropicMessage, string) {
	var anthropicMessages []AnthropicMessage
	var systemMessage string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemMessage = msg.Content
		case "user", "assistant":
			anthropicMessages = append(anthropicMessages, AnthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}
	return anthropicMessages, systemMessage
}

// ConvertAnthropicToOpenAI converts Anthropic response to OpenAI format
func (a *ClaudeAdapter) ConvertAnthropicToOpenAI(anthropicResp map[string]interface{}, model string) map[string]interface{} {
	content := ""
	if contentArray, ok := anthropicResp["content"].([]interface{}); ok && len(contentArray) > 0 {
		if contentItem, ok := contentArray[0].(map[string]interface{}); ok {
			if text, ok := contentItem["text"].(string); ok {
				content = text
			}
		}
	}

	// Extract token usage from Anthropic response
	var promptTokens, completionTokens, totalTokens int
	if usage, ok := anthropicResp["usage"].(map[string]interface{}); ok {
		if inputTokens, ok := usage["input_tokens"].(float64); ok {
			promptTokens = int(inputTokens)
		}
		if outputTokens, ok := usage["output_tokens"].(float64); ok {
			completionTokens = int(outputTokens)
		}
		totalTokens = promptTokens + completionTokens
	}

	return map[string]interface{}{
		"id":      "chatcmpl-" + strings.ReplaceAll(strings.ToLower(model), ".", "-"),
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

// BuildAnthropicPayload creates the request payload for Anthropic models
func (a *ClaudeAdapter) BuildAnthropicPayload(messages []AnthropicMessage, systemMessage string) map[string]interface{} {
	payload := map[string]interface{}{
		"anthropic_version": "bedrock-2023-05-31",
		"max_tokens":        1000,
		"messages":          messages,
	}

	if systemMessage != "" {
		payload["system"] = systemMessage
	}

	return payload
}

// BuildAnthropicStreamingPayload creates the request payload for streaming Anthropic models
func (a *ClaudeAdapter) BuildAnthropicStreamingPayload(messages []AnthropicMessage, systemMessage string) map[string]interface{} {
	// For Bedrock streaming, we use the same payload as non-streaming
	// The streaming behavior is controlled by the API call type (InvokeModelWithResponseStream vs InvokeModel)
	return a.BuildAnthropicPayload(messages, systemMessage)
}

// ConvertErrorToOpenAIStream converts an error to OpenAI streaming format  
func (a *ClaudeAdapter) ConvertErrorToOpenAIStream(err error, model string) map[string]interface{} {
	return map[string]interface{}{
		"id":      "chatcmpl-error-" + strings.ReplaceAll(strings.ToLower(model), ".", "-"),
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
func (a *ClaudeAdapter) ConvertErrorToOpenAI(err error, model string) map[string]interface{} {
	return map[string]interface{}{
		"error": map[string]interface{}{
			"message": err.Error(),
			"type":    "invalid_request_error",
			"code":    "model_error",
		},
	}
}

// ConvertAnthropicStreamToOpenAI converts Anthropic stream chunks to OpenAI format
func (a *ClaudeAdapter) ConvertAnthropicStreamToOpenAI(chunk map[string]interface{}, model string) map[string]interface{} {
	openaiChunk := map[string]interface{}{
		"id":      "chatcmpl-" + strings.ReplaceAll(strings.ToLower(model), ".", "-"),
		"object":  "chat.completion.chunk",
		"created": 1699999999,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{},
				"finish_reason": nil,
			},
		},
	}

	// Handle different types of Anthropic streaming events
	if eventType, ok := chunk["type"].(string); ok {
		choice := openaiChunk["choices"].([]map[string]interface{})[0]
		delta := choice["delta"].(map[string]interface{})

		switch eventType {
		case "message_start":
			// Start of message - add role
			delta["role"] = "assistant"
			
		case "content_block_delta":
			// Content chunk - extract text
			if deltaData, ok := chunk["delta"].(map[string]interface{}); ok {
				if text, ok := deltaData["text"].(string); ok {
					delta["content"] = text
				}
			}
			
		case "message_delta":
			// Message completion info
			if deltaData, ok := chunk["delta"].(map[string]interface{}); ok {
				if stopReason, ok := deltaData["stop_reason"].(string); ok && stopReason != "" {
					choice["finish_reason"] = "stop"
				}
			}
			
		case "message_stop":
			// End of message
			choice["finish_reason"] = "stop"
		}
	}

	return openaiChunk
}

// HandleRequest processes requests for Anthropic models via Bedrock
func (a *ClaudeAdapter) HandleRequest(req ChatRequest) (map[string]interface{}, error) {
	log.Printf("Processing non-streaming request for model: %s", req.Model)
	
	anthropicMessages, systemMessage := a.ConvertToAnthropicMessages(req.Messages)
	anthropicPayload := a.BuildAnthropicPayload(anthropicMessages, systemMessage)

	// Call Bedrock
	payloadBytes, _ := json.Marshal(anthropicPayload)
	
	log.Printf("Calling Bedrock with model ID: %s", req.Model)
	
	result, err := a.bedrockClient.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		ModelId:     &req.Model,
		ContentType: &[]string{"application/json"}[0],
		Body:        payloadBytes,
	})

	if err != nil {
		log.Printf("Bedrock error for model %s: %v", req.Model, err)
		return nil, err
	}

	// Parse Anthropic response
	var anthropicResp map[string]interface{}
	if err := json.Unmarshal(result.Body, &anthropicResp); err != nil {
		return nil, err
	}

	// Convert to OpenAI format
	return a.ConvertAnthropicToOpenAI(anthropicResp, req.Model), nil
}

// HandleStreamingRequest processes streaming requests for Anthropic models via Bedrock
func (a *ClaudeAdapter) HandleStreamingRequest(req ChatRequest, w http.ResponseWriter) error {
	log.Printf("Processing streaming request for model: %s", req.Model)
	
	anthropicMessages, systemMessage := a.ConvertToAnthropicMessages(req.Messages)
	anthropicPayload := a.BuildAnthropicStreamingPayload(anthropicMessages, systemMessage)

	// Call Bedrock with streaming
	payloadBytes, _ := json.Marshal(anthropicPayload)
	
	log.Printf("Calling Bedrock streaming with model ID: %s", req.Model)
	log.Printf("Payload: %s", string(payloadBytes))
	
	result, err := a.bedrockClient.InvokeModelWithResponseStream(context.Background(), &bedrockruntime.InvokeModelWithResponseStreamInput{
		ModelId:     &req.Model,
		ContentType: &[]string{"application/json"}[0],
		Body:        payloadBytes,
	})

	if err != nil {
		log.Printf("Bedrock streaming error for model %s: %v", req.Model, err)
		return err
	}

	// Headers are set by the server, not here

	// Process the stream
	stream := result.GetStream()

	for event := range stream.Events() {
		switch v := event.(type) {
		case *types.ResponseStreamMemberChunk:
			// Parse the chunk
			var chunk map[string]interface{}
			if err := json.Unmarshal(v.Value.Bytes, &chunk); err != nil {
				log.Printf("Failed to parse stream chunk: %v", err)
				continue
			}

			// Convert to OpenAI format
			openaiChunk := a.ConvertAnthropicStreamToOpenAI(chunk, req.Model)
			
			// Send as SSE
			data, _ := json.Marshal(openaiChunk)
			w.Write([]byte("data: " + string(data) + "\n\n"))
			
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}

		default:
			// Handle other event types if needed
			log.Printf("Unknown stream event type: %T", v)
		}
	}

	if err := stream.Err(); err != nil {
		return err
	}

	// Send completion marker
	w.Write([]byte("data: [DONE]\n\n"))
	
	return nil
}