package adapter

import (
	"strings"
	"testing"

	"github.com/google/generative-ai-go/genai"
)

// testError is a simple error type for testing
type testError struct {
	message string
}

func (e *testError) Error() string {
	return e.message
}

func TestGeminiAdapter_IsGeminiModel(t *testing.T) {
	adapter := &GeminiAdapter{}

	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		{
			name:     "Gemini 1.5 Pro",
			model:    "gemini-1.5-pro",
			expected: true,
		},
		{
			name:     "Gemini 1.5 Flash",
			model:    "gemini-1.5-flash",
			expected: true,
		},
		{
			name:     "Models prefix",
			model:    "models/gemini-1.5-pro",
			expected: true,
		},
		{
			name:     "GPT model",
			model:    "gpt-4",
			expected: false,
		},
		{
			name:     "Claude model",
			model:    "anthropic.claude-3-sonnet",
			expected: false,
		},
		{
			name:     "Empty model",
			model:    "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.IsGeminiModel(tt.model)
			if result != tt.expected {
				t.Errorf("IsGeminiModel(%q) = %v, want %v", tt.model, result, tt.expected)
			}
		})
	}
}

func TestGeminiAdapter_ConvertToGeminiMessages(t *testing.T) {
	adapter := &GeminiAdapter{}

	tests := []struct {
		name     string
		messages []Message
		wantErr  bool
	}{
		{
			name: "Basic user message",
			messages: []Message{
				{Role: "user", Content: "Hello"},
			},
			wantErr: false,
		},
		{
			name: "User and assistant conversation",
			messages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			wantErr: false,
		},
		{
			name: "System message handling",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant"},
				{Role: "user", Content: "Hello"},
			},
			wantErr: false,
		},
		{
			name: "Mixed roles",
			messages: []Message{
				{Role: "system", Content: "You are helpful"},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi!"},
				{Role: "user", Content: "How are you?"},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := adapter.ConvertToGeminiMessages(tt.messages)

			if (err != nil) != tt.wantErr {
				t.Errorf("ConvertToGeminiMessages() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if result == nil {
					t.Error("ConvertToGeminiMessages() returned nil result")
					return
				}

				// Basic validation - should have converted some messages
				if len(tt.messages) > 0 && len(result) == 0 {
					// Only fail if we had non-system messages
					hasNonSystem := false
					for _, msg := range tt.messages {
						if msg.Role != "system" {
							hasNonSystem = true
							break
						}
					}
					if hasNonSystem {
						t.Error("ConvertToGeminiMessages() returned empty result for non-empty input")
					}
				}

				// Check role conversion
				for _, geminiMsg := range result {
					if geminiMsg.Role != "user" && geminiMsg.Role != "model" {
						t.Errorf("Invalid Gemini role: %s", geminiMsg.Role)
					}
				}
			}
		})
	}
}

func TestGeminiAdapter_ConvertGeminiToOpenAI(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"

	tests := []struct {
		name         string
		geminiResp   *genai.GenerateContentResponse
		expectedKeys []string
	}{
		{
			name: "Valid response with content",
			geminiResp: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Hello, how can I help you?")},
						},
					},
				},
			},
			expectedKeys: []string{"id", "object", "created", "model", "choices", "usage"},
		},
		{
			name:         "Empty response",
			geminiResp:   &genai.GenerateContentResponse{},
			expectedKeys: []string{"id", "object", "created", "model", "choices", "usage"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ConvertGeminiToOpenAI(tt.geminiResp, model)

			// Check required keys exist
			for _, key := range tt.expectedKeys {
				if _, exists := result[key]; !exists {
					t.Errorf("Missing key in result: %s", key)
				}
			}

			// Check specific values
			if result["object"] != "chat.completion" {
				t.Errorf("Expected object to be 'chat.completion', got %v", result["object"])
			}

			if result["model"] != model {
				t.Errorf("Expected model to be %s, got %v", model, result["model"])
			}

			// Check choices structure
			choices, ok := result["choices"].([]map[string]interface{})
			if !ok || len(choices) == 0 {
				t.Error("Invalid or empty choices array")
				return
			}

			choice := choices[0]
			if choice["index"] != 0 {
				t.Errorf("Expected choice index to be 0, got %v", choice["index"])
			}

			message, ok := choice["message"].(map[string]string)
			if !ok {
				t.Error("Invalid message structure in choice")
				return
			}

			if message["role"] != "assistant" {
				t.Errorf("Expected message role to be 'assistant', got %v", message["role"])
			}
		})
	}
}

func TestGeminiAdapter_ConvertErrorToOpenAI(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"
	testError := &testError{"test error message"}

	result := adapter.ConvertErrorToOpenAI(testError, model)

	// Check error structure
	errorObj, ok := result["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error object in result")
	}

	if errorObj["message"] != testError.Error() {
		t.Errorf("Expected error message %q, got %v", testError.Error(), errorObj["message"])
	}

	if errorObj["type"] != "invalid_request_error" {
		t.Errorf("Expected error type 'invalid_request_error', got %v", errorObj["type"])
	}

	if errorObj["code"] != "model_error" {
		t.Errorf("Expected error code 'model_error', got %v", errorObj["code"])
	}
}

func TestGeminiAdapter_ConvertErrorToOpenAIStream(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"
	testError := &testError{"streaming test error"}

	result := adapter.ConvertErrorToOpenAIStream(testError, model)

	// Check basic structure
	if result["object"] != "chat.completion.chunk" {
		t.Errorf("Expected object to be 'chat.completion.chunk', got %v", result["object"])
	}

	if result["model"] != model {
		t.Errorf("Expected model to be %s, got %v", model, result["model"])
	}

	// Check error structure
	errorObj, ok := result["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error object in result")
	}

	if errorObj["message"] != testError.Error() {
		t.Errorf("Expected error message %q, got %v", testError.Error(), errorObj["message"])
	}
}

// TestGeminiAdapter_StreamingChunkParsing tests parsing of Gemini streaming chunks
func TestGeminiAdapter_StreamingChunkParsing(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"

	tests := []struct {
		name             string
		response         *genai.GenerateContentResponse
		isFirst          bool
		expectedContent  string
		shouldHaveRole   bool
		shouldHaveFinish bool
		expectedFinish   string
	}{
		{
			name: "first chunk with role",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Hello")},
						},
					},
				},
			},
			isFirst:         true,
			expectedContent: "Hello",
			shouldHaveRole:  true,
		},
		{
			name: "subsequent chunk without role",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text(" world!")},
						},
					},
				},
			},
			isFirst:         false,
			expectedContent: " world!",
			shouldHaveRole:  false,
		},
		{
			name: "chunk with stop finish reason",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Final chunk")},
						},
						FinishReason: genai.FinishReasonStop,
					},
				},
			},
			isFirst:          false,
			expectedContent:  "Final chunk",
			shouldHaveFinish: true,
			expectedFinish:   "stop",
		},
		{
			name: "chunk with max tokens finish reason",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Truncated")},
						},
						FinishReason: genai.FinishReasonMaxTokens,
					},
				},
			},
			isFirst:          false,
			expectedContent:  "Truncated",
			shouldHaveFinish: true,
			expectedFinish:   "stop",
		},
		{
			name: "chunk with safety finish reason",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Filtered")},
						},
						FinishReason: genai.FinishReasonSafety,
					},
				},
			},
			isFirst:          false,
			expectedContent:  "Filtered",
			shouldHaveFinish: true,
			expectedFinish:   "content_filter",
		},
		{
			name: "empty content chunk",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("")},
						},
					},
				},
			},
			isFirst:         false,
			expectedContent: "",
			shouldHaveRole:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ConvertGeminiStreamToOpenAI(tt.response, model, tt.isFirst)

			// Check basic structure
			if result["object"] != "chat.completion.chunk" {
				t.Errorf("Expected object to be 'chat.completion.chunk', got %v", result["object"])
			}

			if result["model"] != model {
				t.Errorf("Expected model to be %s, got %v", model, result["model"])
			}

			// Check choices structure
			choices, ok := result["choices"].([]map[string]interface{})
			if !ok || len(choices) == 0 {
				t.Fatal("Expected choices to be non-empty array")
			}

			choice := choices[0]
			delta, ok := choice["delta"].(map[string]interface{})
			if !ok {
				t.Fatal("Expected delta to be present")
			}

			// Test role expectation
			if tt.shouldHaveRole {
				if role, exists := delta["role"]; !exists || role != "assistant" {
					t.Errorf("Expected role 'assistant', got %v", role)
				}
			} else {
				if _, exists := delta["role"]; exists {
					t.Error("Did not expect role in delta")
				}
			}

			// Test content
			if tt.expectedContent != "" {
				if content, exists := delta["content"]; !exists || content != tt.expectedContent {
					t.Errorf("Expected content '%s', got %v", tt.expectedContent, content)
				}
			}

			// Test finish reason
			if tt.shouldHaveFinish {
				if finishReason := choice["finish_reason"]; finishReason != tt.expectedFinish {
					t.Errorf("Expected finish_reason '%s', got %v", tt.expectedFinish, finishReason)
				}
			} else {
				if finishReason := choice["finish_reason"]; finishReason != nil {
					t.Errorf("Expected finish_reason to be nil, got %v", finishReason)
				}
			}
		})
	}
}

// TestGeminiAdapter_ModelNameNormalization tests model name handling
func TestGeminiAdapter_ModelNameNormalization(t *testing.T) {
	adapter := &GeminiAdapter{}

	tests := []struct {
		name           string
		inputModel     string
		expectedDetect bool
	}{
		{
			name:           "gemini-1.5-pro",
			inputModel:     "gemini-1.5-pro",
			expectedDetect: true,
		},
		{
			name:           "gemini-1.5-flash",
			inputModel:     "gemini-1.5-flash",
			expectedDetect: true,
		},
		{
			name:           "models/gemini-1.5-pro",
			inputModel:     "models/gemini-1.5-pro",
			expectedDetect: true,
		},
		{
			name:           "GEMINI-1.5-PRO (case insensitive)",
			inputModel:     "GEMINI-1.5-PRO",
			expectedDetect: true,
		},
		{
			name:           "gemini-pro-vision",
			inputModel:     "gemini-pro-vision",
			expectedDetect: true,
		},
		{
			name:           "gpt-4 (non-gemini)",
			inputModel:     "gpt-4",
			expectedDetect: false,
		},
		{
			name:           "claude-3-sonnet (non-gemini)",
			inputModel:     "claude-3-sonnet",
			expectedDetect: false,
		},
		{
			name:           "text-embedding-004 (non-gemini)",
			inputModel:     "text-embedding-004",
			expectedDetect: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.IsGeminiModel(tt.inputModel)
			if result != tt.expectedDetect {
				t.Errorf("IsGeminiModel(%q) = %v, want %v", tt.inputModel, result, tt.expectedDetect)
			}
		})
	}
}

// TestGeminiAdapter_MessageFormatConversion tests message format conversion edge cases
func TestGeminiAdapter_MessageFormatConversion(t *testing.T) {
	adapter := &GeminiAdapter{}

	tests := []struct {
		name                   string
		messages               []Message
		expectedCount          int
		expectedSystemHandling bool
		hasUserMessage         bool
	}{
		{
			name: "system message only",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant."},
			},
			expectedCount:  0, // System-only messages get filtered out
			hasUserMessage: false,
		},
		{
			name: "system + user message",
			messages: []Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hello!"},
			},
			expectedCount:          1,
			expectedSystemHandling: true,
			hasUserMessage:         true,
		},
		{
			name: "multiple system messages",
			messages: []Message{
				{Role: "system", Content: "First instruction."},
				{Role: "system", Content: "Second instruction."},
				{Role: "user", Content: "Hello!"},
			},
			expectedCount:          1,
			expectedSystemHandling: true,
			hasUserMessage:         true,
		},
		{
			name: "system message in middle (ignored)",
			messages: []Message{
				{Role: "user", Content: "Hello!"},
				{Role: "system", Content: "Mid-conversation instruction."},
				{Role: "assistant", Content: "Hi there!"},
			},
			expectedCount:  2, // user + assistant, system gets ignored
			hasUserMessage: true,
		},
		{
			name: "role normalization",
			messages: []Message{
				{Role: "user", Content: "Hello!"},
				{Role: "assistant", Content: "Hi!"},
				{Role: "function", Content: "Function result"},
			},
			expectedCount:  3, // function role becomes user
			hasUserMessage: true,
		},
		{
			name: "empty content handling",
			messages: []Message{
				{Role: "user", Content: ""},
				{Role: "user", Content: "Hello!"},
			},
			expectedCount:  2,
			hasUserMessage: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := adapter.ConvertToGeminiMessages(tt.messages)
			if err != nil {
				t.Errorf("ConvertToGeminiMessages() unexpected error: %v", err)
				return
			}

			if len(result) != tt.expectedCount {
				t.Errorf("Expected %d Gemini messages, got %d", tt.expectedCount, len(result))
			}

			// Check role conversion
			for _, geminiMsg := range result {
				if geminiMsg.Role != "user" && geminiMsg.Role != "model" {
					t.Errorf("Invalid Gemini role: %s (expected 'user' or 'model')", geminiMsg.Role)
				}
			}

			// Check if system handling is working (system content prepended to user message)
			if tt.expectedSystemHandling && tt.hasUserMessage && len(result) > 0 {
				// Find the user message
				for _, geminiMsg := range result {
					if geminiMsg.Role == "user" && len(geminiMsg.Parts) > 0 {
						if text, ok := geminiMsg.Parts[0].(genai.Text); ok {
							content := string(text)
							if !strings.Contains(content, "System:") {
								t.Error("Expected system message to be prepended to user message")
							}
						}
					}
				}
			}
		})
	}
}

// TestGeminiAdapter_ResponseFormatValidation tests response format validation
func TestGeminiAdapter_ResponseFormatValidation(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"

	tests := []struct {
		name          string
		response      *genai.GenerateContentResponse
		expectedValid bool
		expectContent bool
		expectUsage   bool
	}{
		{
			name: "complete response with usage",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Hello! How can I help you?")},
						},
						FinishReason: genai.FinishReasonStop,
					},
				},
				UsageMetadata: &genai.UsageMetadata{
					PromptTokenCount:     10,
					CandidatesTokenCount: 15,
					TotalTokenCount:      25,
				},
			},
			expectedValid: true,
			expectContent: true,
			expectUsage:   true,
		},
		{
			name: "response without usage metadata",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Response without usage")},
						},
					},
				},
			},
			expectedValid: true,
			expectContent: true,
			expectUsage:   false,
		},
		{
			name: "empty response",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{},
			},
			expectedValid: true,
			expectContent: false,
			expectUsage:   false,
		},
		{
			name: "response with empty content",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("")},
						},
					},
				},
			},
			expectedValid: true,
			expectContent: false, // Empty content should result in no content
			expectUsage:   false,
		},
		{
			name: "response with non-text parts",
			response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{
								genai.Text("Text part"),
								// Note: In real usage, there could be other part types
							},
						},
					},
				},
			},
			expectedValid: true,
			expectContent: true,
			expectUsage:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ConvertGeminiToOpenAI(tt.response, model)

			// Check basic structure
			if result["object"] != "chat.completion" {
				t.Errorf("Expected object to be 'chat.completion', got %v", result["object"])
			}

			if result["model"] != model {
				t.Errorf("Expected model to be %s, got %v", model, result["model"])
			}

			// Check choices
			choices, ok := result["choices"].([]map[string]interface{})
			if !ok || len(choices) == 0 {
				t.Fatal("Expected choices to be non-empty array")
			}

			choice := choices[0]
			message, ok := choice["message"].(map[string]string)
			if !ok {
				t.Fatal("Expected message to be present")
			}

			// Check content expectation
			if tt.expectContent {
				if message["content"] == "" {
					t.Error("Expected non-empty content")
				}
			}

			// Check usage expectation
			usage, hasUsage := result["usage"].(map[string]interface{})
			if tt.expectUsage {
				if !hasUsage {
					t.Error("Expected usage metadata to be present")
				} else {
					// Verify usage structure
					if _, ok := usage["prompt_tokens"]; !ok {
						t.Error("Expected prompt_tokens in usage")
					}
					if _, ok := usage["completion_tokens"]; !ok {
						t.Error("Expected completion_tokens in usage")
					}
					if _, ok := usage["total_tokens"]; !ok {
						t.Error("Expected total_tokens in usage")
					}
				}
			} else {
				// Usage should still be present but with zero values
				if !hasUsage {
					t.Error("Expected usage object to be present even without metadata")
				}
			}
		})
	}
}

// TestGeminiAdapter_ErrorHandling tests error handling scenarios
func TestGeminiAdapter_ErrorHandling(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"

	tests := []struct {
		name       string
		error      error
		expectType string
		expectCode string
	}{
		{
			name:       "generic error",
			error:      &testError{"Generic API error"},
			expectType: "invalid_request_error",
			expectCode: "model_error",
		},
		{
			name:       "authentication error",
			error:      &testError{"API key not valid"},
			expectType: "invalid_request_error",
			expectCode: "model_error",
		},
		{
			name:       "rate limit error",
			error:      &testError{"Rate limit exceeded"},
			expectType: "invalid_request_error",
			expectCode: "model_error",
		},
		{
			name:       "safety filter error",
			error:      &testError{"Content blocked by safety filters"},
			expectType: "invalid_request_error",
			expectCode: "model_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test non-streaming error conversion
			result := adapter.ConvertErrorToOpenAI(tt.error, model)

			errorObj, ok := result["error"].(map[string]interface{})
			if !ok {
				t.Fatal("Expected error object in result")
			}

			if errorObj["message"] != tt.error.Error() {
				t.Errorf("Expected error message %q, got %v", tt.error.Error(), errorObj["message"])
			}

			if errorObj["type"] != tt.expectType {
				t.Errorf("Expected error type %s, got %v", tt.expectType, errorObj["type"])
			}

			if errorObj["code"] != tt.expectCode {
				t.Errorf("Expected error code %s, got %v", tt.expectCode, errorObj["code"])
			}

			// Test streaming error conversion
			streamResult := adapter.ConvertErrorToOpenAIStream(tt.error, model)

			if streamResult["object"] != "chat.completion.chunk" {
				t.Errorf("Expected streaming object to be 'chat.completion.chunk', got %v", streamResult["object"])
			}

			streamErrorObj, ok := streamResult["error"].(map[string]interface{})
			if !ok {
				t.Fatal("Expected error object in streaming result")
			}

			if streamErrorObj["message"] != tt.error.Error() {
				t.Errorf("Expected streaming error message %q, got %v", tt.error.Error(), streamErrorObj["message"])
			}
		})
	}
}

// TestGeminiAdapter_SafetyAndFinishReasons tests safety filtering and finish reasons
func TestGeminiAdapter_SafetyAndFinishReasons(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"

	tests := []struct {
		name           string
		finishReason   genai.FinishReason
		expectedReason string
	}{
		{
			name:           "normal completion",
			finishReason:   genai.FinishReasonStop,
			expectedReason: "stop",
		},
		{
			name:           "max tokens reached",
			finishReason:   genai.FinishReasonMaxTokens,
			expectedReason: "stop",
		},
		{
			name:           "safety filter triggered",
			finishReason:   genai.FinishReasonSafety,
			expectedReason: "content_filter",
		},
		{
			name:           "recitation filter (unhandled)",
			finishReason:   genai.FinishReasonRecitation,
			expectedReason: "", // Unhandled case, should be nil/empty
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []genai.Part{genai.Text("Test response")},
						},
						FinishReason: tt.finishReason,
					},
				},
			}

			// Test non-streaming conversion
			result := adapter.ConvertGeminiToOpenAI(response, model)
			choices, ok := result["choices"].([]map[string]interface{})
			if !ok || len(choices) == 0 {
				t.Fatal("Expected choices array")
			}

			if choices[0]["finish_reason"] != "stop" {
				t.Errorf("Expected finish_reason 'stop' in non-streaming, got %v", choices[0]["finish_reason"])
			}

			// Test streaming conversion
			streamResult := adapter.ConvertGeminiStreamToOpenAI(response, model, false)
			streamChoices, ok := streamResult["choices"].([]map[string]interface{})
			if !ok || len(streamChoices) == 0 {
				t.Fatal("Expected choices array in stream")
			}

			if tt.expectedReason == "" {
				// Expect nil for unhandled finish reasons
				if streamChoices[0]["finish_reason"] != nil {
					t.Errorf("Expected streaming finish_reason to be nil, got %v", streamChoices[0]["finish_reason"])
				}
			} else {
				if streamChoices[0]["finish_reason"] != tt.expectedReason {
					t.Errorf("Expected streaming finish_reason %s, got %v", tt.expectedReason, streamChoices[0]["finish_reason"])
				}
			}
		})
	}
}
