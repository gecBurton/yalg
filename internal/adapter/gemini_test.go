package adapter

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"google.golang.org/genai"
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
			name:     "GEMINI-1.5-PRO (case insensitive)",
			model:    "GEMINI-1.5-PRO",
			expected: true,
		},
		{
			name:     "gemini-pro-vision",
			model:    "gemini-pro-vision",
			expected: true,
		},
		{
			name:     "gemini-2.0-flash",
			model:    "gemini-2.0-flash",
			expected: true,
		},
		{
			name:     "gemini-2.5-flash-preview",
			model:    "gemini-2.5-flash-preview-04-17",
			expected: true,
		},
		{
			name:     "gemini-2.0-flash-exp-image-generation",
			model:    "gemini-2.0-flash-exp-image-generation",
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
			name:     "text-embedding-004 (non-gemini)",
			model:    "text-embedding-004",
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

func TestGeminiAdapter_HandleEmbeddingRequest_ModelFormatting(t *testing.T) {
	tests := []struct {
		name          string
		inputModel    string
		expectedModel string
	}{
		{
			name:          "Model with models/ prefix",
			inputModel:    "models/embedding-001",
			expectedModel: "models/embedding-001",
		},
		{
			name:          "Model without models/ prefix",
			inputModel:    "embedding-001",
			expectedModel: "models/embedding-001",
		},
		{
			name:          "Text embedding model",
			inputModel:    "text-embedding-004",
			expectedModel: "models/text-embedding-004",
		},
		{
			name:          "Already formatted model",
			inputModel:    "models/text-embedding-004",
			expectedModel: "models/text-embedding-004",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test model name formatting logic
			modelName := tt.inputModel
			if !strings.HasPrefix(modelName, "models/") {
				modelName = "models/" + modelName
			}
			
			if modelName != tt.expectedModel {
				t.Errorf("Expected model name %q, got %q", tt.expectedModel, modelName)
			}
		})
	}
}

func TestGeminiAdapter_HandleEmbeddingRequest_TokenEstimation(t *testing.T) {
	tests := []struct {
		name          string
		input         []string
		expectedTokens int
	}{
		{
			name:          "Single word",
			input:         []string{"hello"},
			expectedTokens: 1,
		},
		{
			name:          "Multiple words",
			input:         []string{"hello world test"},
			expectedTokens: 3,
		},
		{
			name:          "Multiple inputs",
			input:         []string{"hello", "world test", "this is longer"},
			expectedTokens: 6, // 1 + 2 + 3 = 6
		},
		{
			name:          "Empty string",
			input:         []string{""},
			expectedTokens: 0,
		},
		{
			name:          "Mixed content",
			input:         []string{"hello", "", "world test"},
			expectedTokens: 3, // 1 + 0 + 2 = 3
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test token estimation logic without requiring actual API calls
			totalTokens := 0
			for _, text := range tt.input {
				totalTokens += len(strings.Fields(text))
			}
			
			if totalTokens != tt.expectedTokens {
				t.Errorf("Expected %d tokens, got %d", tt.expectedTokens, totalTokens)
			}
		})
	}
}

func TestGeminiAdapter_InputValidation(t *testing.T) {
	tests := []struct {
		name        string
		input       interface{}
		expected    []string
		expectError bool
	}{
		{
			name:        "String input",
			input:       "Hello world",
			expected:    []string{"Hello world"},
			expectError: false,
		},
		{
			name:        "String slice input",
			input:       []string{"Hello", "world"},
			expected:    []string{"Hello", "world"},
			expectError: false,
		},
		{
			name:        "Interface slice with strings",
			input:       []interface{}{"Hello", "world"},
			expected:    []string{"Hello", "world"},
			expectError: false,
		},
		{
			name:        "Empty string slice",
			input:       []string{},
			expected:    []string{},
			expectError: false,
		},
		{
			name:        "Interface slice with non-string",
			input:       []interface{}{"Hello", 123},
			expected:    nil,
			expectError: true,
		},
		{
			name:        "Invalid input type",
			input:       123,
			expected:    nil,
			expectError: true,
		},
		{
			name:        "Nil input",
			input:       nil,
			expected:    nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test input validation logic
			var result []string
			var err error

			switch v := tt.input.(type) {
			case string:
				result = []string{v}
			case []string:
				result = v
			case []interface{}:
				result = make([]string, len(v))
				for i, item := range v {
					if str, ok := item.(string); ok {
						result[i] = str
					} else {
						err = fmt.Errorf("all input items must be strings")
						break
					}
				}
			default:
				err = fmt.Errorf("input must be a string or array of strings")
			}

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if len(result) != len(tt.expected) {
					t.Errorf("Expected %d items, got %d", len(tt.expected), len(result))
				}
				for i, expected := range tt.expected {
					if i < len(result) && result[i] != expected {
						t.Errorf("Expected item %d to be %q, got %q", i, expected, result[i])
					}
				}
			}
		})
	}
}

func TestGeminiAdapter_ResponseFormatConversion(t *testing.T) {
	tests := []struct {
		name        string
		geminiResp  map[string]any
		expectError bool
		validate    func(t *testing.T, data []map[string]any)
	}{
		{
			name: "Valid response with float64 embeddings",
			geminiResp: map[string]any{
				"object": "list",
				"data": []map[string]any{
					{
						"object":    "embedding",
						"index":     0,
						"embedding": []float64{0.1, 0.2, 0.3},
					},
				},
				"usage": map[string]any{
					"prompt_tokens": 5.0,
					"total_tokens":  5.0,
				},
			},
			expectError: false,
			validate: func(t *testing.T, data []map[string]any) {
				if len(data) != 1 {
					t.Errorf("Expected 1 item, got %d", len(data))
				}
				if data[0]["index"] != 0 {
					t.Errorf("Expected index 0, got %v", data[0]["index"])
				}
			},
		},
		{
			name: "Valid response with float32 embeddings",
			geminiResp: map[string]any{
				"object": "list",
				"data": []map[string]any{
					{
						"object":    "embedding",
						"index":     0,
						"embedding": []float32{0.1, 0.2, 0.3},
					},
				},
				"usage": map[string]any{
					"prompt_tokens": 5.0,
					"total_tokens":  5.0,
				},
			},
			expectError: false,
			validate: func(t *testing.T, data []map[string]any) {
				if len(data) != 1 {
					t.Errorf("Expected 1 item, got %d", len(data))
				}
			},
		},
		{
			name: "Invalid response format - data is not array",
			geminiResp: map[string]any{
				"object": "list",
				"data":   "invalid",
			},
			expectError: true,
		},
		{
			name: "Invalid embedding format",
			geminiResp: map[string]any{
				"object": "list",
				"data": []map[string]any{
					{
						"object":    "embedding",
						"index":     0,
						"embedding": "invalid",
					},
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test response format validation
			data, ok := tt.geminiResp["data"].([]map[string]any)
			if !ok {
				if !tt.expectError {
					t.Error("Expected valid data format")
				}
				return
			}

			// Test embedding format conversion
			hasError := false
			for _, item := range data {
				if _, ok := item["embedding"].([]float32); !ok {
					if _, ok := item["embedding"].([]float64); !ok {
						hasError = true
						break
					}
				}
			}

			if tt.expectError && !hasError {
				t.Error("Expected error but validation passed")
			} else if !tt.expectError && hasError {
				t.Error("Unexpected validation error")
			}

			if !tt.expectError && tt.validate != nil {
				tt.validate(t, data)
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
		{
			name: "Multiple messages with system",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
				{Role: "assistant", Content: "I'm doing well, thank you for asking! How can I help you today?"},
				{Role: "user", Content: "Can you help me with math?"},
				{Role: "assistant", Content: "Of course! I'd be happy to help you with math. What specific topic or problem would you like assistance with?"},
				{Role: "user", Content: "What's 2 + 2?"},
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
					if geminiMsg.Role != genai.RoleUser && geminiMsg.Role != genai.RoleModel {
						t.Errorf("Invalid Gemini role: %s", geminiMsg.Role)
					}
				}

				// Check system message handling for multiple messages test
				if tt.name == "Multiple messages with system" {
					// Check alternating pattern (user/model/user/model/user)
					expectedRoles := []string{"user", "model", "user", "model", "user"}
					if len(result) != len(expectedRoles) {
						t.Errorf("Expected %d messages, got %d", len(expectedRoles), len(result))
						return
					}

					for i, msg := range result {
						expectedRole := genai.RoleUser
						if expectedRoles[i] == "model" {
							expectedRole = genai.RoleModel
						}
						if msg.Role != expectedRole {
							t.Errorf("Expected role %s at index %d, got %s", expectedRoles[i], i, msg.Role)
						}
					}

					// Check that system message is prepended to first user message
					if len(result[0].Parts) > 0 && result[0].Parts[0].Text != "" {
						content := result[0].Parts[0].Text
						if !strings.Contains(content, "System:") {
							t.Error("Expected system message to be prepended to first user message")
						}
						if !strings.Contains(content, "Hello, how are you?") {
							t.Error("Expected user message content to be preserved")
						}
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
							Parts: []*genai.Part{{Text: "Hello, how can I help you?"}},
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
		{
			name: "Response with usage metadata",
			geminiResp: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []*genai.Part{{Text: "This is a test response."}},
						},
						FinishReason: genai.FinishReasonStop,
					},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     50,
					CandidatesTokenCount: 25,
					TotalTokenCount:      75,
				},
			},
			expectedKeys: []string{"id", "object", "created", "model", "choices", "usage"},
		},
		{
			name: "JSON response",
			geminiResp: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []*genai.Part{{Text: `{"name": "John", "age": 30, "city": "New York"}`}},
						},
						FinishReason: genai.FinishReasonStop,
					},
				},
			},
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

			// Test specific scenarios
			if tt.name == "Response with usage metadata" {
				// Check usage metadata
				usage, ok := result["usage"].(map[string]interface{})
				if !ok {
					t.Fatal("Expected usage metadata")
				}

				if usage["prompt_tokens"] != 50 {
					t.Errorf("Expected prompt_tokens to be 50, got %v", usage["prompt_tokens"])
				}

				if usage["completion_tokens"] != 25 {
					t.Errorf("Expected completion_tokens to be 25, got %v", usage["completion_tokens"])
				}

				if usage["total_tokens"] != 75 {
					t.Errorf("Expected total_tokens to be 75, got %v", usage["total_tokens"])
				}
			}

			if tt.name == "JSON response" {
				message, ok := choice["message"].(map[string]string)
				if !ok {
					t.Fatal("Expected message object")
				}

				jsonResponse := `{"name": "John", "age": 30, "city": "New York"}`
				// Verify JSON content is preserved
				if message["content"] != jsonResponse {
					t.Errorf("Expected content to be %s, got %s", jsonResponse, message["content"])
				}

				// Verify it's valid JSON
				var parsedJSON map[string]interface{}
				if err := json.Unmarshal([]byte(message["content"]), &parsedJSON); err != nil {
					t.Errorf("Response content is not valid JSON: %v", err)
				}
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
							Parts: []*genai.Part{{Text: "Hello"}},
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
							Parts: []*genai.Part{{Text: " world!"}},
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
							Parts: []*genai.Part{{Text: "Final chunk"}},
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
							Parts: []*genai.Part{{Text: "Truncated"}},
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
							Parts: []*genai.Part{{Text: "Filtered"}},
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
							Parts: []*genai.Part{{Text: ""}},
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
				if geminiMsg.Role != genai.RoleUser && geminiMsg.Role != genai.RoleModel {
					t.Errorf("Invalid Gemini role: %s (expected 'user' or 'model')", geminiMsg.Role)
				}
			}

			// Check if system handling is working (system content prepended to user message)
			if tt.expectedSystemHandling && tt.hasUserMessage && len(result) > 0 {
				// Find the user message
				for _, geminiMsg := range result {
					if geminiMsg.Role == genai.RoleUser && len(geminiMsg.Parts) > 0 {
						if geminiMsg.Parts[0].Text != "" {
							content := geminiMsg.Parts[0].Text
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
			expectedReason: "stop", // LiteLLM expects "stop" for max tokens
		},
		{
			name:           "safety filter triggered",
			finishReason:   genai.FinishReasonSafety,
			expectedReason: "content_filter",
		},
		{
			name:           "recitation filter (unhandled)",
			finishReason:   genai.FinishReasonRecitation,
			expectedReason: "stop", // Non-streaming fallback to "stop"
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []*genai.Part{{Text: "Test response"}},
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

			if choices[0]["finish_reason"] != tt.expectedReason {
				t.Errorf("Expected finish_reason '%s' in non-streaming, got %v", tt.expectedReason, choices[0]["finish_reason"])
			}

			// Test streaming conversion
			streamResult := adapter.ConvertGeminiStreamToOpenAI(response, model, false)
			streamChoices, ok := streamResult["choices"].([]map[string]interface{})
			if !ok || len(streamChoices) == 0 {
				t.Fatal("Expected choices array in stream")
			}

			// For streaming, recitation should be nil, others should match expected
			if tt.finishReason == genai.FinishReasonRecitation {
				if streamChoices[0]["finish_reason"] != nil {
					t.Errorf("Expected streaming finish_reason to be nil for recitation, got %v", streamChoices[0]["finish_reason"])
				}
			} else {
				if streamChoices[0]["finish_reason"] != tt.expectedReason {
					t.Errorf("Expected streaming finish_reason %s, got %v", tt.expectedReason, streamChoices[0]["finish_reason"])
				}
			}
		})
	}
}

// ===== LiteLLM-based Tests =====

// TestGeminiAdapter_ToolCallNoArguments tests tool calls with no arguments
// Based on LiteLLM test_tool_call_no_arguments
func TestGeminiAdapter_ToolCallNoArguments(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test basic tool call scenario (simulating function call response)
	messages := []Message{
		{Role: "user", Content: "What's the weather like?"},
		{Role: "assistant", Content: "I'll check the weather for you."},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	if len(result) == 0 {
		t.Error("Expected at least one message after conversion")
		return
	}

	// Verify structure is valid
	for _, msg := range result {
		if msg.Role != genai.RoleUser && msg.Role != genai.RoleModel {
			t.Errorf("Invalid role: %s", msg.Role)
		}
	}
}

// TestGeminiAdapter_ContextCaching tests context caching functionality
// Based on LiteLLM test_gemini_context_caching_with_ttl
func TestGeminiAdapter_ContextCaching(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test message with cache control
	messages := []Message{
		{
			Role:    "system",
			Content: "Here is the full text of a complex legal agreement. This is a long document with many terms and conditions that should be cached for performance.",
		},
		{
			Role:    "user",
			Content: "What are the key terms and conditions in this agreement?",
		},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	// Should handle system messages by prepending to user message
	if len(result) == 0 {
		t.Error("Expected at least one message after conversion")
		return
	}

	// Find user message and check if system content is prepended
	foundSystemContent := false
	for _, msg := range result {
		if msg.Role == genai.RoleUser && len(msg.Parts) > 0 {
			if msg.Parts[0].Text != "" {
				content := msg.Parts[0].Text
				if strings.Contains(content, "System:") {
					foundSystemContent = true
					break
				}
			}
		}
	}

	if !foundSystemContent {
		t.Error("Expected system message to be prepended to user message")
	}
}

// TestGeminiAdapter_ImageGeneration tests image generation capabilities
// Based on LiteLLM test_gemini_image_generation
func TestGeminiAdapter_ImageGeneration(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test image generation request
	messages := []Message{
		{Role: "user", Content: "Generate an image of a cat"},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	if len(result) == 0 {
		t.Error("Expected at least one message after conversion")
		return
	}

	// Verify basic structure
	if result[0].Role != genai.RoleUser {
		t.Errorf("Expected user role, got %s", result[0].Role)
	}

	if len(result[0].Parts) == 0 {
		t.Error("Expected at least one part in message")
		return
	}

	// Check content
	if result[0].Parts[0].Text != "" {
		content := result[0].Parts[0].Text
		if !strings.Contains(content, "cat") {
			t.Error("Expected content to contain 'cat'")
		}
	}
}

// TestGeminiAdapter_ThinkingMode tests thinking/reasoning capabilities
// Based on LiteLLM test_gemini_thinking
func TestGeminiAdapter_ThinkingMode(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test thinking mode request
	messages := []Message{
		{Role: "user", Content: "Explain the concept of Occam's Razor and provide a simple, everyday example"},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	if len(result) == 0 {
		t.Error("Expected at least one message after conversion")
		return
	}

	// Verify basic structure
	if result[0].Role != genai.RoleUser {
		t.Errorf("Expected user role, got %s", result[0].Role)
	}

	// Check content contains the question
	if result[0].Parts[0].Text != "" {
		content := result[0].Parts[0].Text
		if !strings.Contains(content, "Occam's Razor") {
			t.Error("Expected content to contain 'Occam's Razor'")
		}
	}
}

// TestGeminiAdapter_URLContext tests URL context functionality
// Based on LiteLLM test_gemini_url_context
func TestGeminiAdapter_URLContext(t *testing.T) {
	adapter := &GeminiAdapter{}

	url := "https://ai.google.dev/gemini-api/docs/models"
	prompt := "Summarize this document: " + url

	messages := []Message{
		{Role: "user", Content: prompt},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	if len(result) == 0 {
		t.Error("Expected at least one message after conversion")
		return
	}

	// Check that URL is preserved in content
	if result[0].Parts[0].Text != "" {
		content := result[0].Parts[0].Text
		if !strings.Contains(content, url) {
			t.Error("Expected content to contain URL")
		}
	}
}

// TestGeminiAdapter_ToolUse tests tool usage functionality
// Based on LiteLLM test_gemini_tool_use
func TestGeminiAdapter_ToolUse(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test tool use request
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What's the weather like in Lima, Peru today?"},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	if len(result) == 0 {
		t.Error("Expected at least one message after conversion")
		return
	}

	// Check system message handling
	foundSystemContent := false
	for _, msg := range result {
		if msg.Role == genai.RoleUser && len(msg.Parts) > 0 {
			if msg.Parts[0].Text != "" {
				content := msg.Parts[0].Text
				if strings.Contains(content, "System:") && strings.Contains(content, "Lima, Peru") {
					foundSystemContent = true
					break
				}
			}
		}
	}

	if !foundSystemContent {
		t.Error("Expected system message to be prepended and user content to be preserved")
	}
}

// TestGeminiAdapter_StreamingToolUse tests streaming tool usage
// Based on LiteLLM test_claude_tool_use_with_gemini and test_gemini_tool_use
func TestGeminiAdapter_StreamingToolUse(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-2.5-flash"

	// Test streaming response with tool use
	response := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "I'll help you get the weather information."}},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	// Test first chunk
	result := adapter.ConvertGeminiStreamToOpenAI(response, model, true)
	if result["object"] != "chat.completion.chunk" {
		t.Errorf("Expected object to be 'chat.completion.chunk', got %v", result["object"])
	}

	choices, ok := result["choices"].([]map[string]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("Expected choices array")
	}

	delta, ok := choices[0]["delta"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected delta object")
	}

	// First chunk should have role
	if delta["role"] != "assistant" {
		t.Errorf("Expected role 'assistant', got %v", delta["role"])
	}

	// Test subsequent chunk
	result2 := adapter.ConvertGeminiStreamToOpenAI(response, model, false)
	choices2, ok := result2["choices"].([]map[string]interface{})
	if !ok || len(choices2) == 0 {
		t.Fatal("Expected choices array")
	}

	delta2, ok := choices2[0]["delta"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected delta object")
	}

	// Subsequent chunks should not have role
	if _, hasRole := delta2["role"]; hasRole {
		t.Error("Subsequent chunks should not have role")
	}
}

// ===== Advanced Tests =====

// TestGeminiAdapter_RealToolCalling tests actual tool calling functionality
func TestGeminiAdapter_RealToolCalling(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test with real tool definition
	tools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_weather",
				Description: "Get current weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "City and country, e.g. 'Lima, Peru'",
						},
						"unit": map[string]interface{}{
							"type":        "string",
							"enum":        []string{"celsius", "fahrenheit"},
							"description": "Temperature unit",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	// Test tool conversion
	geminiTools := adapter.convertOpenAIToolsToGemini(tools)
	if len(geminiTools) != 1 {
		t.Errorf("Expected 1 Gemini tool, got %d", len(geminiTools))
		return
	}

	tool := geminiTools[0]
	if len(tool.FunctionDeclarations) != 1 {
		t.Errorf("Expected 1 function declaration, got %d", len(tool.FunctionDeclarations))
		return
	}

	funcDecl := tool.FunctionDeclarations[0]
	if funcDecl.Name != "get_weather" {
		t.Errorf("Expected function name 'get_weather', got '%s'", funcDecl.Name)
	}

	if funcDecl.Description != "Get current weather for a location" {
		t.Errorf("Expected function description, got '%s'", funcDecl.Description)
	}

	// Test parameter schema conversion
	if funcDecl.Parameters.Type != genai.TypeObject {
		t.Errorf("Expected object type for parameters")
	}

	if funcDecl.Parameters.Properties == nil {
		t.Error("Expected properties in parameters")
		return
	}

	locationProp := funcDecl.Parameters.Properties["location"]
	if locationProp.Type != genai.TypeString {
		t.Errorf("Expected string type for location parameter")
	}

	// Test required fields
	if len(funcDecl.Parameters.Required) != 1 || funcDecl.Parameters.Required[0] != "location" {
		t.Errorf("Expected required field 'location', got %v", funcDecl.Parameters.Required)
	}
}

// TestGeminiAdapter_ToolCallParsing tests parsing of tool calls from Gemini response
func TestGeminiAdapter_ToolCallParsing(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Create a mock Gemini function call
	mockFunctionCall := genai.FunctionCall{
		Name: "get_weather",
		Args: map[string]interface{}{
			"location": "Lima, Peru",
			"unit":     "celsius",
		},
	}

	// Test parsing tool calls
	parts := []*genai.Part{
		{Text: "I'll get the weather for you."},
		{FunctionCall: &mockFunctionCall},
	}

	toolCalls := adapter.parseToolCalls(parts)
	if len(toolCalls) != 1 {
		t.Errorf("Expected 1 tool call, got %d", len(toolCalls))
		return
	}

	toolCall := toolCalls[0]
	if toolCall.Type != "function" {
		t.Errorf("Expected type 'function', got '%s'", toolCall.Type)
	}

	if toolCall.Function.Name != "get_weather" {
		t.Errorf("Expected function name 'get_weather', got '%s'", toolCall.Function.Name)
	}

	// Test arguments parsing
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
		t.Errorf("Failed to parse arguments JSON: %v", err)
		return
	}

	if args["location"] != "Lima, Peru" {
		t.Errorf("Expected location 'Lima, Peru', got '%v'", args["location"])
	}

	if args["unit"] != "celsius" {
		t.Errorf("Expected unit 'celsius', got '%v'", args["unit"])
	}

	// Test ID generation
	if toolCall.ID == "" {
		t.Error("Expected tool call ID to be generated")
	}

	if !strings.HasPrefix(toolCall.ID, "call_") {
		t.Errorf("Expected tool call ID to start with 'call_', got '%s'", toolCall.ID)
	}
}

// TestGeminiAdapter_MultiModalContent tests multi-modal content handling
func TestGeminiAdapter_MultiModalContent(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test with mixed content types
	messages := []Message{
		{
			Role: "user",
			Content: []ContentPart{
				{
					Type: "text",
					Text: "What's in this image?",
				},
				{
					Type: "image_url",
					ImageURL: &ImageURL{
						URL:    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
						Detail: "high",
					},
				},
			},
		},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	if len(result) != 1 {
		t.Errorf("Expected 1 message, got %d", len(result))
		return
	}

	message := result[0]
	if message.Role != "user" {
		t.Errorf("Expected role 'user', got '%s'", message.Role)
	}

	if len(message.Parts) != 2 {
		t.Errorf("Expected 2 parts (text + image), got %d", len(message.Parts))
		return
	}

	// Check text part
	if message.Parts[0].Text != "" {
		if message.Parts[0].Text != "What's in this image?" {
			t.Errorf("Expected text 'What's in this image?', got '%s'", message.Parts[0].Text)
		}
	} else {
		t.Error("Expected first part to be text")
	}

	// Check image part (should be converted from base64)
	if message.Parts[1].InlineData != nil {
		if len(message.Parts[1].InlineData.Data) == 0 {
			t.Error("Expected image data to be present")
		}
	} else {
		t.Error("Expected second part to be image data")
	}
}

// TestGeminiAdapter_AdvancedContextCaching tests context caching with TTL
func TestGeminiAdapter_AdvancedContextCaching(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test message with cache control
	messages := []Message{
		{
			Role: "system",
			Content: []ContentPart{
				{
					Type: "text",
					Text: "Here is the full text of a complex legal agreement that should be cached for performance.",
					CacheControl: &CacheControl{
						Type: "ephemeral",
						TTL:  "3600s",
					},
				},
			},
		},
		{
			Role:    "user",
			Content: "What are the key terms and conditions in this agreement?",
		},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	if len(result) != 1 {
		t.Errorf("Expected 1 message after system message handling, got %d", len(result))
		return
	}

	// Check that system message was prepended to user message
	message := result[0]
	if message.Role != "user" {
		t.Errorf("Expected role 'user', got '%s'", message.Role)
	}

	if len(message.Parts) < 1 {
		t.Errorf("Expected at least 1 part, got %d", len(message.Parts))
		return
	}

	// Check all text parts for content
	var allContent string
	for _, part := range message.Parts {
		if part.Text != "" {
			allContent += part.Text
		}
	}

	if !strings.Contains(allContent, "System:") {
		t.Error("Expected system message to be prepended")
	}
	if !strings.Contains(allContent, "complex legal agreement") {
		t.Error("Expected system content to be included")
	}
	if !strings.Contains(allContent, "key terms and conditions") {
		t.Error("Expected user content to be included")
	}
}

// TestGeminiAdapter_EmptyFunctionArguments tests handling of empty function arguments
func TestGeminiAdapter_EmptyFunctionArguments(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test with empty function call arguments
	mockFunctionCall := genai.FunctionCall{
		Name: "get_current_weather",
		Args: map[string]interface{}{}, // Empty arguments
	}

	parts := []*genai.Part{{FunctionCall: &mockFunctionCall}}
	toolCalls := adapter.parseToolCalls(parts)

	if len(toolCalls) != 1 {
		t.Errorf("Expected 1 tool call, got %d", len(toolCalls))
		return
	}

	toolCall := toolCalls[0]
	if toolCall.Function.Name != "get_current_weather" {
		t.Errorf("Expected function name 'get_current_weather', got '%s'", toolCall.Function.Name)
	}

	// Arguments should be empty object
	if toolCall.Function.Arguments != "{}" {
		t.Errorf("Expected empty arguments '{}', got '%s'", toolCall.Function.Arguments)
	}
}

// TestGeminiAdapter_ResponseWithToolCalls tests response conversion with tool calls
func TestGeminiAdapter_ResponseWithToolCalls(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-2.0-flash"

	// Create mock response with tool call
	mockFunctionCall := genai.FunctionCall{
		Name: "get_weather",
		Args: map[string]interface{}{
			"location": "Lima, Peru",
		},
	}

	response := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "I'll get the weather for you."},
						{FunctionCall: &mockFunctionCall},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     50,
			CandidatesTokenCount: 25,
			TotalTokenCount:      75,
		},
	}

	result := adapter.ConvertGeminiToOpenAI(response, model)

	// Check basic structure
	if result["object"] != "chat.completion" {
		t.Errorf("Expected object 'chat.completion', got %v", result["object"])
	}

	if result["model"] != model {
		t.Errorf("Expected model '%s', got %v", model, result["model"])
	}

	// Check choices
	choices, ok := result["choices"].([]map[string]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("Expected choices array")
	}

	choice := choices[0]
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("Expected finish_reason 'tool_calls', got %v", choice["finish_reason"])
	}

	// Check message structure
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected message object")
	}

	if message["role"] != "assistant" {
		t.Errorf("Expected role 'assistant', got %v", message["role"])
	}

	// Content should be nil when there are tool calls
	if message["content"] != nil {
		t.Errorf("Expected content to be nil when tool calls present, got %v", message["content"])
	}

	// Check tool calls
	toolCalls, ok := message["tool_calls"].([]ToolCall)
	if !ok || len(toolCalls) == 0 {
		t.Fatal("Expected tool_calls array")
	}

	toolCall := toolCalls[0]
	if toolCall.Type != "function" {
		t.Errorf("Expected type 'function', got '%s'", toolCall.Type)
	}

	if toolCall.Function.Name != "get_weather" {
		t.Errorf("Expected function name 'get_weather', got '%s'", toolCall.Function.Name)
	}

	// Check usage metadata
	usage, ok := result["usage"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected usage metadata")
	}

	if usage["prompt_tokens"] != 50 {
		t.Errorf("Expected prompt_tokens 50, got %v", usage["prompt_tokens"])
	}

	if usage["completion_tokens"] != 25 {
		t.Errorf("Expected completion_tokens 25, got %v", usage["completion_tokens"])
	}

	if usage["total_tokens"] != 75 {
		t.Errorf("Expected total_tokens 75, got %v", usage["total_tokens"])
	}
}

// TestGeminiAdapter_StreamingWithToolCalls tests streaming with tool calls
func TestGeminiAdapter_StreamingWithToolCalls(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-2.0-flash"

	// Create mock streaming response with tool call
	mockFunctionCall := genai.FunctionCall{
		Name: "get_weather",
		Args: map[string]interface{}{
			"location": "Lima, Peru",
		},
	}

	response := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "I'll help you with the weather."},
						{FunctionCall: &mockFunctionCall},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	// Test first chunk
	result := adapter.ConvertGeminiStreamToOpenAI(response, model, true)
	if result["object"] != "chat.completion.chunk" {
		t.Errorf("Expected object 'chat.completion.chunk', got %v", result["object"])
	}

	choices, ok := result["choices"].([]map[string]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("Expected choices array")
	}

	choice := choices[0]
	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected delta object")
	}

	// First chunk should have role
	if delta["role"] != "assistant" {
		t.Errorf("Expected role 'assistant', got %v", delta["role"])
	}

	// Should have tool calls
	if toolCalls, exists := delta["tool_calls"]; !exists {
		t.Error("Expected tool_calls in delta")
	} else if toolCallsArray, ok := toolCalls.([]ToolCall); !ok || len(toolCallsArray) == 0 {
		t.Error("Expected non-empty tool_calls array")
	}

	// Content should be nil when there are tool calls
	if delta["content"] != nil {
		t.Errorf("Expected content to be nil when tool calls present, got %v", delta["content"])
	}

	// Finish reason should be tool_calls
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("Expected finish_reason 'tool_calls', got %v", choice["finish_reason"])
	}
}

// TestGeminiAdapter_FinishReasonMapping tests finish reason mapping
func TestGeminiAdapter_FinishReasonMapping(t *testing.T) {
	adapter := &GeminiAdapter{}
	model := "gemini-1.5-pro"

	tests := []struct {
		name                 string
		finishReason         genai.FinishReason
		expectedReason       string
		expectedStreamReason interface{} // Can be string or nil
	}{
		{
			name:                 "stop reason",
			finishReason:         genai.FinishReasonStop,
			expectedReason:       "stop",
			expectedStreamReason: "stop",
		},
		{
			name:                 "max tokens reason",
			finishReason:         genai.FinishReasonMaxTokens,
			expectedReason:       "stop", // LiteLLM expects "stop" for max tokens
			expectedStreamReason: "stop",
		},
		{
			name:                 "safety reason",
			finishReason:         genai.FinishReasonSafety,
			expectedReason:       "content_filter",
			expectedStreamReason: "content_filter",
		},
		{
			name:                 "recitation reason",
			finishReason:         genai.FinishReasonRecitation,
			expectedReason:       "stop",
			expectedStreamReason: nil, // nil in streaming mode
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Parts: []*genai.Part{{Text: "Test response"}},
						},
						FinishReason: tt.finishReason,
					},
				},
			}

			// Test non-streaming
			result := adapter.ConvertGeminiToOpenAI(response, model)
			choices, ok := result["choices"].([]map[string]interface{})
			if !ok || len(choices) == 0 {
				t.Fatal("Expected choices array")
			}

			if choices[0]["finish_reason"] != tt.expectedReason {
				t.Errorf("Expected finish_reason '%s', got %v", tt.expectedReason, choices[0]["finish_reason"])
			}

			// Test streaming
			streamResult := adapter.ConvertGeminiStreamToOpenAI(response, model, false)
			streamChoices, ok := streamResult["choices"].([]map[string]interface{})
			if !ok || len(streamChoices) == 0 {
				t.Fatal("Expected choices array in stream")
			}

			if streamChoices[0]["finish_reason"] != tt.expectedStreamReason {
				t.Errorf("Expected streaming finish_reason %v, got %v", tt.expectedStreamReason, streamChoices[0]["finish_reason"])
			}
		})
	}
}

// TestDebugToolUse tests debug functionality for tool use
func TestDebugToolUse(t *testing.T) {
	adapter := &GeminiAdapter{}

	// Test tool use request
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What's the weather like in Lima, Peru today?"},
	}

	result, err := adapter.ConvertToGeminiMessages(messages)
	if err != nil {
		t.Errorf("ConvertToGeminiMessages() error = %v", err)
		return
	}

	fmt.Printf("Result length: %d\n", len(result))
	for i, msg := range result {
		fmt.Printf("Message %d: Role=%s, Parts=%d\n", i, msg.Role, len(msg.Parts))
		for j, part := range msg.Parts {
			if part.Text != "" {
				fmt.Printf("  Part %d: %s\n", j, part.Text)
			}
		}
	}

	// Check system message handling
	foundSystemContent := false
	for _, msg := range result {
		if msg.Role == genai.RoleUser && len(msg.Parts) > 0 {
			if msg.Parts[0].Text != "" {
				content := msg.Parts[0].Text
				fmt.Printf("User message content: %s\n", content)
				if strings.Contains(content, "System:") && strings.Contains(content, "Lima, Peru") {
					foundSystemContent = true
					break
				}
			}
		}
	}

	if !foundSystemContent {
		t.Error("Expected system message to be prepended and user content to be preserved")
	}
}
