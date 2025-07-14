package adapter

import (
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