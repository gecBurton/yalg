package adapter

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/aws/aws-sdk-go-v2/aws"
)

func TestClaudeAdapter_IsClaudeModel(t *testing.T) {
	adapter := &ClaudeAdapter{}
	
	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		{
			name:     "Claude model",
			model:    "claude-3-sonnet",
			expected: true,
		},
		{
			name:     "Anthropic prefixed model",
			model:    "anthropic.claude-3-haiku-20240307-v1:0",
			expected: true,
		},
		{
			name:     "Direct Anthropic model",
			model:    "anthropic/claude-3-5-sonnet-20241022",
			expected: true,
		},
		{
			name:     "Bedrock Claude model",
			model:    "aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
			expected: true,
		},
		{
			name:     "GPT model",
			model:    "gpt-4",
			expected: false,
		},
		{
			name:     "GPT-3.5 model",
			model:    "gpt-35-turbo",
			expected: false,
		},
		{
			name:     "Gemini model",
			model:    "gemini-1.5-pro",
			expected: false,
		},
		{
			name:     "Empty model",
			model:    "",
			expected: false,
		},
		{
			name:     "Mixed case Claude",
			model:    "CLAUDE-3-OPUS",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.IsClaudeModel(tt.model)
			if result != tt.expected {
				t.Errorf("IsClaudeModel(%q) = %v, want %v", tt.model, result, tt.expected)
			}
		})
	}
}

func TestClaudeAdapter_getClientForModel(t *testing.T) {
	// Create adapter with both clients initialized
	adapter := &ClaudeAdapter{
		hasDirectClient:  true,
		hasBedrockClient: true,
	}
	
	tests := []struct {
		name        string
		model       string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "Direct Anthropic model",
			model:       "anthropic/claude-3-5-sonnet-20241022",
			expectError: false,
		},
		{
			name:        "Bedrock Claude model",
			model:       "aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
			expectError: false,
		},
		{
			name:        "Unsupported prefix",
			model:       "openai/gpt-4",
			expectError: true,
			errorMsg:    "unsupported model prefix",
		},
		{
			name:        "No prefix",
			model:       "claude-3-sonnet",
			expectError: true,
			errorMsg:    "unsupported model prefix",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := adapter.getClientForModel(tt.model)
			
			if tt.expectError {
				if err == nil {
					t.Errorf("getClientForModel(%q) expected error, got nil", tt.model)
				} else if tt.errorMsg != "" && !containsString(err.Error(), tt.errorMsg) {
					t.Errorf("getClientForModel(%q) error = %q, want to contain %q", tt.model, err.Error(), tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("getClientForModel(%q) unexpected error: %v", tt.model, err)
				}
				if client == nil {
					t.Errorf("getClientForModel(%q) returned nil client", tt.model)
				}
			}
		})
	}
}

func TestClaudeAdapter_getClientForModel_MissingClients(t *testing.T) {
	tests := []struct {
		name             string
		hasDirectClient  bool
		hasBedrockClient bool
		model            string
		expectError      bool
		errorMsg         string
	}{
		{
			name:             "Direct client missing",
			hasDirectClient:  false,
			hasBedrockClient: true,
			model:            "anthropic/claude-3-5-sonnet-20241022",
			expectError:      true,
			errorMsg:         "direct Anthropic client not configured",
		},
		{
			name:             "Bedrock client missing",
			hasDirectClient:  true,
			hasBedrockClient: false,
			model:            "aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
			expectError:      true,
			errorMsg:         "Bedrock client not configured",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &ClaudeAdapter{
				hasDirectClient:  tt.hasDirectClient,
				hasBedrockClient: tt.hasBedrockClient,
			}
			
			client, err := adapter.getClientForModel(tt.model)
			
			if tt.expectError {
				if err == nil {
					t.Errorf("getClientForModel(%q) expected error, got nil", tt.model)
				} else if !containsString(err.Error(), tt.errorMsg) {
					t.Errorf("getClientForModel(%q) error = %q, want to contain %q", tt.model, err.Error(), tt.errorMsg)
				}
				if client != nil {
					t.Errorf("getClientForModel(%q) expected nil client on error", tt.model)
				}
			}
		})
	}
}

func TestClaudeAdapter_normalizeModelName(t *testing.T) {
	adapter := &ClaudeAdapter{}
	
	tests := []struct {
		name     string
		model    string
		expected string
	}{
		{
			name:     "Direct Anthropic model",
			model:    "anthropic/claude-3-5-sonnet-20241022",
			expected: "claude-3-5-sonnet-20241022",
		},
		{
			name:     "Bedrock Claude model",
			model:    "aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
			expected: "anthropic.claude-3-sonnet-20240229-v1:0",
		},
		{
			name:     "Model without prefix",
			model:    "claude-3-sonnet",
			expected: "claude-3-sonnet",
		},
		{
			name:     "Empty model",
			model:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.normalizeModelName(tt.model)
			if result != tt.expected {
				t.Errorf("normalizeModelName(%q) = %q, want %q", tt.model, result, tt.expected)
			}
		})
	}
}

func TestClaudeAdapter_ConvertToAnthropicMessages(t *testing.T) {
	adapter := &ClaudeAdapter{}
	
	tests := []struct {
		name              string
		messages          []Message
		expectedMsgCount  int
		expectedSystemMsg string
	}{
		{
			name: "Simple user message",
			messages: []Message{
				{Role: "user", Content: "Hello"},
			},
			expectedMsgCount:  1,
			expectedSystemMsg: "",
		},
		{
			name: "System and user messages",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant"},
				{Role: "user", Content: "Hello"},
			},
			expectedMsgCount:  1,
			expectedSystemMsg: "You are a helpful assistant",
		},
		{
			name: "User and assistant conversation",
			messages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			expectedMsgCount:  3,
			expectedSystemMsg: "",
		},
		{
			name: "Mixed message types",
			messages: []Message{
				{Role: "system", Content: "You are helpful"},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi!"},
				{Role: "function", Content: "Function result"}, // Should be ignored
			},
			expectedMsgCount:  2,
			expectedSystemMsg: "You are helpful",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages, systemMsg := adapter.ConvertToAnthropicMessages(tt.messages)
			
			if len(messages) != tt.expectedMsgCount {
				t.Errorf("ConvertToAnthropicMessages() message count = %d, want %d", len(messages), tt.expectedMsgCount)
			}
			
			if systemMsg != tt.expectedSystemMsg {
				t.Errorf("ConvertToAnthropicMessages() systemMsg = %q, want %q", systemMsg, tt.expectedSystemMsg)
			}
		})
	}
}

func TestClaudeAdapter_ConvertAnthropicToOpenAI(t *testing.T) {
	adapter := &ClaudeAdapter{}
	
	// Create a mock Anthropic response
	anthropicResp := &anthropic.Message{
		ID:         "msg_123",
		StopReason: "end_turn",
		Content: []anthropic.ContentBlockUnion{
			{
				Type: "text",
				Text: "Hello, how can I help you?",
			},
		},
		Usage: anthropic.Usage{
			InputTokens:  10,
			OutputTokens: 15,
		},
	}
	
	result := adapter.ConvertAnthropicToOpenAI(anthropicResp, "claude-3-5-sonnet-20241022")
	
	// Check basic structure
	if result["object"] != "chat.completion" {
		t.Errorf("ConvertAnthropicToOpenAI() object = %v, want %q", result["object"], "chat.completion")
	}
	
	if result["model"] != "claude-3-5-sonnet-20241022" {
		t.Errorf("ConvertAnthropicToOpenAI() model = %v, want %q", result["model"], "claude-3-5-sonnet-20241022")
	}
	
	// Check ID
	if result["id"] != "chatcmpl-msg_123" {
		t.Errorf("ConvertAnthropicToOpenAI() id = %v, want %q", result["id"], "chatcmpl-msg_123")
	}
	
	// Check choices
	choices, ok := result["choices"].([]map[string]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("ConvertAnthropicToOpenAI() choices not found or empty")
	}
	
	message, ok := choices[0]["message"].(map[string]string)
	if !ok {
		t.Fatal("ConvertAnthropicToOpenAI() message not found")
	}
	
	if message["content"] != "Hello, how can I help you?" {
		t.Errorf("ConvertAnthropicToOpenAI() content = %q, want %q", message["content"], "Hello, how can I help you?")
	}
	
	if message["role"] != "assistant" {
		t.Errorf("ConvertAnthropicToOpenAI() role = %q, want %q", message["role"], "assistant")
	}
	
	// Check usage
	usage, ok := result["usage"].(map[string]interface{})
	if !ok {
		t.Fatal("ConvertAnthropicToOpenAI() usage not found")
	}
	
	if usage["prompt_tokens"] != 10 {
		t.Errorf("ConvertAnthropicToOpenAI() prompt_tokens = %v, want %d", usage["prompt_tokens"], 10)
	}
	
	if usage["completion_tokens"] != 15 {
		t.Errorf("ConvertAnthropicToOpenAI() completion_tokens = %v, want %d", usage["completion_tokens"], 15)
	}
	
	if usage["total_tokens"] != 25 {
		t.Errorf("ConvertAnthropicToOpenAI() total_tokens = %v, want %d", usage["total_tokens"], 25)
	}
}

func TestNewClaudeAdapter(t *testing.T) {
	tests := []struct {
		name                 string
		config               ClaudeAdapterConfig
		expectedDirectClient bool
		expectedBedrockClient bool
	}{
		{
			name: "Both clients configured",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "sk-test-key",
				AWSConfig:       &aws.Config{},
			},
			expectedDirectClient:  true,
			expectedBedrockClient: true,
		},
		{
			name: "Only direct client configured",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "sk-test-key",
				AWSConfig:       nil,
			},
			expectedDirectClient:  true,
			expectedBedrockClient: false,
		},
		{
			name: "Only Bedrock client configured",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "",
				AWSConfig:       &aws.Config{},
			},
			expectedDirectClient:  false,
			expectedBedrockClient: true,
		},
		{
			name: "No clients configured",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "",
				AWSConfig:       nil,
			},
			expectedDirectClient:  false,
			expectedBedrockClient: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := NewClaudeAdapter(tt.config)
			
			if adapter.hasDirectClient != tt.expectedDirectClient {
				t.Errorf("NewClaudeAdapter() hasDirectClient = %v, want %v", adapter.hasDirectClient, tt.expectedDirectClient)
			}
			
			if adapter.hasBedrockClient != tt.expectedBedrockClient {
				t.Errorf("NewClaudeAdapter() hasBedrockClient = %v, want %v", adapter.hasBedrockClient, tt.expectedBedrockClient)
			}
		})
	}
}

// Helper function to check if a string contains a substring
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || 
		(len(substr) > 0 && len(s) > len(substr) && 
			(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
				findInString(s, substr))))
}

func findInString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}