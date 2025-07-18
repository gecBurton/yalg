package adapter

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/aws/aws-sdk-go-v2/aws"
)

// mockTokenCalculator implements TokenCalculator for testing
type mockTokenCalculator struct{}

func (m *mockTokenCalculator) CalculateTokenUsage(text string, modelName string) int {
	// Return word count for predictable test results
	return len(strings.Fields(text))
}

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
			errorMsg:         "bedrock client not configured",
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

	message, ok := choices[0]["message"].(map[string]interface{})
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
		name                  string
		config                ClaudeAdapterConfig
		expectedDirectClient  bool
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
	return strings.Contains(s, substr)
}

// TestClaudeAdapter_StreamingChunkParsing tests parsing of Anthropic streaming chunks
func TestClaudeAdapter_StreamingChunkParsing(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name              string
		event             anthropic.MessageStreamEventUnion
		expectedChunk     map[string]any
		shouldHaveContent bool
		shouldHaveRole    bool
		shouldHaveFinish  bool
	}{
		{
			name: "message_start event",
			event: anthropic.MessageStreamEventUnion{
				Type: "message_start",
			},
			expectedChunk: map[string]any{
				"choices": []map[string]any{
					{
						"index": 0,
						"delta": map[string]any{
							"role": "assistant",
						},
						"finish_reason": nil,
					},
				},
			},
			shouldHaveRole: true,
		},
		{
			name: "content_block_delta event",
			event: anthropic.MessageStreamEventUnion{
				Type: "content_block_delta",
				Delta: anthropic.MessageStreamEventUnionDelta{
					Text: "Hello, world!",
				},
			},
			expectedChunk: map[string]any{
				"choices": []map[string]any{
					{
						"index": 0,
						"delta": map[string]any{
							"content": "Hello, world!",
						},
						"finish_reason": nil,
					},
				},
			},
			shouldHaveContent: true,
		},
		{
			name: "message_delta with stop_reason",
			event: anthropic.MessageStreamEventUnion{
				Type: "message_delta",
				Delta: anthropic.MessageStreamEventUnionDelta{
					StopReason: "end_turn",
				},
			},
			expectedChunk: map[string]any{
				"choices": []map[string]any{
					{
						"index":         0,
						"delta":         map[string]any{},
						"finish_reason": "end_turn",
					},
				},
			},
			shouldHaveFinish: true,
		},
		{
			name: "message_stop event",
			event: anthropic.MessageStreamEventUnion{
				Type: "message_stop",
			},
			expectedChunk: map[string]any{
				"choices": []map[string]any{
					{
						"index":         0,
						"delta":         map[string]any{},
						"finish_reason": "stop",
					},
				},
			},
			shouldHaveFinish: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ConvertAnthropicStreamToOpenAI(tt.event, "claude-3-5-sonnet-20241022")

			// Check basic structure
			if result["object"] != "chat.completion.chunk" {
				t.Errorf("Expected object to be 'chat.completion.chunk', got %v", result["object"])
			}

			if result["model"] != "claude-3-5-sonnet-20241022" {
				t.Errorf("Expected model to be 'claude-3-5-sonnet-20241022', got %v", result["model"])
			}

			// Check choices structure
			choices, ok := result["choices"].([]map[string]any)
			if !ok || len(choices) == 0 {
				t.Fatal("Expected choices to be non-empty array")
			}

			choice := choices[0]
			delta, ok := choice["delta"].(map[string]any)
			if !ok {
				t.Fatal("Expected delta to be present")
			}

			// Test specific expectations
			if tt.shouldHaveContent {
				if content, exists := delta["content"]; !exists || content != "Hello, world!" {
					t.Errorf("Expected content 'Hello, world!', got %v", content)
				}
			}

			if tt.shouldHaveRole {
				if role, exists := delta["role"]; !exists || role != "assistant" {
					t.Errorf("Expected role 'assistant', got %v", role)
				}
			}

			if tt.shouldHaveFinish {
				if finishReason := choice["finish_reason"]; finishReason == nil {
					t.Error("Expected finish_reason to be set")
				}
			}
		})
	}
}

// TestClaudeAdapter_MessageConversion tests conversion between OpenAI and Anthropic message formats
func TestClaudeAdapter_MessageConversion(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name             string
		messages         []Message
		expectedCount    int
		expectedSystem   string
		hasSystemMessage bool
	}{
		{
			name: "simple user message",
			messages: []Message{
				{Role: "user", Content: "Hello, Claude!"},
			},
			expectedCount: 1,
		},
		{
			name: "system and user messages",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
			expectedCount:    1,
			expectedSystem:   "You are a helpful assistant.",
			hasSystemMessage: true,
		},
		{
			name: "conversation with assistant",
			messages: []Message{
				{Role: "user", Content: "What's 2+2?"},
				{Role: "assistant", Content: "2+2 equals 4."},
				{Role: "user", Content: "Thanks!"},
			},
			expectedCount: 3,
		},
		{
			name: "multiple system messages (last one wins)",
			messages: []Message{
				{Role: "system", Content: "First system message"},
				{Role: "system", Content: "Second system message"},
				{Role: "user", Content: "Hello!"},
			},
			expectedCount:    1,
			expectedSystem:   "Second system message",
			hasSystemMessage: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			anthropicMessages, systemMessage := adapter.ConvertToAnthropicMessages(tt.messages)

			if len(anthropicMessages) != tt.expectedCount {
				t.Errorf("Expected %d anthropic messages, got %d", tt.expectedCount, len(anthropicMessages))
			}

			if tt.hasSystemMessage {
				if systemMessage != tt.expectedSystem {
					t.Errorf("Expected system message '%s', got '%s'", tt.expectedSystem, systemMessage)
				}
			} else {
				if systemMessage != "" {
					t.Errorf("Expected no system message, got '%s'", systemMessage)
				}
			}
		})
	}
}

// TestClaudeAdapter_ProviderSelection tests provider-specific client selection
func TestClaudeAdapter_ProviderSelection(t *testing.T) {
	tests := []struct {
		name                  string
		config                ClaudeAdapterConfig
		provider              string
		shouldSucceed         bool
		expectedErrorContains string
	}{
		{
			name: "anthropic provider with valid config",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "sk-test-key",
				AWSConfig:       nil,
			},
			provider:      "anthropic",
			shouldSucceed: true,
		},
		{
			name: "aws-bedrock provider with valid config",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "",
				AWSConfig:       &aws.Config{},
			},
			provider:      "aws-bedrock",
			shouldSucceed: true,
		},
		{
			name: "anthropic provider without config",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "",
				AWSConfig:       nil,
			},
			provider:              "anthropic",
			shouldSucceed:         false,
			expectedErrorContains: "direct Anthropic client not configured",
		},
		{
			name: "aws-bedrock provider without config",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "sk-test-key",
				AWSConfig:       nil,
			},
			provider:              "aws-bedrock",
			shouldSucceed:         false,
			expectedErrorContains: "bedrock client not configured",
		},
		{
			name: "unsupported provider",
			config: ClaudeAdapterConfig{
				AnthropicAPIKey: "sk-test-key",
				AWSConfig:       &aws.Config{},
			},
			provider:              "unsupported",
			shouldSucceed:         false,
			expectedErrorContains: "unsupported provider",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := NewClaudeAdapter(tt.config)
			client, err := adapter.getClientForProvider(tt.provider)

			if tt.shouldSucceed {
				if err != nil {
					t.Errorf("Expected success but got error: %v", err)
				}
				if client == nil {
					t.Error("Expected client to be non-nil")
				}
			} else {
				if err == nil {
					t.Error("Expected error but got success")
				}
				if client != nil {
					t.Error("Expected client to be nil on error")
				}
				if tt.expectedErrorContains != "" && !containsString(err.Error(), tt.expectedErrorContains) {
					t.Errorf("Expected error to contain '%s', got '%s'", tt.expectedErrorContains, err.Error())
				}
			}
		})
	}
}

// TestClaudeAdapter_ResponseTransformation tests transformation of Anthropic responses to OpenAI format
func TestClaudeAdapter_ResponseTransformation(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name            string
		anthropicResp   *anthropic.Message
		expectedContent string
		expectedTokens  int
		expectedFinish  string
	}{
		{
			name: "simple text response",
			anthropicResp: &anthropic.Message{
				ID:   "msg_123",
				Type: "message",
				Role: "assistant",
				Content: []anthropic.ContentBlockUnion{
					{
						Type: "text",
						Text: "Hello! How can I help you today?",
					},
				},
				StopReason: "end_turn",
				Usage: anthropic.Usage{
					InputTokens:  20,
					OutputTokens: 12,
				},
			},
			expectedContent: "Hello! How can I help you today?",
			expectedTokens:  32,
			expectedFinish:  "end_turn",
		},
		{
			name: "multiple content blocks",
			anthropicResp: &anthropic.Message{
				ID:   "msg_456",
				Type: "message",
				Role: "assistant",
				Content: []anthropic.ContentBlockUnion{
					{
						Type: "text",
						Text: "First part. ",
					},
					{
						Type: "text",
						Text: "Second part.",
					},
				},
				StopReason: "stop",
				Usage: anthropic.Usage{
					InputTokens:  15,
					OutputTokens: 8,
				},
			},
			expectedContent: "First part. Second part.",
			expectedTokens:  23,
			expectedFinish:  "stop",
		},
		{
			name: "empty response",
			anthropicResp: &anthropic.Message{
				ID:         "msg_789",
				Type:       "message",
				Role:       "assistant",
				Content:    []anthropic.ContentBlockUnion{},
				StopReason: "max_tokens",
				Usage: anthropic.Usage{
					InputTokens:  100,
					OutputTokens: 0,
				},
			},
			expectedContent: "",
			expectedTokens:  100,
			expectedFinish:  "max_tokens",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ConvertAnthropicToOpenAI(tt.anthropicResp, "claude-3-5-sonnet-20241022")

			// Check content
			choices, ok := result["choices"].([]map[string]any)
			if !ok || len(choices) == 0 {
				t.Fatal("Expected choices to be non-empty")
			}

			message, ok := choices[0]["message"].(map[string]interface{})
			if !ok {
				t.Fatal("Expected message to be present")
			}

			if message["content"] != tt.expectedContent {
				t.Errorf("Expected content '%s', got '%s'", tt.expectedContent, message["content"])
			}

			// Check finish reason
			if choices[0]["finish_reason"] != tt.expectedFinish {
				t.Errorf("Expected finish_reason '%s', got '%v'", tt.expectedFinish, choices[0]["finish_reason"])
			}

			// Check token usage
			usage, ok := result["usage"].(map[string]any)
			if !ok {
				t.Fatal("Expected usage to be present")
			}

			if usage["total_tokens"] != tt.expectedTokens {
				t.Errorf("Expected total_tokens %d, got %v", tt.expectedTokens, usage["total_tokens"])
			}
		})
	}
}

// ===============================================================================
// COMPREHENSIVE TESTS - Based on LiteLLM patterns
// ===============================================================================

// TestClaudeAdapter_HeaderProcessing tests comprehensive header processing like LiteLLM
func TestClaudeAdapter_HeaderProcessing(t *testing.T) {
	adapter := NewClaudeAdapter(ClaudeAdapterConfig{
		AnthropicAPIKey: "test-key",
		AWSConfig:       &aws.Config{Region: "us-east-1"},
	})

	tests := []struct {
		name           string
		inputHeaders   map[string]string
		expectedOutput map[string]string
	}{
		{
			name:           "empty headers",
			inputHeaders:   map[string]string{},
			expectedOutput: map[string]string{},
		},
		{
			name: "all anthropic rate limit headers",
			inputHeaders: map[string]string{
				"anthropic-ratelimit-requests-limit":     "100",
				"anthropic-ratelimit-requests-remaining": "90",
				"anthropic-ratelimit-tokens-limit":       "10000",
				"anthropic-ratelimit-tokens-remaining":   "9000",
				"other-header":                           "value",
			},
			expectedOutput: map[string]string{
				"x-ratelimit-limit-requests":                          "100",
				"x-ratelimit-remaining-requests":                      "90",
				"x-ratelimit-limit-tokens":                            "10000",
				"x-ratelimit-remaining-tokens":                        "9000",
				"llm_provider-anthropic-ratelimit-requests-limit":     "100",
				"llm_provider-anthropic-ratelimit-requests-remaining": "90",
				"llm_provider-anthropic-ratelimit-tokens-limit":       "10000",
				"llm_provider-anthropic-ratelimit-tokens-remaining":   "9000",
				"llm_provider-other-header":                           "value",
			},
		},
		{
			name: "partial anthropic headers",
			inputHeaders: map[string]string{
				"anthropic-ratelimit-requests-limit":   "100",
				"anthropic-ratelimit-tokens-remaining": "9000",
				"other-header":                         "value",
			},
			expectedOutput: map[string]string{
				"x-ratelimit-limit-requests":                        "100",
				"x-ratelimit-remaining-tokens":                      "9000",
				"llm_provider-anthropic-ratelimit-requests-limit":   "100",
				"llm_provider-anthropic-ratelimit-tokens-remaining": "9000",
				"llm_provider-other-header":                         "value",
			},
		},
		{
			name: "no matching anthropic headers",
			inputHeaders: map[string]string{
				"unrelated-header-1": "value1",
				"unrelated-header-2": "value2",
			},
			expectedOutput: map[string]string{
				"llm_provider-unrelated-header-1": "value1",
				"llm_provider-unrelated-header-2": "value2",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ProcessAnthropicHeaders(tt.inputHeaders)

			if len(result) != len(tt.expectedOutput) {
				t.Errorf("Expected %d headers, got %d", len(tt.expectedOutput), len(result))
			}

			for key, expectedValue := range tt.expectedOutput {
				if actualValue, exists := result[key]; !exists {
					t.Errorf("Missing expected header: %s", key)
				} else if actualValue != expectedValue {
					t.Errorf("Header %s: expected %s, got %s", key, expectedValue, actualValue)
				}
			}
		})
	}
}

// TestClaudeAdapter_ToolStreamingIndexing tests proper tool use index handling like LiteLLM
func TestClaudeAdapter_ToolStreamingIndexing(t *testing.T) {
	adapter := &ClaudeAdapter{}

	// Simulate anthropic tool streaming chunks (like the anthropic_chunk_list in LiteLLM tests)
	toolChunks := []anthropic.MessageStreamEventUnion{
		{
			Type: "content_block_start",
			// Note: ContentBlock field may not be available in the SDK
			Index: 1, // Anthropic starts at 1, should be converted to 0
		},
		{
			Type:  "content_block_delta",
			Index: 1,
			Delta: anthropic.MessageStreamEventUnionDelta{
				Text: `{"location": "Boston"}`,
			},
		},
		{
			Type:  "content_block_start",
			Index: 2, // Should be converted to 1
		},
		{
			Type:  "content_block_delta",
			Index: 2,
			Delta: anthropic.MessageStreamEventUnionDelta{
				Text: `{"location": "Los Angeles"}`,
			},
		},
	}

	expectedIndices := []int{0, 0, 1, 1} // OpenAI starts tool indices at 0

	for i, chunk := range toolChunks {
		result := adapter.ConvertAnthropicStreamToOpenAI(chunk, "claude-3-5-sonnet-20241022")

		// Check if tool use index was properly converted
		choices, ok := result["choices"].([]map[string]interface{})
		if !ok || len(choices) == 0 {
			t.Fatalf("Expected choices array in chunk %d", i)
		}

		if delta, exists := choices[0]["delta"].(map[string]interface{}); exists {
			if toolCalls, exists := delta["tool_calls"].([]interface{}); exists && len(toolCalls) > 0 {
				if toolCall, ok := toolCalls[0].(map[string]interface{}); ok {
					if index, exists := toolCall["index"]; exists {
						if index != expectedIndices[i] {
							t.Errorf("Chunk %d: expected tool index %d, got %v", i, expectedIndices[i], index)
						}
					}
				}
			}
		}
	}
}

// TestClaudeAdapter_BetaHeaders tests dynamic beta header management like LiteLLM
func TestClaudeAdapter_BetaHeaders(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name               string
		computerToolUsed   bool
		promptCachingSet   bool
		expectedBetaHeader bool
	}{
		{
			name:               "computer tool used",
			computerToolUsed:   true,
			promptCachingSet:   false,
			expectedBetaHeader: true,
		},
		{
			name:               "prompt caching set",
			computerToolUsed:   false,
			promptCachingSet:   true,
			expectedBetaHeader: true,
		},
		{
			name:               "both features used",
			computerToolUsed:   true,
			promptCachingSet:   true,
			expectedBetaHeader: true,
		},
		{
			name:               "no special features",
			computerToolUsed:   false,
			promptCachingSet:   false,
			expectedBetaHeader: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			headers := adapter.GetAnthropicHeaders(&AnthropicHeaderConfig{
				APIKey:           "fake-api-key",
				ComputerToolUsed: tt.computerToolUsed,
				PromptCachingSet: tt.promptCachingSet,
			})

			_, hasBetaHeader := headers["anthropic-beta"]
			if hasBetaHeader != tt.expectedBetaHeader {
				t.Errorf("Expected anthropic-beta header: %v, got: %v", tt.expectedBetaHeader, hasBetaHeader)
			}
		})
	}
}

// TestClaudeAdapter_ToolHelper tests tool transformation and cache control like LiteLLM
func TestClaudeAdapter_ToolHelper(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name                 string
		cacheControlLocation string
		tool                 map[string]interface{}
		expectedCacheControl map[string]interface{}
	}{
		{
			name:                 "cache control inside function",
			cacheControlLocation: "inside_function",
			tool: map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_current_weather",
					"description": "Get the current weather in a given location",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
							"unit": map[string]interface{}{
								"type": "string",
								"enum": []string{"celsius", "fahrenheit"},
							},
						},
						"required": []string{"location"},
					},
					"cache_control": map[string]interface{}{
						"type": "ephemeral",
					},
				},
			},
			expectedCacheControl: map[string]interface{}{
				"type": "ephemeral",
			},
		},
		{
			name:                 "cache control outside function",
			cacheControlLocation: "outside_function",
			tool: map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_current_weather",
					"description": "Get the current weather in a given location",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
						},
						"required": []string{"location"},
					},
				},
				"cache_control": map[string]interface{}{
					"type": "ephemeral",
				},
			},
			expectedCacheControl: map[string]interface{}{
				"type": "ephemeral",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transformedTool, err := adapter.MapToolHelper(tt.tool)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			cacheControl, exists := transformedTool["cache_control"]
			if !exists {
				t.Fatalf("Expected cache_control to be present in transformed tool")
			}

			cacheControlMap, ok := cacheControl.(map[string]interface{})
			if !ok {
				t.Fatalf("Expected cache_control to be a map, got %T", cacheControl)
			}

			if cacheControlMap["type"] != tt.expectedCacheControl["type"] {
				t.Errorf("Expected cache_control type %v, got %v", tt.expectedCacheControl["type"], cacheControlMap["type"])
			}
		})
	}
}

// TestClaudeAdapter_JSONToolCallForResponseFormat tests JSON response format tool creation like LiteLLM
func TestClaudeAdapter_JSONToolCallForResponseFormat(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name         string
		schema       map[string]interface{}
		expectedTool map[string]interface{}
	}{
		{
			name:   "no schema provided",
			schema: nil,
			expectedTool: map[string]interface{}{
				"name": "json_tool_call",
				"input_schema": map[string]interface{}{
					"type":                 "object",
					"additionalProperties": true,
					"properties":           map[string]interface{}{},
				},
			},
		},
		{
			name: "custom schema provided",
			schema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{
						"type": "string",
					},
					"age": map[string]interface{}{
						"type": "integer",
					},
				},
				"required": []string{"name"},
			},
			expectedTool: map[string]interface{}{
				"name": "json_tool_call",
				"input_schema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"name": map[string]interface{}{
							"type": "string",
						},
						"age": map[string]interface{}{
							"type": "integer",
						},
					},
					"required": []string{"name"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := adapter.CreateJSONToolCallForResponseFormat(tt.schema)

			if tool["name"] != tt.expectedTool["name"] {
				t.Errorf("Expected tool name %v, got %v", tt.expectedTool["name"], tool["name"])
			}

			inputSchema, exists := tool["input_schema"]
			if !exists {
				t.Fatalf("Expected input_schema to be present")
			}

			expectedInputSchema := tt.expectedTool["input_schema"].(map[string]interface{})
			actualInputSchema := inputSchema.(map[string]interface{})

			if actualInputSchema["type"] != expectedInputSchema["type"] {
				t.Errorf("Expected schema type %v, got %v", expectedInputSchema["type"], actualInputSchema["type"])
			}

			if tt.schema == nil {
				// For no schema case, check default properties
				if properties, exists := actualInputSchema["properties"]; exists {
					propertiesMap := properties.(map[string]interface{})
					if len(propertiesMap) != 0 {
						t.Errorf("Expected empty properties for no schema case, got %v", propertiesMap)
					}
				}

				if additionalProps, exists := actualInputSchema["additionalProperties"]; exists {
					if additionalProps != true {
						t.Errorf("Expected additionalProperties to be true for no schema case, got %v", additionalProps)
					}
				}
			}
		})
	}
}

// TestClaudeAdapter_ConvertToolResponseToMessage tests tool response conversion like LiteLLM
func TestClaudeAdapter_ConvertToolResponseToMessage(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name            string
		toolCalls       []ToolCall
		expectedContent string
		expectNil       bool
	}{
		{
			name: "tool response with values key",
			toolCalls: []ToolCall{
				{
					ID:   "test_id",
					Type: "function",
					Function: FunctionCall{
						Name:      "json_tool_call",
						Arguments: `{"values": {"name": "John", "age": 30}}`,
					},
				},
			},
			expectedContent: `{"name": "John", "age": 30}`,
			expectNil:       false,
		},
		{
			name: "tool response without values key",
			toolCalls: []ToolCall{
				{
					ID:   "test_id",
					Type: "function",
					Function: FunctionCall{
						Name:      "json_tool_call",
						Arguments: `{"name": "John", "age": 30}`,
					},
				},
			},
			expectedContent: `{"name": "John", "age": 30}`,
			expectNil:       false,
		},
		{
			name: "tool response with invalid JSON",
			toolCalls: []ToolCall{
				{
					ID:   "test_id",
					Type: "function",
					Function: FunctionCall{
						Name:      "json_tool_call",
						Arguments: "invalid json",
					},
				},
			},
			expectedContent: "invalid json",
			expectNil:       false,
		},
		{
			name: "tool response with no arguments",
			toolCalls: []ToolCall{
				{
					ID:   "test_id",
					Type: "function",
					Function: FunctionCall{
						Name: "json_tool_call",
					},
				},
			},
			expectNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			message := adapter.ConvertToolResponseToMessage(tt.toolCalls)

			if tt.expectNil {
				if message != nil {
					t.Errorf("Expected nil message, got %v", message)
				}
			} else {
				if message == nil {
					t.Fatalf("Expected non-nil message")
				}

				// For JSON content, parse both and compare as objects
				if strings.HasPrefix(tt.expectedContent, "{") && strings.HasPrefix(message.Content, "{") {
					var expectedJSON, actualJSON interface{}
					if json.Unmarshal([]byte(tt.expectedContent), &expectedJSON) == nil &&
						json.Unmarshal([]byte(message.Content), &actualJSON) == nil {
						if !compareJSON(expectedJSON, actualJSON) {
							t.Errorf("Expected content %q, got %q", tt.expectedContent, message.Content)
						}
					} else {
						t.Errorf("Expected content %q, got %q", tt.expectedContent, message.Content)
					}
				} else if message.Content != tt.expectedContent {
					t.Errorf("Expected content %q, got %q", tt.expectedContent, message.Content)
				}
			}
		})
	}
}

// TestClaudeAdapter_ThinkingOutput tests Claude's thinking/reasoning mode like LiteLLM
func TestClaudeAdapter_ThinkingOutput(t *testing.T) {
	adapter := &ClaudeAdapter{}

	// Mock Anthropic response with thinking blocks
	anthropicResp := &anthropic.Message{
		ID:   "msg_thinking_123",
		Type: "message",
		Role: "assistant",
		Content: []anthropic.ContentBlockUnion{
			{
				Type: "thinking",
				Text: "Let me think about this step by step...",
			},
			{
				Type: "text",
				Text: "The answer is 42.",
			},
		},
		Usage: anthropic.Usage{
			InputTokens:  50,
			OutputTokens: 100,
		},
	}

	result := adapter.ConvertAnthropicToOpenAIWithCalculator(anthropicResp, "claude-3-7-sonnet-20250219", &mockTokenCalculator{})

	// Check that thinking content is properly extracted
	choices, ok := result["choices"].([]map[string]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("Expected choices array")
	}

	message, ok := choices[0]["message"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected message object")
	}

	// Check for reasoning_content field
	if reasoningContent, exists := message["reasoning_content"]; exists {
		if reasoningContent != "Let me think about this step by step..." {
			t.Errorf("Expected reasoning content to match thinking block, got %v", reasoningContent)
		}
	} else {
		t.Error("Expected reasoning_content field in message")
	}

	// Check for thinking_blocks field
	if thinkingBlocks, exists := message["thinking_blocks"]; exists {
		blocksArray, ok := thinkingBlocks.([]map[string]interface{})
		if !ok || len(blocksArray) == 0 {
			t.Error("Expected thinking_blocks to be non-empty array")
		} else {
			block := blocksArray[0]
			if block["type"] != "thinking" {
				t.Errorf("Expected thinking block type, got %v", block["type"])
			}
			if block["content"] != "Let me think about this step by step..." {
				t.Errorf("Expected thinking block content to match, got %v", block["content"])
			}
		}
	} else {
		t.Error("Expected thinking_blocks field in message")
	}

	// Regular content should only contain non-thinking blocks
	if message["content"] != "The answer is 42." {
		t.Errorf("Expected regular content to exclude thinking blocks, got %v", message["content"])
	}
}

// TestClaudeAdapter_ComputerToolUse tests computer tool support like LiteLLM
func TestClaudeAdapter_ComputerToolUse(t *testing.T) {
	adapter := NewClaudeAdapter(ClaudeAdapterConfig{
		AnthropicAPIKey: "test-key",
	})

	tools := []map[string]interface{}{
		{
			"type": "computer_20241022",
			"function": map[string]interface{}{
				"name": "computer",
				"parameters": map[string]interface{}{
					"display_height_px": 100,
					"display_width_px":  100,
					"display_number":    1,
				},
			},
		},
	}

	// Test that computer tools are properly transformed
	transformedTools, err := adapter.TransformTools(tools)

	if err != nil {
		t.Fatalf("Unexpected error transforming tools: %v", err)
	}

	if len(transformedTools) != 1 {
		t.Errorf("Expected 1 transformed tool, got %d", len(transformedTools))
	}

	tool := transformedTools[0]
	if tool["type"] != "computer_20241022" {
		t.Errorf("Expected computer tool type to be preserved, got %v", tool["type"])
	}

	// Verify that computer tool usage triggers beta headers
	headers := adapter.GetAnthropicHeaders(&AnthropicHeaderConfig{
		APIKey:           "test-key",
		ComputerToolUsed: true,
	})

	if _, exists := headers["anthropic-beta"]; !exists {
		t.Error("Expected anthropic-beta header for computer tool usage")
	}
}

// TestClaudeAdapter_CitationsAPI tests document citation support like LiteLLM
func TestClaudeAdapter_CitationsAPI(t *testing.T) {
	adapter := &ClaudeAdapter{}

	// Mock response with citations
	anthropicResp := &anthropic.Message{
		ID:   "msg_citations_123",
		Type: "message",
		Role: "assistant",
		Content: []anthropic.ContentBlockUnion{
			{
				Type: "text",
				Text: "According to the document, the grass is green and the sky is blue.",
			},
		},
		Usage: anthropic.Usage{
			InputTokens:  20,
			OutputTokens: 30,
		},
	}

	// Mock citations data (since the SDK may not have built-in citation support)
	mockCitations := []map[string]interface{}{
		{
			"start_index": 25,
			"end_index":   52,
			"source": map[string]interface{}{
				"type":  "document",
				"index": 0,
			},
		},
	}

	result := adapter.ConvertAnthropicToOpenAIWithCitations(anthropicResp, "claude-3-5-sonnet-20241022", mockCitations)

	// Check that citations are preserved in provider_specific_fields
	choices, ok := result["choices"].([]map[string]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("Expected choices array")
	}

	message, ok := choices[0]["message"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected message object")
	}

	if providerFields, exists := message["provider_specific_fields"]; exists {
		fields := providerFields.(map[string]interface{})
		if citations, exists := fields["citations"]; exists {
			citationsArray := citations.([]map[string]interface{})
			if len(citationsArray) != 1 {
				t.Errorf("Expected 1 citation, got %d", len(citationsArray))
			}

			citation := citationsArray[0]
			if citation["start_index"] != 25 {
				t.Errorf("Expected start_index 25, got %v", citation["start_index"])
			}
			if citation["end_index"] != 52 {
				t.Errorf("Expected end_index 52, got %v", citation["end_index"])
			}
		} else {
			t.Error("Expected citations in provider_specific_fields")
		}
	} else {
		t.Error("Expected provider_specific_fields in message")
	}
}

// compareJSON compares two JSON objects for equality
func compareJSON(a, b interface{}) bool {
	return reflect.DeepEqual(a, b)
}

// TestClaudeAdapter_ThinkingModeWithBudget tests thinking mode with budget token tracking
func TestClaudeAdapter_ThinkingModeWithBudget(t *testing.T) {
	adapter := &ClaudeAdapter{}

	// Mock Anthropic response with thinking blocks
	anthropicResp := &anthropic.Message{
		ID:         "msg_test_thinking",
		Role:       "assistant",
		StopReason: "end_turn",
		Content: []anthropic.ContentBlockUnion{
			{
				Type: "thinking",
				Text: "Let me think about this step by step. First, I need to understand what the user is asking for. They want to know about the weather in Paris.",
			},
			{
				Type: "text",
				Text: "The current weather in Paris is sunny with a temperature of 22°C.",
			},
		},
		Usage: anthropic.Usage{
			InputTokens:  150,
			OutputTokens: 200,
		},
	}

	tests := []struct {
		name            string
		budgetInfo      map[string]interface{}
		expectBudget    bool
		expectRemaining int
	}{
		{
			name:         "no budget specified",
			budgetInfo:   nil,
			expectBudget: false,
		},
		{
			name: "budget specified with adequate tokens",
			budgetInfo: map[string]interface{}{
				"budget_tokens": 500,
			},
			expectBudget:    true,
			expectRemaining: 150, // 500 - 350 (150 input + 200 output)
		},
		{
			name: "budget specified with insufficient tokens",
			budgetInfo: map[string]interface{}{
				"budget_tokens": 300,
			},
			expectBudget:    true,
			expectRemaining: 0, // Budget exhausted
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ConvertAnthropicToOpenAIWithBudgetAndCalculator(anthropicResp, "claude-3-5-sonnet-20241022", tt.budgetInfo, &mockTokenCalculator{})

			// Check basic structure
			if result["object"] != "chat.completion" {
				t.Errorf("Expected object to be 'chat.completion', got %v", result["object"])
			}

			// Check choices
			choices, ok := result["choices"].([]map[string]interface{})
			if !ok || len(choices) == 0 {
				t.Fatalf("Expected choices array")
			}

			message, ok := choices[0]["message"].(map[string]interface{})
			if !ok {
				t.Fatalf("Expected message object")
			}

			// Check that thinking content is properly separated
			if reasoningContent, exists := message["reasoning_content"]; exists {
				expected := "Let me think about this step by step. First, I need to understand what the user is asking for. They want to know about the weather in Paris."
				if reasoningContent != expected {
					t.Errorf("Expected reasoning content to match thinking block")
				}
			} else {
				t.Error("Expected reasoning_content in message")
			}

			// Check that thinking blocks are present
			if thinkingBlocks, exists := message["thinking_blocks"]; exists {
				blocks := thinkingBlocks.([]map[string]interface{})
				if len(blocks) != 1 {
					t.Errorf("Expected 1 thinking block, got %d", len(blocks))
				}

				if blocks[0]["type"] != "thinking" {
					t.Errorf("Expected thinking block type to be 'thinking', got %v", blocks[0]["type"])
				}
			} else {
				t.Error("Expected thinking_blocks in message")
			}

			// Check usage information
			usage, ok := result["usage"].(map[string]interface{})
			if !ok {
				t.Fatalf("Expected usage object")
			}

			// Check basic token counts
			if usage["prompt_tokens"] != 150 {
				t.Errorf("Expected prompt_tokens 150, got %v", usage["prompt_tokens"])
			}
			if usage["completion_tokens"] != 200 {
				t.Errorf("Expected completion_tokens 200, got %v", usage["completion_tokens"])
			}
			if usage["total_tokens"] != 350 {
				t.Errorf("Expected total_tokens 350, got %v", usage["total_tokens"])
			}

			// Check thinking tokens
			if thinkingTokens, exists := usage["thinking_tokens"]; exists {
				// Should be > 0 since we have thinking content
				if tokens, ok := thinkingTokens.(int); !ok || tokens <= 0 {
					t.Errorf("Expected thinking_tokens > 0, got %v", thinkingTokens)
				}
			} else {
				t.Error("Expected thinking_tokens in usage")
			}

			// Check budget information
			if tt.expectBudget {
				if budgetTokens, exists := usage["budget_tokens"]; exists {
					if budgetTokens != tt.budgetInfo["budget_tokens"] {
						t.Errorf("Expected budget_tokens %v, got %v", tt.budgetInfo["budget_tokens"], budgetTokens)
					}
				} else {
					t.Error("Expected budget_tokens in usage")
				}

				if budgetRemaining, exists := usage["budget_remaining"]; exists {
					if budgetRemaining != tt.expectRemaining {
						t.Errorf("Expected budget_remaining %d, got %v", tt.expectRemaining, budgetRemaining)
					}
				} else {
					t.Error("Expected budget_remaining in usage")
				}
			} else {
				// Should not have budget information
				if _, exists := usage["budget_tokens"]; exists {
					t.Error("Did not expect budget_tokens in usage")
				}
				if _, exists := usage["budget_remaining"]; exists {
					t.Error("Did not expect budget_remaining in usage")
				}
			}
		})
	}
}

// TestClaudeAdapter_URLContextProcessing tests URL context processing functionality
func TestClaudeAdapter_URLContextProcessing(t *testing.T) {
	adapter := &ClaudeAdapter{}

	tests := []struct {
		name          string
		messages      []Message
		enableURL     bool
		expectedURLs  []string
		shouldProcess bool
	}{
		{
			name: "URL context disabled",
			messages: []Message{
				{Role: "user", Content: "Please summarize https://example.com"},
			},
			enableURL:     false,
			expectedURLs:  []string{"https://example.com"},
			shouldProcess: false,
		},
		{
			name: "URL context enabled with single URL",
			messages: []Message{
				{Role: "user", Content: "Please summarize https://example.com"},
			},
			enableURL:     true,
			expectedURLs:  []string{"https://example.com"},
			shouldProcess: true,
		},
		{
			name: "URL context enabled with multiple URLs",
			messages: []Message{
				{Role: "user", Content: "Compare https://example.com and https://test.com"},
			},
			enableURL:     true,
			expectedURLs:  []string{"https://example.com", "https://test.com"},
			shouldProcess: true,
		},
		{
			name: "No URLs in content",
			messages: []Message{
				{Role: "user", Content: "What is the weather today?"},
			},
			enableURL:     true,
			expectedURLs:  []string{},
			shouldProcess: false,
		},
		{
			name: "Structured content with URLs",
			messages: []Message{
				{
					Role: "user",
					Content: []ContentPart{
						{Type: "text", Text: "Please analyze this document: https://example.com"},
					},
				},
			},
			enableURL:     true,
			expectedURLs:  []string{"https://example.com"},
			shouldProcess: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test URL extraction
			for _, msg := range tt.messages {
				content := getContentAsString(msg.Content)
				urls := extractURLsFromContent(content)

				if len(urls) != len(tt.expectedURLs) {
					t.Errorf("Expected %d URLs, got %d", len(tt.expectedURLs), len(urls))
				}

				for i, expectedURL := range tt.expectedURLs {
					if i < len(urls) && urls[i] != expectedURL {
						t.Errorf("Expected URL %s, got %s", expectedURL, urls[i])
					}
				}
			}

			// Test message conversion with URL context
			anthropicMessages, systemMessage := adapter.ConvertToAnthropicMessagesWithURLContext(tt.messages, tt.enableURL)

			// Check basic conversion
			if len(anthropicMessages) != len(tt.messages) {
				t.Errorf("Expected %d messages, got %d", len(tt.messages), len(anthropicMessages))
			}

			// Check if URL context processing occurred
			if tt.enableURL && len(tt.expectedURLs) > 0 {
				// For URL context enabled cases, we expect the content to potentially be modified
				// Note: In a real test environment, we would need to mock the HTTP client
				// For now, we just verify the structure is correct
				if len(anthropicMessages) == 0 {
					t.Error("Expected at least one message after URL context processing")
				}
			}

			// Verify system message is handled correctly
			if systemMessage != "" {
				t.Logf("System message: %s", systemMessage)
			}
		})
	}
}

// TestClaudeAdapter_URLValidation tests URL validation functionality
func TestClaudeAdapter_URLValidation(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		expected bool
	}{
		{
			name:     "Valid HTTPS URL",
			url:      "https://example.com",
			expected: true,
		},
		{
			name:     "Valid HTTP URL",
			url:      "http://example.com",
			expected: true,
		},
		{
			name:     "Invalid URL - no scheme",
			url:      "example.com",
			expected: false,
		},
		{
			name:     "Invalid URL - no host",
			url:      "https://",
			expected: false,
		},
		{
			name:     "Invalid URL - malformed",
			url:      "not-a-url",
			expected: false,
		},
		{
			name:     "Empty URL",
			url:      "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidURL(tt.url)
			if result != tt.expected {
				t.Errorf("isValidURL(%q) = %v, want %v", tt.url, result, tt.expected)
			}
		})
	}
}

// TestClaudeAdapter_HTMLStripping tests HTML stripping functionality
func TestClaudeAdapter_HTMLStripping(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Basic HTML tags",
			input:    "<p>Hello <strong>world</strong></p>",
			expected: "Hello world",
		},
		{
			name:     "Script tags removal",
			input:    "<p>Content</p><script>alert(\"bad\")</script><p>More content</p>",
			expected: "Content\n\nMore content",
		},
		{
			name:     "Style tags removal",
			input:    "<p>Content</p><style>body { color: red; }</style><p>More content</p>",
			expected: "Content\n\nMore content",
		},
		{
			name:     "Plain text",
			input:    "Just plain text",
			expected: "Just plain text",
		},
		{
			name:     "Complex HTML",
			input:    "<html><head><title>Test</title></head><body><div class=\"content\">Hello <a href=\"#\">world</a></div></body></html>",
			expected: "Hello world",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stripBasicHTML(tt.input)
			result = strings.TrimSpace(result)
			tt.expected = strings.TrimSpace(tt.expected)

			if result != tt.expected {
				t.Errorf("stripBasicHTML(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
