package adapter

import (
	"reflect"
	"testing"
)

func TestClaudeAdapter_IsAnthropicModel(t *testing.T) {
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
			result := adapter.IsAnthropicModel(tt.model)
			if result != tt.expected {
				t.Errorf("IsAnthropicModel(%q) = %v, want %v", tt.model, result, tt.expected)
			}
		})
	}
}

func TestClaudeAdapter_ConvertToAnthropicMessages(t *testing.T) {
	adapter := &ClaudeAdapter{}
	
	tests := []struct {
		name                string
		messages            []Message
		expectedMessages    []AnthropicMessage
		expectedSystemMsg   string
	}{
		{
			name: "Simple user message",
			messages: []Message{
				{Role: "user", Content: "Hello"},
			},
			expectedMessages: []AnthropicMessage{
				{Role: "user", Content: "Hello"},
			},
			expectedSystemMsg: "",
		},
		{
			name: "System and user messages",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant"},
				{Role: "user", Content: "Hello"},
			},
			expectedMessages: []AnthropicMessage{
				{Role: "user", Content: "Hello"},
			},
			expectedSystemMsg: "You are a helpful assistant",
		},
		{
			name: "User and assistant conversation",
			messages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			expectedMessages: []AnthropicMessage{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
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
			expectedMessages: []AnthropicMessage{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi!"},
			},
			expectedSystemMsg: "You are helpful",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages, systemMsg := adapter.ConvertToAnthropicMessages(tt.messages)
			
			if !reflect.DeepEqual(messages, tt.expectedMessages) {
				t.Errorf("ConvertToAnthropicMessages() messages = %v, want %v", messages, tt.expectedMessages)
			}
			
			if systemMsg != tt.expectedSystemMsg {
				t.Errorf("ConvertToAnthropicMessages() systemMsg = %q, want %q", systemMsg, tt.expectedSystemMsg)
			}
		})
	}
}

func TestClaudeAdapter_BuildAnthropicPayload(t *testing.T) {
	adapter := &ClaudeAdapter{}
	
	tests := []struct {
		name          string
		messages      []AnthropicMessage
		systemMessage string
		expectedKeys  []string
	}{
		{
			name: "Basic payload",
			messages: []AnthropicMessage{
				{Role: "user", Content: "Hello"},
			},
			systemMessage: "",
			expectedKeys:  []string{"anthropic_version", "max_tokens", "messages"},
		},
		{
			name: "Payload with system message",
			messages: []AnthropicMessage{
				{Role: "user", Content: "Hello"},
			},
			systemMessage: "You are helpful",
			expectedKeys:  []string{"anthropic_version", "max_tokens", "messages", "system"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload := adapter.BuildAnthropicPayload(tt.messages, tt.systemMessage)
			
			// Check that all expected keys are present
			for _, key := range tt.expectedKeys {
				if _, exists := payload[key]; !exists {
					t.Errorf("BuildAnthropicPayload() missing key %q", key)
				}
			}
			
			// Check specific values
			if payload["anthropic_version"] != "bedrock-2023-05-31" {
				t.Errorf("BuildAnthropicPayload() anthropic_version = %v, want %q", payload["anthropic_version"], "bedrock-2023-05-31")
			}
			
			if payload["max_tokens"] != 1000 {
				t.Errorf("BuildAnthropicPayload() max_tokens = %v, want %d", payload["max_tokens"], 1000)
			}
			
			if tt.systemMessage != "" && payload["system"] != tt.systemMessage {
				t.Errorf("BuildAnthropicPayload() system = %v, want %q", payload["system"], tt.systemMessage)
			}
		})
	}
}

func TestClaudeAdapter_ConvertAnthropicToOpenAI(t *testing.T) {
	adapter := &ClaudeAdapter{}
	
	tests := []struct {
		name           string
		anthropicResp  map[string]interface{}
		model          string
		expectedContent string
	}{
		{
			name: "Valid Anthropic response",
			anthropicResp: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"text": "Hello, how can I help you?",
					},
				},
			},
			model:          "anthropic.claude-3-sonnet-20240229-v1:0",
			expectedContent: "Hello, how can I help you?",
		},
		{
			name: "Empty content response",
			anthropicResp: map[string]interface{}{
				"content": []interface{}{},
			},
			model:          "claude-3-haiku",
			expectedContent: "",
		},
		{
			name:           "Malformed response",
			anthropicResp:  map[string]interface{}{},
			model:          "claude-3-opus",
			expectedContent: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ConvertAnthropicToOpenAI(tt.anthropicResp, tt.model)
			
			// Check basic structure
			if result["object"] != "chat.completion" {
				t.Errorf("ConvertAnthropicToOpenAI() object = %v, want %q", result["object"], "chat.completion")
			}
			
			if result["model"] != tt.model {
				t.Errorf("ConvertAnthropicToOpenAI() model = %v, want %q", result["model"], tt.model)
			}
			
			// Check message content
			choices, ok := result["choices"].([]map[string]interface{})
			if !ok || len(choices) == 0 {
				t.Fatal("ConvertAnthropicToOpenAI() choices not found or empty")
			}
			
			message, ok := choices[0]["message"].(map[string]string)
			if !ok {
				t.Fatal("ConvertAnthropicToOpenAI() message not found")
			}
			
			if message["content"] != tt.expectedContent {
				t.Errorf("ConvertAnthropicToOpenAI() content = %q, want %q", message["content"], tt.expectedContent)
			}
			
			if message["role"] != "assistant" {
				t.Errorf("ConvertAnthropicToOpenAI() role = %q, want %q", message["role"], "assistant")
			}
		})
	}
}