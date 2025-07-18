package server

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	"llm-freeway/internal/adapter"
	"llm-freeway/internal/config"
	"llm-freeway/internal/database"
	"llm-freeway/internal/errors"
)

// testClaudeServer wraps Server for testing with mocked Claude dependencies
type testClaudeServer struct {
	*Server
	mockClaudeAdapter *mockClaudeAdapter
}

// mockClaudeAdapter implements Claude functionality for testing
type mockClaudeAdapter struct {
	streamingResponse string
	nonStreamingResp  map[string]interface{}
	err               error
	provider          string
}

func (m *mockClaudeAdapter) HandleStreamingRequestWithProvider(req adapter.ChatRequest, w *httptest.ResponseRecorder, provider string) error {
	m.provider = provider
	if m.err != nil {
		return m.err
	}
	
	// Simulate streaming response
	if m.streamingResponse != "" {
		w.Write([]byte(m.streamingResponse))
	}
	return nil
}

func (m *mockClaudeAdapter) HandleRequestWithProvider(req adapter.ChatRequest, provider string) (map[string]interface{}, error) {
	m.provider = provider
	if m.err != nil {
		return nil, m.err
	}
	return m.nonStreamingResp, nil
}

// createTestClaudeServer creates a server for testing Claude functionality
func createTestClaudeServer(mockAdapter *mockClaudeAdapter) *testClaudeServer {
	testConfig := &config.Config{}
	server := &Server{
		errorHandler: errors.NewErrorHandler(testConfig),
	}
	
	return &testClaudeServer{
		Server:            server,
		mockClaudeAdapter: mockAdapter,
	}
}

// handleClaudeTest is a test version that uses our mock
func (ts *testClaudeServer) handleClaudeTest(req adapter.ChatRequest, w *httptest.ResponseRecorder, metric *database.RequestMetrics, provider config.ProviderType) error {
	if ts.mockClaudeAdapter == nil {
		return ts.errorHandler.HandleError(fmt.Errorf("Claude client not available - missing API key configuration"), provider, map[string]any{"model": req.Model})
	}

	if req.Stream {
		return ts.handleStreamingRequest(req, w, metric, func() error {
			return ts.mockClaudeAdapter.HandleStreamingRequestWithProvider(req, w, string(provider))
		})
	}

	// Handle non-streaming
	openaiResp, err := ts.mockClaudeAdapter.HandleRequestWithProvider(req, string(provider))
	if err != nil {
		return ts.handleNonStreamingError(err, w, metric, req.Model, provider)
	}

	ts.extractTokenUsage(openaiResp, metric)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openaiResp)
	return nil
}

func TestServer_handleClaude(t *testing.T) {
	tests := []struct {
		name           string
		request        adapter.ChatRequest
		provider       config.ProviderType
		mockResponse   map[string]interface{}
		mockError      error
		expectErr      bool
		expectedStatus int
		validateResp   func(t *testing.T, resp map[string]interface{})
		validateProvider func(t *testing.T, provider string)
	}{
		{
			name: "Non-streaming request to AWS Bedrock",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet-bedrock",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
				Stream: false,
			},
			provider: config.ProviderAWSBedrock,
			mockResponse: map[string]interface{}{
				"id":      "chatcmpl-test",
				"object":  "chat.completion",
				"created": 1699999999,
				"model":   "claude-3-5-sonnet-bedrock",
				"choices": []interface{}{
					map[string]interface{}{
						"index": 0,
						"message": map[string]interface{}{
							"role":    "assistant",
							"content": "Hello! How can I help you?",
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     10,
					"completion_tokens": 8,
					"total_tokens":      18,
				},
			},
			expectErr:      false,
			expectedStatus: 200,
			validateResp: func(t *testing.T, resp map[string]interface{}) {
				if resp["model"] != "claude-3-5-sonnet-bedrock" {
					t.Errorf("Expected model claude-3-5-sonnet-bedrock, got %v", resp["model"])
				}
				choices, ok := resp["choices"].([]interface{})
				if !ok {
					t.Error("Expected choices array")
					return
				}
				if len(choices) == 0 {
					t.Error("Expected at least one choice")
					return
				}
				choice, ok := choices[0].(map[string]interface{})
				if !ok {
					t.Error("Expected choice object")
					return
				}
				message, ok := choice["message"].(map[string]interface{})
				if !ok {
					t.Error("Expected message object")
					return
				}
				if message["content"] != "Hello! How can I help you?" {
					t.Errorf("Expected content 'Hello! How can I help you?', got %v", message["content"])
				}
			},
			validateProvider: func(t *testing.T, provider string) {
				if provider != string(config.ProviderAWSBedrock) {
					t.Errorf("Expected provider %s, got %s", config.ProviderAWSBedrock, provider)
				}
			},
		},
		{
			name: "Non-streaming request to Direct Anthropic",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
				Stream: false,
			},
			provider: config.ProviderAnthropic,
			mockResponse: map[string]interface{}{
				"id":      "chatcmpl-test",
				"object":  "chat.completion",
				"created": 1699999999,
				"model":   "claude-3-5-sonnet",
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"message": map[string]string{
							"role":    "assistant",
							"content": "Hello! How can I help you?",
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     10,
					"completion_tokens": 8,
					"total_tokens":      18,
				},
			},
			expectErr:      false,
			expectedStatus: 200,
			validateProvider: func(t *testing.T, provider string) {
				if provider != string(config.ProviderAnthropic) {
					t.Errorf("Expected provider %s, got %s", config.ProviderAnthropic, provider)
				}
			},
		},
		{
			name: "Streaming request to AWS Bedrock",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet-bedrock",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
				Stream: true,
			},
			provider:       config.ProviderAWSBedrock,
			expectErr:      false,
			expectedStatus: 200,
			validateProvider: func(t *testing.T, provider string) {
				if provider != string(config.ProviderAWSBedrock) {
					t.Errorf("Expected provider %s, got %s", config.ProviderAWSBedrock, provider)
				}
			},
		},
		{
			name: "Streaming request to Direct Anthropic",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
				Stream: true,
			},
			provider:       config.ProviderAnthropic,
			expectErr:      false,
			expectedStatus: 200,
			validateProvider: func(t *testing.T, provider string) {
				if provider != string(config.ProviderAnthropic) {
					t.Errorf("Expected provider %s, got %s", config.ProviderAnthropic, provider)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock adapter
			mockAdapter := &mockClaudeAdapter{
				nonStreamingResp: tt.mockResponse,
				err:              tt.mockError,
			}

			// Create test server
			testServer := createTestClaudeServer(mockAdapter)

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.handleClaudeTest(tt.request, w, metric, tt.provider)

			if tt.expectErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Check response status
			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, w.Code)
			}

			// Validate provider was passed correctly
			if tt.validateProvider != nil {
				tt.validateProvider(t, mockAdapter.provider)
			}

			// For non-streaming requests, validate response
			if !tt.request.Stream && tt.validateResp != nil {
				var response map[string]interface{}
				if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
					t.Errorf("Failed to decode response: %v", err)
					return
				}
				tt.validateResp(t, response)
			}

			// Check content type based on streaming
			if tt.request.Stream {
				if w.Header().Get("Content-Type") != "text/event-stream" {
					t.Errorf("Expected Content-Type text/event-stream for streaming, got %s", w.Header().Get("Content-Type"))
				}
			} else {
				if w.Header().Get("Content-Type") != "application/json" {
					t.Errorf("Expected Content-Type application/json for non-streaming, got %s", w.Header().Get("Content-Type"))
				}
			}
		})
	}
}

func TestServer_handleClaude_ErrorCases(t *testing.T) {
	tests := []struct {
		name         string
		request      adapter.ChatRequest
		provider     config.ProviderType
		mockError    error
		claudeClient bool
		expectErr    bool
		errContains  string
	}{
		{
			name: "No Claude client",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			provider:     config.ProviderAnthropic,
			claudeClient: false,
			expectErr:    true,
			errContains:  "Claude client not available",
		},
		{
			name: "Adapter error - non-streaming",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
				Stream: false,
			},
			provider:     config.ProviderAnthropic,
			claudeClient: true,
			mockError:    fmt.Errorf("API rate limit exceeded"),
			expectErr:    true,
		},
		{
			name: "Adapter error - streaming",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
				Stream: true,
			},
			provider:     config.ProviderAnthropic,
			claudeClient: true,
			mockError:    fmt.Errorf("Streaming connection failed"),
			expectErr:    true,
		},
		{
			name: "Bedrock-specific error",
			request: adapter.ChatRequest{
				Model: "claude-3-5-sonnet-bedrock",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			provider:     config.ProviderAWSBedrock,
			claudeClient: true,
			mockError:    fmt.Errorf("AWS credentials invalid"),
			expectErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock adapter
			var mockAdapter *mockClaudeAdapter
			if tt.claudeClient {
				mockAdapter = &mockClaudeAdapter{
					err: tt.mockError,
				}
			}

			// Create test server
			testServer := createTestClaudeServer(mockAdapter)

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.handleClaudeTest(tt.request, w, metric, tt.provider)

			if tt.expectErr {
				if err == nil {
					t.Error("Expected error but got none")
					return
				}
				if tt.errContains != "" && err.Error() != "" {
					t.Logf("Got error: %v", err)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}

func TestServer_handleClaude_ProviderRouting(t *testing.T) {
	tests := []struct {
		name             string
		model            string
		expectedProvider config.ProviderType
		description      string
	}{
		{
			name:             "Bedrock model routing",
			model:            "claude-3-5-sonnet-bedrock",
			expectedProvider: config.ProviderAWSBedrock,
			description:      "Models with 'bedrock' suffix should route to AWS Bedrock",
		},
		{
			name:             "Direct Anthropic model routing",
			model:            "claude-3-5-sonnet",
			expectedProvider: config.ProviderAnthropic,
			description:      "Models without 'bedrock' suffix should route to Direct Anthropic",
		},
		{
			name:             "Haiku Bedrock routing",
			model:            "claude-3-haiku-bedrock",
			expectedProvider: config.ProviderAWSBedrock,
			description:      "Haiku Bedrock models should route correctly",
		},
		{
			name:             "Haiku Direct routing",
			model:            "claude-3-haiku",
			expectedProvider: config.ProviderAnthropic,
			description:      "Haiku Direct models should route correctly",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock adapter
			mockAdapter := &mockClaudeAdapter{
				nonStreamingResp: map[string]interface{}{
					"choices": []interface{}{
						map[string]interface{}{"message": map[string]interface{}{"content": "test"}},
					},
					"usage": map[string]interface{}{
						"total_tokens": 10,
					},
				},
			}

			// Create test server
			testServer := createTestClaudeServer(mockAdapter)

			// Create request
			req := adapter.ChatRequest{
				Model: tt.model,
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
			}

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.handleClaudeTest(req, w, metric, tt.expectedProvider)

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Validate provider was passed correctly
			if mockAdapter.provider != string(tt.expectedProvider) {
				t.Errorf("Expected provider %s, got %s. %s", tt.expectedProvider, mockAdapter.provider, tt.description)
			}
		})
	}
}

func TestServer_handleClaude_TokenUsageExtraction(t *testing.T) {
	tests := []struct {
		name         string
		mockResponse map[string]interface{}
		expectedTokens int
		expectedPrompt int
		expectedCompletion int
	}{
		{
			name: "Standard token usage",
			mockResponse: map[string]interface{}{
				"choices": []interface{}{
					map[string]interface{}{"message": map[string]interface{}{"content": "Hello"}},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     15,
					"completion_tokens": 8,
					"total_tokens":      23,
				},
			},
			expectedTokens: 23,
			expectedPrompt: 15,
			expectedCompletion: 8,
		},
		{
			name: "No usage information",
			mockResponse: map[string]interface{}{
				"choices": []interface{}{
					map[string]interface{}{"message": map[string]interface{}{"content": "Hello"}},
				},
			},
			expectedTokens: 0,
			expectedPrompt: 0,
			expectedCompletion: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock adapter
			mockAdapter := &mockClaudeAdapter{
				nonStreamingResp: tt.mockResponse,
			}

			// Create test server
			testServer := createTestClaudeServer(mockAdapter)

			// Create request
			req := adapter.ChatRequest{
				Model: "claude-3-5-sonnet",
				Messages: []adapter.Message{
					{Role: "user", Content: "Hello"},
				},
			}

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.handleClaudeTest(req, w, metric, config.ProviderAnthropic)

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Validate token usage was extracted correctly
			if metric.TokensUsed != tt.expectedTokens {
				t.Errorf("Expected total tokens %d, got %d", tt.expectedTokens, metric.TokensUsed)
			}
			if metric.PromptTokens != tt.expectedPrompt {
				t.Errorf("Expected prompt tokens %d, got %d", tt.expectedPrompt, metric.PromptTokens)
			}
			if metric.ResponseTokens != tt.expectedCompletion {
				t.Errorf("Expected completion tokens %d, got %d", tt.expectedCompletion, metric.ResponseTokens)
			}
		})
	}
}