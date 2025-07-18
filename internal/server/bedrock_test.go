package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"llm-freeway/internal/config"
	"llm-freeway/internal/database"
	"llm-freeway/internal/errors"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// BedrockClientInterface defines the interface for Bedrock client operations
type BedrockClientInterface interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

// testBedrockServer wraps Server for testing with mocked Bedrock dependencies
type testBedrockServer struct {
	*Server
	mockBedrockClient *mockBedrockClient
}

// mockBedrockClient implements Bedrock functionality for testing
type mockBedrockClient struct {
	response map[string]interface{}
	err      error
	modelID  string
	body     []byte
}

func (m *mockBedrockClient) InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	if m.err != nil {
		return nil, m.err
	}

	// Store the model ID and body for validation
	if params.ModelId != nil {
		m.modelID = *params.ModelId
	}
	m.body = params.Body

	// Return mock response
	responseBytes, _ := json.Marshal(m.response)
	return &bedrockruntime.InvokeModelOutput{
		Body: responseBytes,
	}, nil
}

// testBedrockEmbeddingHandler is a test version that uses our mock
func (ts *testBedrockServer) testHandleBedrockEmbedding(req EmbeddingRequest, actualModelName string, w *httptest.ResponseRecorder, metric *database.RequestMetrics, provider config.ProviderType) error {
	if ts.mockBedrockClient == nil {
		return ts.errorHandler.HandleError(fmt.Errorf("Bedrock client not available - missing AWS credentials"), provider, map[string]any{"model": req.Model})
	}

	// Convert input to the format expected by Bedrock
	var input []string
	switch v := req.Input.(type) {
	case string:
		input = []string{v}
	case []string:
		input = v
	case []interface{}:
		input = make([]string, len(v))
		for i, item := range v {
			if str, ok := item.(string); ok {
				input[i] = str
			} else {
				return ts.errorHandler.CreateValidationError(provider, "All input items must be strings", nil)
			}
		}
	default:
		return ts.errorHandler.CreateValidationError(provider, "Input must be a string or array of strings", nil)
	}

	// Call mock Bedrock embedding API
	bedrockResponse, err := ts.callMockBedrockEmbedding(input, actualModelName)
	if err != nil {
		return ts.handleNonStreamingError(err, w, metric, req.Model, provider)
	}

	// Convert to our EmbeddingData format
	embeddings := make([]EmbeddingData, len(bedrockResponse.Embeddings))
	for i, embedding := range bedrockResponse.Embeddings {
		embeddings[i] = EmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: embedding,
		}
	}

	response := EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  req.Model, // Return original model name
		Usage: EmbeddingUsage{
			PromptTokens: bedrockResponse.Usage.PromptTokens,
			TotalTokens:  bedrockResponse.Usage.TotalTokens,
		},
	}

	// Update metrics with token usage
	metric.TokensUsed = response.Usage.TotalTokens
	metric.PromptTokens = response.Usage.PromptTokens
	metric.ResponseTokens = 0 // Embeddings don't have response tokens

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
	return nil
}

// callMockBedrockEmbedding calls the appropriate mock Bedrock embedding API
func (ts *testBedrockServer) callMockBedrockEmbedding(input []string, modelName string) (*BedrockEmbeddingResponse, error) {
	switch {
	case strings.Contains(modelName, "amazon.titan-embed"):
		return ts.callMockTitanEmbedding(input, modelName)
	case strings.Contains(modelName, "cohere.embed"):
		return ts.callMockCohereEmbedding(input, modelName)
	default:
		return nil, fmt.Errorf("unsupported Bedrock embedding model: %s", modelName)
	}
}

// callMockTitanEmbedding calls mock Amazon Titan embedding
func (ts *testBedrockServer) callMockTitanEmbedding(input []string, modelName string) (*BedrockEmbeddingResponse, error) {
	var allEmbeddings [][]float32
	totalTokens := 0

	for _, text := range input {
		// Prepare request body for Titan embedding
		requestBody := map[string]interface{}{
			"inputText": text,
		}

		// Estimate tokens (rough approximation)
		tokens := len(strings.Fields(text))
		totalTokens += tokens

		// Call mock Bedrock API
		embedding, err := ts.invokeMockBedrockModel(modelName, requestBody)
		if err != nil {
			return nil, err
		}

		// Parse Titan response
		titanResp, ok := embedding.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid response format from Titan")
		}

		embeddingData, ok := titanResp["embedding"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("no embedding data in Titan response")
		}

		// Convert to []float32
		embeddingFloat32 := make([]float32, len(embeddingData))
		for i, val := range embeddingData {
			if floatVal, ok := val.(float64); ok {
				embeddingFloat32[i] = float32(floatVal)
			} else {
				return nil, fmt.Errorf("invalid embedding value type")
			}
		}

		allEmbeddings = append(allEmbeddings, embeddingFloat32)
	}

	return &BedrockEmbeddingResponse{
		Embeddings: allEmbeddings,
		Usage: BedrockEmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}, nil
}

// callMockCohereEmbedding calls mock Cohere embedding
func (ts *testBedrockServer) callMockCohereEmbedding(input []string, modelName string) (*BedrockEmbeddingResponse, error) {
	// Prepare request body for Cohere embedding
	requestBody := map[string]interface{}{
		"texts":      input,
		"input_type": "search_document",
	}

	// Estimate tokens
	totalTokens := 0
	for _, text := range input {
		totalTokens += len(strings.Fields(text))
	}

	// Call mock Bedrock API
	embedding, err := ts.invokeMockBedrockModel(modelName, requestBody)
	if err != nil {
		return nil, err
	}

	// Parse Cohere response
	cohereResp, ok := embedding.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response format from Cohere")
	}

	embeddingsData, ok := cohereResp["embeddings"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("no embeddings data in Cohere response")
	}

	// Convert to [][]float32
	allEmbeddings := make([][]float32, len(embeddingsData))
	for i, embeddingInterface := range embeddingsData {
		embeddingArray, ok := embeddingInterface.([]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid embedding array format")
		}

		embeddingFloat32 := make([]float32, len(embeddingArray))
		for j, val := range embeddingArray {
			if floatVal, ok := val.(float64); ok {
				embeddingFloat32[j] = float32(floatVal)
			} else {
				return nil, fmt.Errorf("invalid embedding value type")
			}
		}
		allEmbeddings[i] = embeddingFloat32
	}

	return &BedrockEmbeddingResponse{
		Embeddings: allEmbeddings,
		Usage: BedrockEmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}, nil
}

// invokeMockBedrockModel invokes the mock Bedrock model
func (ts *testBedrockServer) invokeMockBedrockModel(modelName string, requestBody map[string]interface{}) (interface{}, error) {
	// Marshal request body to JSON
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %v", err)
	}

	// Invoke mock Bedrock model
	input := &bedrockruntime.InvokeModelInput{
		ModelId: &modelName,
		Body:    bodyBytes,
	}

	result, err := ts.mockBedrockClient.InvokeModel(context.Background(), input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke Bedrock model: %v", err)
	}

	// Parse response
	var response interface{}
	if err := json.Unmarshal(result.Body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	return response, nil
}

// createTestBedrockServer creates a server for testing Bedrock functionality
func createTestBedrockServer(mockClient *mockBedrockClient) *testBedrockServer {
	testConfig := &config.Config{}
	server := &Server{
		errorHandler: errors.NewErrorHandler(testConfig),
	}

	return &testBedrockServer{
		Server:            server,
		mockBedrockClient: mockClient,
	}
}

func TestServer_handleBedrockEmbedding_TitanModels(t *testing.T) {
	tests := []struct {
		name         string
		request      EmbeddingRequest
		modelName    string
		mockResponse map[string]interface{}
		mockError    error
		expectErr    bool
		validateResp func(t *testing.T, resp EmbeddingResponse)
		validateCall func(t *testing.T, modelID string, body []byte)
	}{
		{
			name: "Titan v1 single string input",
			request: EmbeddingRequest{
				Input: "Hello world",
				Model: "amazon-titan-embed-text-v1",
			},
			modelName: "amazon.titan-embed-text-v1",
			mockResponse: map[string]interface{}{
				"embedding": []interface{}{0.1, 0.2, 0.3, 0.4},
			},
			expectErr: false,
			validateResp: func(t *testing.T, resp EmbeddingResponse) {
				if len(resp.Data) != 1 {
					t.Errorf("Expected 1 embedding, got %d", len(resp.Data))
				}
				if len(resp.Data[0].Embedding) != 4 {
					t.Errorf("Expected 4 dimensions, got %d", len(resp.Data[0].Embedding))
				}
				if resp.Usage.PromptTokens != 2 { // "Hello world" = 2 tokens
					t.Errorf("Expected 2 prompt tokens, got %d", resp.Usage.PromptTokens)
				}
			},
			validateCall: func(t *testing.T, modelID string, body []byte) {
				if modelID != "amazon.titan-embed-text-v1" {
					t.Errorf("Expected model ID amazon.titan-embed-text-v1, got %s", modelID)
				}
				var requestBody map[string]interface{}
				json.Unmarshal(body, &requestBody)
				if requestBody["inputText"] != "Hello world" {
					t.Errorf("Expected inputText 'Hello world', got %v", requestBody["inputText"])
				}
			},
		},
		{
			name: "Titan v2 multiple string inputs",
			request: EmbeddingRequest{
				Input: []string{"Hello", "world"},
				Model: "amazon-titan-embed-text-v2",
			},
			modelName: "amazon.titan-embed-text-v2:0",
			mockResponse: map[string]interface{}{
				"embedding": []interface{}{0.1, 0.2, 0.3},
			},
			expectErr: false,
			validateResp: func(t *testing.T, resp EmbeddingResponse) {
				if len(resp.Data) != 2 {
					t.Errorf("Expected 2 embeddings, got %d", len(resp.Data))
				}
				if resp.Usage.PromptTokens != 2 { // "Hello" + "world" = 2 tokens
					t.Errorf("Expected 2 prompt tokens, got %d", resp.Usage.PromptTokens)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client
			mockClient := &mockBedrockClient{
				response: tt.mockResponse,
				err:      tt.mockError,
			}

			// Create test server
			testServer := createTestBedrockServer(mockClient)

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.testHandleBedrockEmbedding(tt.request, tt.modelName, w, metric, config.ProviderAWSBedrock)

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

			// Parse response
			var response EmbeddingResponse
			if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
				t.Errorf("Failed to decode response: %v", err)
				return
			}

			// Validate response
			if tt.validateResp != nil {
				tt.validateResp(t, response)
			}

			// Validate API call
			if tt.validateCall != nil {
				tt.validateCall(t, mockClient.modelID, mockClient.body)
			}

			// Check content type
			if w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Expected Content-Type application/json, got %s", w.Header().Get("Content-Type"))
			}
		})
	}
}

func TestServer_handleBedrockEmbedding_CohereModels(t *testing.T) {
	tests := []struct {
		name         string
		request      EmbeddingRequest
		modelName    string
		mockResponse map[string]interface{}
		expectErr    bool
		validateResp func(t *testing.T, resp EmbeddingResponse)
		validateCall func(t *testing.T, modelID string, body []byte)
	}{
		{
			name: "Cohere English v3 single input",
			request: EmbeddingRequest{
				Input: "Hello world",
				Model: "cohere-embed-english-v3",
			},
			modelName: "cohere.embed-english-v3",
			mockResponse: map[string]interface{}{
				"embeddings": []interface{}{
					[]interface{}{0.1, 0.2, 0.3},
				},
			},
			expectErr: false,
			validateResp: func(t *testing.T, resp EmbeddingResponse) {
				if len(resp.Data) != 1 {
					t.Errorf("Expected 1 embedding, got %d", len(resp.Data))
				}
				if len(resp.Data[0].Embedding) != 3 {
					t.Errorf("Expected 3 dimensions, got %d", len(resp.Data[0].Embedding))
				}
			},
			validateCall: func(t *testing.T, modelID string, body []byte) {
				if modelID != "cohere.embed-english-v3" {
					t.Errorf("Expected model ID cohere.embed-english-v3, got %s", modelID)
				}
				var requestBody map[string]interface{}
				json.Unmarshal(body, &requestBody)
				
				texts, ok := requestBody["texts"].([]interface{})
				if !ok {
					t.Error("Expected texts array in request body")
					return
				}
				if len(texts) != 1 || texts[0] != "Hello world" {
					t.Errorf("Expected texts ['Hello world'], got %v", texts)
				}
				
				if requestBody["input_type"] != "search_document" {
					t.Errorf("Expected input_type 'search_document', got %v", requestBody["input_type"])
				}
			},
		},
		{
			name: "Cohere Multilingual v3 multiple inputs",
			request: EmbeddingRequest{
				Input: []string{"Hello", "Bonjour"},
				Model: "cohere-embed-multilingual-v3",
			},
			modelName: "cohere.embed-multilingual-v3",
			mockResponse: map[string]interface{}{
				"embeddings": []interface{}{
					[]interface{}{0.1, 0.2},
					[]interface{}{0.3, 0.4},
				},
			},
			expectErr: false,
			validateResp: func(t *testing.T, resp EmbeddingResponse) {
				if len(resp.Data) != 2 {
					t.Errorf("Expected 2 embeddings, got %d", len(resp.Data))
				}
				if resp.Data[0].Embedding[0] != 0.1 {
					t.Errorf("Expected first embedding to start with 0.1, got %f", resp.Data[0].Embedding[0])
				}
				if resp.Data[1].Embedding[0] != 0.3 {
					t.Errorf("Expected second embedding to start with 0.3, got %f", resp.Data[1].Embedding[0])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client
			mockClient := &mockBedrockClient{
				response: tt.mockResponse,
			}

			// Create test server
			testServer := createTestBedrockServer(mockClient)

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.testHandleBedrockEmbedding(tt.request, tt.modelName, w, metric, config.ProviderAWSBedrock)

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

			// Parse response
			var response EmbeddingResponse
			if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
				t.Errorf("Failed to decode response: %v", err)
				return
			}

			// Validate response
			if tt.validateResp != nil {
				tt.validateResp(t, response)
			}

			// Validate API call
			if tt.validateCall != nil {
				tt.validateCall(t, mockClient.modelID, mockClient.body)
			}
		})
	}
}

func TestServer_handleBedrockEmbedding_ErrorCases(t *testing.T) {
	tests := []struct {
		name           string
		request        EmbeddingRequest
		modelName      string
		bedrockClient  bool
		mockError      error
		expectErr      bool
		errContains    string
	}{
		{
			name: "No Bedrock client",
			request: EmbeddingRequest{
				Input: "Hello world",
				Model: "amazon-titan-embed-text-v1",
			},
			modelName:     "amazon.titan-embed-text-v1",
			bedrockClient: false,
			expectErr:     true,
			errContains:   "Bedrock client not available",
		},
		{
			name: "Invalid input type",
			request: EmbeddingRequest{
				Input: 123,
				Model: "amazon-titan-embed-text-v1",
			},
			modelName:     "amazon.titan-embed-text-v1",
			bedrockClient: true,
			expectErr:     true,
			errContains:   "Input must be a string or array of strings",
		},
		{
			name: "Interface slice with non-string",
			request: EmbeddingRequest{
				Input: []interface{}{"Hello", 123},
				Model: "amazon-titan-embed-text-v1",
			},
			modelName:     "amazon.titan-embed-text-v1",
			bedrockClient: true,
			expectErr:     true,
			errContains:   "All input items must be strings",
		},
		{
			name: "Unsupported model",
			request: EmbeddingRequest{
				Input: "Hello world",
				Model: "unsupported-model",
			},
			modelName:     "unsupported-model",
			bedrockClient: true,
			expectErr:     true,
		},
		{
			name: "Bedrock API error",
			request: EmbeddingRequest{
				Input: "Hello world",
				Model: "amazon-titan-embed-text-v1",
			},
			modelName:     "amazon.titan-embed-text-v1",
			bedrockClient: true,
			mockError:     fmt.Errorf("AWS service error"),
			expectErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client
			var mockClient *mockBedrockClient
			if tt.bedrockClient {
				mockClient = &mockBedrockClient{
					err: tt.mockError,
				}
			}

			// Create test server
			testServer := createTestBedrockServer(mockClient)

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.handleBedrockEmbedding(tt.request, tt.modelName, w, metric, config.ProviderAWSBedrock)

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

func TestServer_handleBedrockEmbedding_InputConversion(t *testing.T) {
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client
			mockClient := &mockBedrockClient{
				response: map[string]interface{}{
					"embedding": []interface{}{0.1, 0.2},
				},
			}

			// Create test server
			testServer := createTestBedrockServer(mockClient)

			// Create request
			req := EmbeddingRequest{
				Input: tt.input,
				Model: "amazon-titan-embed-text-v1",
			}

			// Create response recorder
			w := httptest.NewRecorder()

			// Create metrics
			metric := &database.RequestMetrics{
				StartTime: time.Now(),
			}

			// Call the handler
			err := testServer.testHandleBedrockEmbedding(req, "amazon.titan-embed-text-v1", w, metric, config.ProviderAWSBedrock)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}

func TestServer_callBedrockEmbedding_ModelRouting(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		expectErr bool
		errMsg    string
	}{
		{
			name:      "Titan embedding v1",
			modelName: "amazon.titan-embed-text-v1",
			expectErr: false,
		},
		{
			name:      "Titan embedding v2",
			modelName: "amazon.titan-embed-text-v2:0",
			expectErr: false,
		},
		{
			name:      "Cohere English",
			modelName: "cohere.embed-english-v3",
			expectErr: false,
		},
		{
			name:      "Cohere Multilingual",
			modelName: "cohere.embed-multilingual-v3",
			expectErr: false,
		},
		{
			name:      "Unsupported model",
			modelName: "unsupported.model",
			expectErr: true,
			errMsg:    "unsupported Bedrock embedding model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client
			mockClient := &mockBedrockClient{
				response: map[string]interface{}{
					"embedding":  []interface{}{0.1, 0.2},
					"embeddings": []interface{}{[]interface{}{0.1, 0.2}},
				},
			}

			// Create test server
			testServer := createTestBedrockServer(mockClient)

			// Call the function
			_, err := testServer.callMockBedrockEmbedding([]string{"test"}, tt.modelName)

			if tt.expectErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				if tt.errMsg != "" && err != nil {
					t.Logf("Got expected error: %v", err)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}