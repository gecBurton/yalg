package errors

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"llm-freeway/internal/config"
)

// ErrorType represents different categories of errors
type ErrorType string

const (
	ErrorTypeAuthentication  ErrorType = "authentication_error"
	ErrorTypeAuthorization   ErrorType = "authorization_error"
	ErrorTypeRateLimit       ErrorType = "rate_limit_error"
	ErrorTypeInvalidRequest  ErrorType = "invalid_request_error"
	ErrorTypeModelError      ErrorType = "model_error"
	ErrorTypeProviderError   ErrorType = "provider_error"
	ErrorTypeTimeout         ErrorType = "timeout_error"
	ErrorTypeNetworkError    ErrorType = "network_error"
	ErrorTypeInternalError   ErrorType = "internal_server_error"
	ErrorTypeQuotaExceeded   ErrorType = "quota_exceeded_error"
	ErrorTypeSafetyError     ErrorType = "safety_error"
	ErrorTypeValidationError ErrorType = "validation_error"
)

// ErrorCode represents specific error codes
type ErrorCode string

const (
	CodeInvalidAPIKey       ErrorCode = "invalid_api_key"
	CodeInsufficientQuota   ErrorCode = "insufficient_quota"
	CodeModelNotFound       ErrorCode = "model_not_found"
	CodeTokenLimitExceeded  ErrorCode = "token_limit_exceeded"
	CodeContentFiltered     ErrorCode = "content_filtered"
	CodeProviderUnavailable ErrorCode = "provider_unavailable"
	CodeRequestTimeout      ErrorCode = "request_timeout"
	CodeInvalidModel        ErrorCode = "invalid_model"
	CodeMissingParameter    ErrorCode = "missing_parameter"
	CodeInvalidParameter    ErrorCode = "invalid_parameter"
	CodeConnectionFailed    ErrorCode = "connection_failed"
	CodeServerOverloaded    ErrorCode = "server_overloaded"
)

// ProviderError represents a provider-specific error
type ProviderError struct {
	Provider      config.ProviderType    `json:"provider"`
	Type          ErrorType              `json:"type"`
	Code          ErrorCode              `json:"code"`
	Message       string                 `json:"message"`
	Details       map[string]interface{} `json:"details,omitempty"`
	Retryable     bool                   `json:"retryable"`
	RetryAfter    int                    `json:"retry_after,omitempty"` // seconds
	HTTPStatus    int                    `json:"http_status"`
	OriginalError string                 `json:"original_error,omitempty"`
}

// Error implements the error interface
func (e *ProviderError) Error() string {
	return fmt.Sprintf("[%s] %s: %s", e.Provider, e.Type, e.Message)
}

// ToOpenAIError converts the provider error to OpenAI-compatible format
func (e *ProviderError) ToOpenAIError(requestID string) map[string]interface{} {
	return map[string]interface{}{
		"error": map[string]interface{}{
			"message": e.Message,
			"type":    string(e.Type),
			"code":    string(e.Code),
			"param":   nil,
		},
		"id":      requestID,
		"object":  "error",
		"created": nil,
	}
}

// ToOpenAIStreamError converts the provider error to OpenAI streaming format
func (e *ProviderError) ToOpenAIStreamError(requestID, model string) map[string]interface{} {
	return map[string]interface{}{
		"id":      requestID,
		"object":  "chat.completion.chunk",
		"created": nil,
		"model":   model,
		"error": map[string]interface{}{
			"message": e.Message,
			"type":    string(e.Type),
			"code":    string(e.Code),
		},
	}
}

// ErrorHandler provides centralized error handling and conversion
type ErrorHandler struct {
	config *config.Config
}

// NewErrorHandler creates a new error handler
func NewErrorHandler(cfg *config.Config) *ErrorHandler {
	return &ErrorHandler{config: cfg}
}

// HandleError converts various error types to ProviderError
func (eh *ErrorHandler) HandleError(err error, provider config.ProviderType, context map[string]interface{}) *ProviderError {
	if err == nil {
		return nil
	}

	// If it's already a ProviderError, return it
	if providerErr, ok := err.(*ProviderError); ok {
		return providerErr
	}

	errorStr := err.Error()
	errorLower := strings.ToLower(errorStr)

	// Create base error
	providerErr := &ProviderError{
		Provider:      provider,
		OriginalError: errorStr,
		Details:       context,
		HTTPStatus:    http.StatusInternalServerError,
	}

	// Detect error type and code based on provider and error message
	switch provider {
	case config.ProviderAzureOpenAI:
		eh.handleAzureOpenAIError(providerErr, errorLower)
	case config.ProviderAWSBedrock:
		eh.handleAWSBedrockError(providerErr, errorLower)
	case config.ProviderGoogleAI:
		eh.handleGoogleAIError(providerErr, errorLower)
	default:
		eh.handleGenericError(providerErr, errorLower)
	}

	return providerErr
}

// CreateValidationError creates a validation error
func (eh *ErrorHandler) CreateValidationError(provider config.ProviderType, message string, details map[string]interface{}) *ProviderError {
	return &ProviderError{
		Provider:   provider,
		Type:       ErrorTypeValidationError,
		Code:       CodeInvalidParameter,
		Message:    message,
		Details:    details,
		Retryable:  false,
		HTTPStatus: http.StatusBadRequest,
	}
}

// CreateRateLimitError creates a rate limit error
func (eh *ErrorHandler) CreateRateLimitError(provider config.ProviderType, retryAfter int) *ProviderError {
	return &ProviderError{
		Provider:   provider,
		Type:       ErrorTypeRateLimit,
		Code:       CodeServerOverloaded,
		Message:    "Rate limit exceeded. Please try again later.",
		Retryable:  true,
		RetryAfter: retryAfter,
		HTTPStatus: http.StatusTooManyRequests,
	}
}

// CreateModelNotFoundError creates a model not found error
func (eh *ErrorHandler) CreateModelNotFoundError(provider config.ProviderType, model string) *ProviderError {
	return &ProviderError{
		Provider: provider,
		Type:     ErrorTypeModelError,
		Code:     CodeModelNotFound,
		Message:  fmt.Sprintf("Model '%s' not found or not available", model),
		Details: map[string]interface{}{
			"model": model,
		},
		Retryable:  false,
		HTTPStatus: http.StatusNotFound,
	}
}

// Private methods for provider-specific error handling

func (eh *ErrorHandler) handleAzureOpenAIError(err *ProviderError, errorLower string) {
	switch {
	case strings.Contains(errorLower, "unauthorized") || strings.Contains(errorLower, "invalid api key"):
		err.Type = ErrorTypeAuthentication
		err.Code = CodeInvalidAPIKey
		err.Message = "Invalid API key provided"
		err.HTTPStatus = http.StatusUnauthorized
		err.Retryable = false

	case strings.Contains(errorLower, "quota"):
		err.Type = ErrorTypeQuotaExceeded
		err.Code = CodeInsufficientQuota
		err.Message = "Quota exceeded"
		err.HTTPStatus = http.StatusPaymentRequired
		err.Retryable = false

	case strings.Contains(errorLower, "rate limit"):
		err.Type = ErrorTypeRateLimit
		err.Code = CodeServerOverloaded
		err.Message = "Rate limit exceeded"
		err.HTTPStatus = http.StatusTooManyRequests
		err.Retryable = true
		err.RetryAfter = 60

	case strings.Contains(errorLower, "model") && strings.Contains(errorLower, "not found"):
		err.Type = ErrorTypeModelError
		err.Code = CodeModelNotFound
		err.Message = "Model not found or not available"
		err.HTTPStatus = http.StatusNotFound
		err.Retryable = false

	case strings.Contains(errorLower, "timeout"):
		err.Type = ErrorTypeTimeout
		err.Code = CodeRequestTimeout
		err.Message = "Request timeout"
		err.HTTPStatus = http.StatusRequestTimeout
		err.Retryable = true

	case strings.Contains(errorLower, "content filter") || strings.Contains(errorLower, "safety"):
		err.Type = ErrorTypeSafetyError
		err.Code = CodeContentFiltered
		err.Message = "Content filtered by safety system"
		err.HTTPStatus = http.StatusBadRequest
		err.Retryable = false

	default:
		err.Type = ErrorTypeProviderError
		err.Code = CodeConnectionFailed
		err.Message = "Azure OpenAI service error"
		err.HTTPStatus = http.StatusBadGateway
		err.Retryable = true
	}
}

func (eh *ErrorHandler) handleAWSBedrockError(err *ProviderError, errorLower string) {
	switch {
	case strings.Contains(errorLower, "validationexception"):
		err.Type = ErrorTypeValidationError
		err.Code = CodeInvalidParameter
		err.Message = "Invalid request parameters"
		err.HTTPStatus = http.StatusBadRequest
		err.Retryable = false

	case strings.Contains(errorLower, "accessdeniedexception") || strings.Contains(errorLower, "unauthorized"):
		err.Type = ErrorTypeAuthorization
		err.Code = CodeInvalidAPIKey
		err.Message = "Access denied or invalid credentials"
		err.HTTPStatus = http.StatusUnauthorized
		err.Retryable = false

	case strings.Contains(errorLower, "throttlingexception") || strings.Contains(errorLower, "rate"):
		err.Type = ErrorTypeRateLimit
		err.Code = CodeServerOverloaded
		err.Message = "Request throttled by AWS Bedrock"
		err.HTTPStatus = http.StatusTooManyRequests
		err.Retryable = true
		err.RetryAfter = 30

	case strings.Contains(errorLower, "model identifier is invalid") || strings.Contains(errorLower, "model not found"):
		err.Type = ErrorTypeModelError
		err.Code = CodeModelNotFound
		err.Message = "Model identifier is invalid or model not available"
		err.HTTPStatus = http.StatusNotFound
		err.Retryable = false

	case strings.Contains(errorLower, "serviceexception"):
		err.Type = ErrorTypeProviderError
		err.Code = CodeProviderUnavailable
		err.Message = "AWS Bedrock service error"
		err.HTTPStatus = http.StatusBadGateway
		err.Retryable = true

	case strings.Contains(errorLower, "timeout"):
		err.Type = ErrorTypeTimeout
		err.Code = CodeRequestTimeout
		err.Message = "Request timeout"
		err.HTTPStatus = http.StatusRequestTimeout
		err.Retryable = true

	default:
		err.Type = ErrorTypeProviderError
		err.Code = CodeConnectionFailed
		err.Message = "AWS Bedrock connection error"
		err.HTTPStatus = http.StatusBadGateway
		err.Retryable = true
	}
}

func (eh *ErrorHandler) handleGoogleAIError(err *ProviderError, errorLower string) {
	switch {
	case strings.Contains(errorLower, "api key not valid") || strings.Contains(errorLower, "unauthorized"):
		err.Type = ErrorTypeAuthentication
		err.Code = CodeInvalidAPIKey
		err.Message = "Invalid API key provided"
		err.HTTPStatus = http.StatusUnauthorized
		err.Retryable = false

	case strings.Contains(errorLower, "quota") || strings.Contains(errorLower, "limit exceeded"):
		err.Type = ErrorTypeQuotaExceeded
		err.Code = CodeInsufficientQuota
		err.Message = "API quota exceeded"
		err.HTTPStatus = http.StatusPaymentRequired
		err.Retryable = false

	case strings.Contains(errorLower, "rate limit") || strings.Contains(errorLower, "too many requests"):
		err.Type = ErrorTypeRateLimit
		err.Code = CodeServerOverloaded
		err.Message = "Rate limit exceeded"
		err.HTTPStatus = http.StatusTooManyRequests
		err.Retryable = true
		err.RetryAfter = 60

	case strings.Contains(errorLower, "model not found") || strings.Contains(errorLower, "invalid model"):
		err.Type = ErrorTypeModelError
		err.Code = CodeModelNotFound
		err.Message = "Model not found or not available"
		err.HTTPStatus = http.StatusNotFound
		err.Retryable = false

	case strings.Contains(errorLower, "safety") || strings.Contains(errorLower, "blocked"):
		err.Type = ErrorTypeSafetyError
		err.Code = CodeContentFiltered
		err.Message = "Content blocked by safety filters"
		err.HTTPStatus = http.StatusBadRequest
		err.Retryable = false

	case strings.Contains(errorLower, "timeout"):
		err.Type = ErrorTypeTimeout
		err.Code = CodeRequestTimeout
		err.Message = "Request timeout"
		err.HTTPStatus = http.StatusRequestTimeout
		err.Retryable = true

	default:
		err.Type = ErrorTypeProviderError
		err.Code = CodeConnectionFailed
		err.Message = "Google AI service error"
		err.HTTPStatus = http.StatusBadGateway
		err.Retryable = true
	}
}

func (eh *ErrorHandler) handleGenericError(err *ProviderError, errorLower string) {
	switch {
	case strings.Contains(errorLower, "timeout"):
		err.Type = ErrorTypeTimeout
		err.Code = CodeRequestTimeout
		err.Message = "Request timeout"
		err.HTTPStatus = http.StatusRequestTimeout
		err.Retryable = true

	case strings.Contains(errorLower, "connection") || strings.Contains(errorLower, "network"):
		err.Type = ErrorTypeNetworkError
		err.Code = CodeConnectionFailed
		err.Message = "Network connection error"
		err.HTTPStatus = http.StatusBadGateway
		err.Retryable = true

	default:
		err.Type = ErrorTypeInternalError
		err.Code = CodeConnectionFailed
		err.Message = "Internal server error"
		err.HTTPStatus = http.StatusInternalServerError
		err.Retryable = false
	}
}

// LogError logs the error with appropriate detail level
func (eh *ErrorHandler) LogError(err *ProviderError, requestID string, additionalContext map[string]interface{}) {
	if !eh.config.Logging.LogRequests {
		return
	}

	logData := map[string]interface{}{
		"request_id":  requestID,
		"provider":    err.Provider,
		"error_type":  err.Type,
		"error_code":  err.Code,
		"message":     err.Message,
		"retryable":   err.Retryable,
		"http_status": err.HTTPStatus,
	}

	// Add additional context
	for k, v := range additionalContext {
		logData[k] = v
	}

	// Add details if present
	if len(err.Details) > 0 {
		logData["details"] = err.Details
	}

	// Convert to JSON for structured logging
	if eh.config.Logging.Format == "json" {
		if jsonData, jsonErr := json.Marshal(logData); jsonErr == nil {
			fmt.Printf("ERROR: %s\n", string(jsonData))
			return
		}
	}

	// Fallback to simple text logging
	fmt.Printf("ERROR [%s]: %s (%s/%s) - %s\n",
		requestID, err.Provider, err.Type, err.Code, err.Message)
}
