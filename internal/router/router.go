package router

import (
	"fmt"
	"strings"

	"llm-freeway/internal/config"
)

// ModelRoute contains routing information for a model
type ModelRoute struct {
	Provider     config.ProviderType `json:"provider"`
	OriginalName string              `json:"original_name"`
	DisplayName  string              `json:"display_name"`
	Description  string              `json:"description"`
	MaxTokens    int                 `json:"max_tokens"`
	SupportsStreaming bool           `json:"supports_streaming"`
}

// Router handles model detection and routing
type Router struct {
	config *config.Config
	routes map[string]ModelRoute
}

// NewRouter creates a new model router
func NewRouter(cfg *config.Config) *Router {
	router := &Router{
		config: cfg,
		routes: make(map[string]ModelRoute),
	}
	
	router.initializeRoutes()
	return router
}

// DetectProvider determines the provider from a model name (explicit routing only)
func (r *Router) DetectProvider(model string) (config.ProviderType, error) {
	// Only handle explicitly prefixed model names (e.g., "azure-openai/gpt-4", "aws-bedrock/claude-3", "google-ai/gemini-1.5-pro")
	if strings.Contains(model, "/") {
		parts := strings.SplitN(model, "/", 2)
		prefix := strings.ToLower(parts[0])
		
		switch prefix {
		case "azure-openai":
			return config.ProviderAzureOpenAI, nil
		case "aws-bedrock":
			return config.ProviderAWSBedrock, nil
		case "google-ai":
			return config.ProviderGoogleAI, nil
		case "anthropic":
			return config.ProviderAnthropic, nil
		default:
			return "", fmt.Errorf("unsupported provider prefix '%s' in model '%s' - supported prefixes: azure-openai, aws-bedrock, google-ai, anthropic", prefix, model)
		}
	}
	
	// No provider prefix found
	return "", fmt.Errorf("model '%s' requires explicit provider prefix (e.g., 'azure-openai/%s', 'aws-bedrock/%s', 'google-ai/%s', 'anthropic/%s')", 
		model, model, model, model, model)
}


// GetAvailableModels returns all available models for enabled providers
func (r *Router) GetAvailableModels() map[config.ProviderType][]ModelRoute {
	result := make(map[config.ProviderType][]ModelRoute)
	
	for _, route := range r.routes {
		if r.config.IsProviderEnabled(route.Provider) {
			result[route.Provider] = append(result[route.Provider], route)
		}
	}
	
	return result
}

// NormalizeModelName converts prefixed model names to their canonical form
func (r *Router) NormalizeModelName(model string) string {
	// Remove provider prefix if present
	if strings.Contains(model, "/") {
		parts := strings.SplitN(model, "/", 2)
		if len(parts) == 2 {
			return parts[1]
		}
	}
	return model
}

// ValidateModel checks if a model is supported and enabled (requires explicit provider prefix)
func (r *Router) ValidateModel(model string) error {
	provider, err := r.DetectProvider(model)
	if err != nil {
		return err
	}
	
	if !r.config.IsProviderEnabled(provider) {
		return fmt.Errorf("provider %s is not enabled or configured", provider)
	}
	
	normalizedModel := r.NormalizeModelName(model)
	
	// Check if model exists in our routes - EXACT MATCH ONLY
	if _, exists := r.routes[normalizedModel]; !exists {
		return fmt.Errorf("model %s is not supported for provider %s", normalizedModel, provider)
	}
	
	return nil
}


// Private methods

func (r *Router) initializeRoutes() {
	// Azure OpenAI models
	r.routes["gpt-35-turbo"] = ModelRoute{
		Provider:          config.ProviderAzureOpenAI,
		OriginalName:      "gpt-35-turbo",
		DisplayName:       "GPT-3.5 Turbo",
		Description:       "Fast, efficient model for simple tasks",
		MaxTokens:         4096,
		SupportsStreaming: true,
	}
	r.routes["gpt-4"] = ModelRoute{
		Provider:          config.ProviderAzureOpenAI,
		OriginalName:      "gpt-4",
		DisplayName:       "GPT-4",
		Description:       "Most capable GPT-4 model for complex tasks",
		MaxTokens:         8192,
		SupportsStreaming: true,
	}
	r.routes["gpt-4-turbo"] = ModelRoute{
		Provider:          config.ProviderAzureOpenAI,
		OriginalName:      "gpt-4-turbo",
		DisplayName:       "GPT-4 Turbo",
		Description:       "Latest GPT-4 model with improved speed",
		MaxTokens:         128000,
		SupportsStreaming: true,
	}
	r.routes["gpt-4o"] = ModelRoute{
		Provider:          config.ProviderAzureOpenAI,
		OriginalName:      "gpt-4o",
		DisplayName:       "GPT-4o",
		Description:       "Omni-modal GPT-4 model",
		MaxTokens:         128000,
		SupportsStreaming: true,
	}
	r.routes["gpt-4o-mini"] = ModelRoute{
		Provider:          config.ProviderAzureOpenAI,
		OriginalName:      "gpt-4o-mini",
		DisplayName:       "GPT-4o Mini",
		Description:       "Smaller, faster version of GPT-4o",
		MaxTokens:         128000,
		SupportsStreaming: true,
	}
	
	// AWS Bedrock (Anthropic) models
	r.routes["anthropic.claude-3-haiku-20240307-v1:0"] = ModelRoute{
		Provider:          config.ProviderAWSBedrock,
		OriginalName:      "anthropic.claude-3-haiku-20240307-v1:0",
		DisplayName:       "Claude 3 Haiku",
		Description:       "Fast and efficient model for simple tasks",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["anthropic.claude-3-sonnet-20240229-v1:0"] = ModelRoute{
		Provider:          config.ProviderAWSBedrock,
		OriginalName:      "anthropic.claude-3-sonnet-20240229-v1:0",
		DisplayName:       "Claude 3 Sonnet",
		Description:       "Balanced model for most use cases",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["anthropic.claude-3-opus-20240229-v1:0"] = ModelRoute{
		Provider:          config.ProviderAWSBedrock,
		OriginalName:      "anthropic.claude-3-opus-20240229-v1:0",
		DisplayName:       "Claude 3 Opus",
		Description:       "Most capable Claude model for complex reasoning",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["anthropic.claude-3-5-sonnet-20240620-v1:0"] = ModelRoute{
		Provider:          config.ProviderAWSBedrock,
		OriginalName:      "anthropic.claude-3-5-sonnet-20240620-v1:0",
		DisplayName:       "Claude 3.5 Sonnet",
		Description:       "Enhanced Claude 3.5 model with improved capabilities",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["anthropic.claude-3-5-sonnet-20241022-v2:0"] = ModelRoute{
		Provider:          config.ProviderAWSBedrock,
		OriginalName:      "anthropic.claude-3-5-sonnet-20241022-v2:0",
		DisplayName:       "Claude 3.5 Sonnet v2",
		Description:       "Latest Claude 3.5 model with latest improvements",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	
	// Google AI models
	r.routes["gemini-1.5-pro"] = ModelRoute{
		Provider:          config.ProviderGoogleAI,
		OriginalName:      "gemini-1.5-pro",
		DisplayName:       "Gemini 1.5 Pro",
		Description:       "Most capable Gemini model for complex tasks",
		MaxTokens:         2097152, // 2M tokens
		SupportsStreaming: true,
	}
	r.routes["gemini-1.5-flash"] = ModelRoute{
		Provider:          config.ProviderGoogleAI,
		OriginalName:      "gemini-1.5-flash",
		DisplayName:       "Gemini 1.5 Flash",
		Description:       "Fast and efficient Gemini model",
		MaxTokens:         1048576, // 1M tokens
		SupportsStreaming: true,
	}
	r.routes["gemini-2.0-flash-exp"] = ModelRoute{
		Provider:          config.ProviderGoogleAI,
		OriginalName:      "gemini-2.0-flash-exp",
		DisplayName:       "Gemini 2.0 Flash (Experimental)",
		Description:       "Experimental next-generation Gemini model",
		MaxTokens:         1048576, // 1M tokens
		SupportsStreaming: true,
	}
	r.routes["gemini-1.0-pro"] = ModelRoute{
		Provider:          config.ProviderGoogleAI,
		OriginalName:      "gemini-1.0-pro",
		DisplayName:       "Gemini 1.0 Pro",
		Description:       "Original Gemini Pro model",
		MaxTokens:         32768,
		SupportsStreaming: true,
	}
	
	// Direct Anthropic models
	r.routes["claude-3-5-sonnet-20241022"] = ModelRoute{
		Provider:          config.ProviderAnthropic,
		OriginalName:      "claude-3-5-sonnet-20241022",
		DisplayName:       "Claude 3.5 Sonnet",
		Description:       "Latest Claude 3.5 model via direct Anthropic API",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["claude-3-5-haiku-20241022"] = ModelRoute{
		Provider:          config.ProviderAnthropic,
		OriginalName:      "claude-3-5-haiku-20241022",
		DisplayName:       "Claude 3.5 Haiku",
		Description:       "Fast and efficient Claude 3.5 model via direct Anthropic API",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["claude-3-opus-20240229"] = ModelRoute{
		Provider:          config.ProviderAnthropic,
		OriginalName:      "claude-3-opus-20240229",
		DisplayName:       "Claude 3 Opus",
		Description:       "Most capable Claude model via direct Anthropic API",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["claude-3-sonnet-20240229"] = ModelRoute{
		Provider:          config.ProviderAnthropic,
		OriginalName:      "claude-3-sonnet-20240229",
		DisplayName:       "Claude 3 Sonnet",
		Description:       "Balanced Claude model via direct Anthropic API",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
	r.routes["claude-3-haiku-20240307"] = ModelRoute{
		Provider:          config.ProviderAnthropic,
		OriginalName:      "claude-3-haiku-20240307",
		DisplayName:       "Claude 3 Haiku",
		Description:       "Fast Claude model via direct Anthropic API",
		MaxTokens:         200000,
		SupportsStreaming: true,
	}
}

