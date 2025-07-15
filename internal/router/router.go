package router

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"llm-freeway/internal/config"
	"gopkg.in/yaml.v3"
)

// ModelRoute contains routing information for a model
type ModelRoute struct {
	Provider          config.ProviderType `json:"provider" yaml:"provider"`
	ModelName         string              `json:"model_name" yaml:"model_name"`         // Actual model name for API calls
	RouteName         string              `json:"route_name" yaml:"route_name"`         // Custom route name for requests
	DisplayName       string              `json:"display_name" yaml:"display_name"`
	Description       string              `json:"description" yaml:"description"`
	MaxTokens         int                 `json:"max_tokens" yaml:"max_tokens"`
	SupportsStreaming bool                `json:"supports_streaming" yaml:"supports_streaming"`
	Enabled           bool                `json:"enabled" yaml:"enabled"`
	RateLimit         int                 `json:"rate_limit" yaml:"rate_limit"`
}

// ModelConfig represents the structure of the models.yaml file
type ModelConfig struct {
	Models map[string]ModelConfigEntry `yaml:"models"`
}

// ModelConfigEntry represents a model entry in the config file
type ModelConfigEntry struct {
	Provider          string `yaml:"provider"`
	ModelName         string `yaml:"model_name"`
	DisplayName       string `yaml:"display_name"`
	Description       string `yaml:"description"`
	MaxTokens         int    `yaml:"max_tokens"`
	SupportsStreaming bool   `yaml:"supports_streaming"`
	Enabled           bool   `yaml:"enabled"`
	RateLimit         int    `yaml:"rate_limit"`
}

// Router handles model detection and routing
type Router struct {
	config      *config.Config
	routes      map[string]ModelRoute
	modelConfig *ModelConfig
}

// NewRouter creates a new model router
func NewRouter(cfg *config.Config) *Router {
	router := &Router{
		config: cfg,
		routes: make(map[string]ModelRoute),
	}
	
	if err := router.loadModelConfig(); err != nil {
		log.Fatalf("Failed to load models configuration: %v. Please ensure models.yaml exists in the current directory.", err)
	}
	
	router.initializeRoutesFromConfig()
	return router
}

// DetectProvider determines the provider from a route name
func (r *Router) DetectProvider(routeName string) (config.ProviderType, error) {
	// Look up the route in our configuration
	if route, exists := r.routes[routeName]; exists {
		return route.Provider, nil
	}
	
	return "", fmt.Errorf("route '%s' not found in model configuration", routeName)
}


// GetAvailableModels returns all available models for enabled providers
func (r *Router) GetAvailableModels() map[config.ProviderType][]ModelRoute {
	result := make(map[config.ProviderType][]ModelRoute)
	
	for _, route := range r.routes {
		// Check if provider is enabled in config AND model is enabled
		if r.config.IsProviderEnabled(route.Provider) && route.Enabled {
			result[route.Provider] = append(result[route.Provider], route)
		}
	}
	
	return result
}

// GetModelName returns the actual model name for API calls from a route name
func (r *Router) GetModelName(routeName string) (string, error) {
	if route, exists := r.routes[routeName]; exists {
		return route.ModelName, nil
	}
	return "", fmt.Errorf("route '%s' not found in model configuration", routeName)
}

// ValidateModel checks if a route name is supported and enabled
func (r *Router) ValidateModel(routeName string) error {
	provider, err := r.DetectProvider(routeName)
	if err != nil {
		return err
	}
	
	if !r.config.IsProviderEnabled(provider) {
		return fmt.Errorf("provider %s is not enabled or configured", provider)
	}
	
	// Check if route exists in our routes - EXACT MATCH ONLY
	route, exists := r.routes[routeName]
	if !exists {
		return fmt.Errorf("route '%s' is not supported", routeName)
	}
	
	// Check if model is enabled
	if !route.Enabled {
		return fmt.Errorf("route '%s' is disabled", routeName)
	}
	
	return nil
}


// Private methods

// loadModelConfig loads model configuration from models.yaml
func (r *Router) loadModelConfig() error {
	// Try to find models.yaml in multiple locations relative to the project root
	possiblePaths := []string{
		"models.yaml",                    // Current directory
		"./models.yaml",                 // Explicit current directory
		"../../models.yaml",             // For tests running from subdirectories
		"../../../models.yaml",          // For deeply nested tests
		filepath.Join("..", "..", "models.yaml"), // Alternative path for tests
	}
	
	var configPath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			configPath = path
			break
		}
	}
	
	if configPath == "" {
		return fmt.Errorf("models.yaml not found in any of: %v", possiblePaths)
	}
	
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", configPath, err)
	}
	
	r.modelConfig = &ModelConfig{}
	if err := yaml.Unmarshal(data, r.modelConfig); err != nil {
		return fmt.Errorf("failed to parse %s: %w", configPath, err)
	}
	
	log.Printf("Loaded model configuration from %s with %d models", configPath, len(r.modelConfig.Models))
	return nil
}

// initializeRoutesFromConfig initializes routes from loaded config
func (r *Router) initializeRoutesFromConfig() {
	for modelName, modelEntry := range r.modelConfig.Models {
		// Convert provider string to ProviderType
		var provider config.ProviderType
		switch modelEntry.Provider {
		case "azure-openai":
			provider = config.ProviderAzureOpenAI
		case "openai":
			provider = config.ProviderOpenAI
		case "aws-bedrock":
			provider = config.ProviderAWSBedrock
		case "anthropic":
			provider = config.ProviderAnthropic
		case "google-ai":
			provider = config.ProviderGoogleAI
		default:
			log.Printf("Warning: Unknown provider '%s' for model '%s', skipping", modelEntry.Provider, modelName)
			continue
		}
		
		// Use the config key as the route name
		r.routes[modelName] = ModelRoute{
			Provider:          provider,
			ModelName:         modelEntry.ModelName,
			RouteName:         modelName, // The key from models.yaml
			DisplayName:       modelEntry.DisplayName,
			Description:       modelEntry.Description,
			MaxTokens:         modelEntry.MaxTokens,
			SupportsStreaming: modelEntry.SupportsStreaming,
			Enabled:           modelEntry.Enabled,
			RateLimit:         modelEntry.RateLimit,
		}
	}
	
	log.Printf("Initialized %d model routes from configuration", len(r.routes))
}


