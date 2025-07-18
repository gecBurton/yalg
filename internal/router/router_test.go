package router

import (
	"strings"
	"testing"

	"llm-freeway/internal/config"

	"github.com/stretchr/testify/assert"
)

func TestRouter(t *testing.T) {
	cfg := &config.Config{
		Providers: map[config.ProviderType]config.ProviderConfig{
			config.ProviderAzureOpenAI: {Enabled: true},
			config.ProviderAWSBedrock:  {Enabled: true},
			config.ProviderAnthropic:   {Enabled: true},
			config.ProviderGoogleAI:    {Enabled: true},
		},
	}
	router := NewRouter(cfg)

	t.Run("exact model matches are accepted", func(t *testing.T) {
		// Test exact matches for known models (from models.yaml)
		validModels := []struct {
			routeName string
			provider  config.ProviderType
		}{
			{"gpt-4o", config.ProviderAzureOpenAI},
			{"claude-3-sonnet-bedrock", config.ProviderAWSBedrock},
			{"gemini-1-5-pro", config.ProviderGoogleAI},
		}

		for _, test := range validModels {
			t.Run(test.routeName, func(t *testing.T) {
				err := router.ValidateModel(test.routeName)
				assert.NoError(t, err, "Exact model match should be valid")
			})
		}
	})

	t.Run("pattern matches are rejected - no fuzzy matching", func(t *testing.T) {
		// Test that partial matches are rejected
		invalidModels := []string{
			"gpt-5",       // Not in our exact list
			"gpt",         // Partial match should fail
			"claude",      // Partial match should fail
			"gemini",      // Partial match should fail
			"davinci-003", // Pattern match should fail
			"claude-4",    // Non-existent version should fail
			"palm-2",      // Pattern match should fail
		}

		for _, route := range invalidModels {
			t.Run(route, func(t *testing.T) {
				err := router.ValidateModel(route)
				assert.Error(t, err, "Non-existent routes should be rejected")
				assert.Contains(t, err.Error(), "not found", "Error should indicate route is not found")
			})
		}
	})

	t.Run("provider detection works correctly", func(t *testing.T) {
		tests := []struct {
			route            string
			expectedProvider config.ProviderType
		}{
			{"gpt-4o", config.ProviderAzureOpenAI},
			{"claude-3-sonnet-bedrock", config.ProviderAWSBedrock},
			{"gemini-1-5-pro", config.ProviderGoogleAI},
		}

		for _, test := range tests {
			t.Run(test.route, func(t *testing.T) {
				provider, err := router.DetectProvider(test.route)
				assert.NoError(t, err)
				assert.Equal(t, test.expectedProvider, provider)
			})
		}
	})

	t.Run("model normalization removes prefixes", func(t *testing.T) {
		tests := []struct {
			route    string
			expected string
		}{
			{"gpt-4o", "gpt-4o"},
			{"claude-3-sonnet-bedrock", "anthropic.claude-3-sonnet-20240229-v1:0"},
			{"gemini-1-5-pro", "gemini-1.5-pro"},
			{"my-favorite-claude", "claude-3-5-sonnet-20241022"},
		}

		for _, test := range tests {
			t.Run(test.route, func(t *testing.T) {
				modelName, err := router.GetModelName(test.route)
				assert.NoError(t, err)
				assert.Equal(t, test.expected, modelName)
			})
		}
	})

	t.Run("get available models returns only enabled providers", func(t *testing.T) {
		models := router.GetAvailableModels()

		// Should have models for all enabled providers
		assert.Contains(t, models, config.ProviderAzureOpenAI)
		assert.Contains(t, models, config.ProviderAWSBedrock)
		assert.Contains(t, models, config.ProviderGoogleAI)

		// Check that we have some models for each provider
		assert.NotEmpty(t, models[config.ProviderAzureOpenAI])
		assert.NotEmpty(t, models[config.ProviderAWSBedrock])
		assert.NotEmpty(t, models[config.ProviderGoogleAI])
	})

	t.Run("disabled providers are rejected", func(t *testing.T) {
		// Create config with disabled provider
		disabledCfg := &config.Config{
			Providers: map[config.ProviderType]config.ProviderConfig{
				config.ProviderAzureOpenAI: {Enabled: false}, // Disabled
				config.ProviderAWSBedrock:  {Enabled: true},
				config.ProviderGoogleAI:    {Enabled: true},
			},
		}
		disabledRouter := NewRouter(disabledCfg)

		err := disabledRouter.ValidateModel("gpt-4o")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not enabled or configured")
	})

	t.Run("non-existent routes are rejected", func(t *testing.T) {
		// Non-existent routes should be rejected
		nonExistentRoutes := []string{
			"non-existent-model",
			"fake-route",
			"invalid-model-name",
		}

		for _, route := range nonExistentRoutes {
			t.Run(route, func(t *testing.T) {
				err := router.ValidateModel(route)
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "not found")
			})
		}
	})
}

func TestTableNameConsistency(t *testing.T) {
	cfg := &config.Config{
		Providers: map[config.ProviderType]config.ProviderConfig{
			config.ProviderAzureOpenAI: {Enabled: true},
		},
	}
	router := NewRouter(cfg)

	t.Run("model routes have consistent data", func(t *testing.T) {
		models := router.GetAvailableModels()

		for provider, routes := range models {
			for _, route := range routes {
				// Verify all routes have required fields
				assert.NotEmpty(t, route.RouteName, "RouteName should not be empty")
				assert.NotEmpty(t, route.ModelName, "ModelName should not be empty")
				assert.NotEmpty(t, route.DisplayName, "DisplayName should not be empty")
				assert.NotEmpty(t, route.Description, "Description should not be empty")
				assert.Greater(t, route.MaxTokens, 0, "MaxTokens should be positive")
				assert.Greater(t, route.RateLimit, 0, "RateLimit should be positive")
				assert.Equal(t, provider, route.Provider, "Provider should match")

				// Verify SupportsStreaming is set correctly (embedding models don't support streaming)
				if strings.Contains(route.RouteName, "embedding") || strings.Contains(route.RouteName, "embed") {
					assert.False(t, route.SupportsStreaming, "Embedding models should not support streaming")
				} else {
					assert.True(t, route.SupportsStreaming, "Non-embedding models should support streaming")
				}
			}
		}
	})
}

func TestSecurityValidation(t *testing.T) {
	cfg := &config.Config{
		Providers: map[config.ProviderType]config.ProviderConfig{
			config.ProviderAzureOpenAI: {Enabled: true},
			config.ProviderAWSBedrock:  {Enabled: true},
			config.ProviderAnthropic:   {Enabled: true},
			config.ProviderGoogleAI:    {Enabled: true},
		},
	}
	router := NewRouter(cfg)

	t.Run("injection attempts are blocked", func(t *testing.T) {
		maliciousInputs := []string{
			"openai/gpt-4; DROP TABLE users;",
			"anthropic/../../../etc/passwd",
			"google/gemini-pro' OR '1'='1",
			"openai/gpt-4\x00hidden",
			"anthropic/claude-3\n\rmalicious",
		}

		for _, input := range maliciousInputs {
			t.Run(input, func(t *testing.T) {
				err := router.ValidateModel(input)
				assert.Error(t, err, "Malicious input should be rejected")
			})
		}
	})

	t.Run("only whitelisted exact models are accepted", func(t *testing.T) {
		// Verify that ONLY the exact routes we defined in models.yaml are accepted
		// This test ensures no pattern matching bypasses exist

		acceptedRoutes := []string{
			"gpt-35-turbo",
			"gpt-4o",
			"gpt-4o-mini",
			"gpt-4o-creative",
			"claude-3-5-sonnet-bedrock",
			"claude-3-sonnet-bedrock",
			"claude-3-haiku-bedrock",
			"claude-3-5-sonnet",
			"claude-3-sonnet",
			"claude-3-haiku",
			"my-favorite-claude",
			"gemini-1-5-pro",
			"gemini-1-5-flash",
		}

		// Test that all our defined routes are accepted
		for _, route := range acceptedRoutes {
			t.Run(route, func(t *testing.T) {
				err := router.ValidateModel(route)
				assert.NoError(t, err, "Whitelisted route should be accepted")
			})
		}

		// Test that similar but not exact routes are rejected
		rejectedSimilar := []string{
			"gpt-4",              // Route not in config
			"claude-sonnet",      // Partial route name
			"gemini-pro",         // Wrong route name
			"GPT-4O",             // Wrong case
			"claude-3-sonnet-v2", // Similar but different route
		}

		for _, route := range rejectedSimilar {
			t.Run("reject_"+route, func(t *testing.T) {
				err := router.ValidateModel(route)
				assert.Error(t, err, "Non-exact route should be rejected")
			})
		}
	})
}
