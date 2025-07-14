package router

import (
	"testing"

	"llm-freeway/internal/config"

	"github.com/stretchr/testify/assert"
)

func TestRouter(t *testing.T) {
	cfg := &config.Config{
		Providers: map[config.ProviderType]config.ProviderConfig{
			config.ProviderAzureOpenAI: {Enabled: true},
			config.ProviderAWSBedrock:  {Enabled: true},
			config.ProviderGoogleAI:    {Enabled: true},
		},
	}
	router := NewRouter(cfg)

	t.Run("exact model matches are accepted", func(t *testing.T) {
		// Test exact matches for known models
		validModels := []struct {
			prefixedModel string
			provider      config.ProviderType
		}{
			{"azure-openai/gpt-4", config.ProviderAzureOpenAI},
			{"aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:0", config.ProviderAWSBedrock},
			{"google-ai/gemini-1.5-pro", config.ProviderGoogleAI},
		}

		for _, test := range validModels {
			t.Run(test.prefixedModel, func(t *testing.T) {
				err := router.ValidateModel(test.prefixedModel)
				assert.NoError(t, err, "Exact model match should be valid")
			})
		}
	})

	t.Run("pattern matches are rejected - no fuzzy matching", func(t *testing.T) {
		// Test that partial matches are rejected
		invalidModels := []string{
			"azure-openai/gpt-5",        // Not in our exact list
			"azure-openai/gpt",          // Partial match should fail
			"aws-bedrock/claude",        // Partial match should fail
			"google-ai/gemini",          // Partial match should fail
			"azure-openai/davinci-003",  // Pattern match should fail
			"aws-bedrock/claude-4",      // Non-existent version should fail
			"google-ai/palm-2",          // Pattern match should fail
		}

		for _, model := range invalidModels {
			t.Run(model, func(t *testing.T) {
				err := router.ValidateModel(model)
				assert.Error(t, err, "Pattern/partial matches should be rejected")
				assert.Contains(t, err.Error(), "not supported", "Error should indicate model is not supported")
			})
		}
	})

	t.Run("provider detection works correctly", func(t *testing.T) {
		tests := []struct {
			model            string
			expectedProvider config.ProviderType
		}{
			{"azure-openai/gpt-4", config.ProviderAzureOpenAI},
			{"aws-bedrock/claude-3", config.ProviderAWSBedrock},
			{"google-ai/gemini-pro", config.ProviderGoogleAI},
		}

		for _, test := range tests {
			t.Run(test.model, func(t *testing.T) {
				provider, err := router.DetectProvider(test.model)
				assert.NoError(t, err)
				assert.Equal(t, test.expectedProvider, provider)
			})
		}
	})

	t.Run("model normalization removes prefixes", func(t *testing.T) {
		tests := []struct {
			input    string
			expected string
		}{
			{"azure-openai/gpt-4", "gpt-4"},
			{"aws-bedrock/claude-3", "claude-3"},
			{"google-ai/gemini-pro", "gemini-pro"},
			{"gpt-4", "gpt-4"}, // No prefix
		}

		for _, test := range tests {
			t.Run(test.input, func(t *testing.T) {
				normalized := router.NormalizeModelName(test.input)
				assert.Equal(t, test.expected, normalized)
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

		err := disabledRouter.ValidateModel("azure-openai/gpt-4")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not enabled or configured")
	})

	t.Run("models without provider prefix are rejected", func(t *testing.T) {
		// Models without explicit provider prefix should be rejected
		modelsWithoutPrefix := []string{
			"gpt-4",
			"claude-3-sonnet",
			"gemini-pro",
		}

		for _, model := range modelsWithoutPrefix {
			t.Run(model, func(t *testing.T) {
				err := router.ValidateModel(model)
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "requires explicit provider prefix")
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
				assert.NotEmpty(t, route.OriginalName, "OriginalName should not be empty")
				assert.NotEmpty(t, route.DisplayName, "DisplayName should not be empty")
				assert.NotEmpty(t, route.Description, "Description should not be empty")
				assert.Greater(t, route.MaxTokens, 0, "MaxTokens should be positive")
				assert.Equal(t, provider, route.Provider, "Provider should match")
				
				// Verify SupportsStreaming is set
				assert.True(t, route.SupportsStreaming, "All current models should support streaming")
			}
		}
	})
}

func TestSecurityValidation(t *testing.T) {
	cfg := &config.Config{
		Providers: map[config.ProviderType]config.ProviderConfig{
			config.ProviderAzureOpenAI: {Enabled: true},
			config.ProviderAWSBedrock:  {Enabled: true},
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
		// Verify that ONLY the exact models we defined are accepted
		// This test ensures no pattern matching bypasses exist
		
		acceptedModels := []string{
			"azure-openai/gpt-35-turbo",
			"azure-openai/gpt-4",
			"azure-openai/gpt-4-turbo", 
			"azure-openai/gpt-4o",
			"azure-openai/gpt-4o-mini",
			"aws-bedrock/anthropic.claude-3-haiku-20240307-v1:0",
			"aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
			"aws-bedrock/anthropic.claude-3-opus-20240229-v1:0",
			"aws-bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
			"aws-bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
			"google-ai/gemini-1.5-pro",
			"google-ai/gemini-1.5-flash",
			"google-ai/gemini-2.0-flash-exp",
			"google-ai/gemini-1.0-pro",
		}

		// Test that all our defined models are accepted
		for _, model := range acceptedModels {
			t.Run(model, func(t *testing.T) {
				err := router.ValidateModel(model)
				assert.NoError(t, err, "Whitelisted model should be accepted")
			})
		}

		// Test that similar but not exact models are rejected
		rejectedSimilar := []string{
			"azure-openai/gpt-4-turbo-preview", // Similar but not exact
			"aws-bedrock/claude-3-sonnet",      // Missing version suffix
			"google-ai/gemini-pro",             // Missing version number
			"azure-openai/GPT-4",               // Wrong case
			"aws-bedrock/anthropic.claude-3-sonnet-20240229-v1:1", // Wrong version
		}

		for _, model := range rejectedSimilar {
			t.Run("reject_"+model, func(t *testing.T) {
				err := router.ValidateModel(model)
				assert.Error(t, err, "Non-exact model should be rejected")
			})
		}
	})
}