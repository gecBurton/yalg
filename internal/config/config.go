package config

import (
	"os"
	"strconv"
	"time"
)

// ProviderType represents different AI providers
type ProviderType string

const (
	ProviderAzureOpenAI ProviderType = "azure-openai"
	ProviderOpenAI      ProviderType = "openai"
	ProviderAWSBedrock  ProviderType = "aws-bedrock"
	ProviderGoogleAI    ProviderType = "google-ai"
	ProviderAnthropic   ProviderType = "anthropic"
)

// ProviderConfig holds configuration for each provider
type ProviderConfig struct {
	Enabled       bool          `json:"enabled"`
	APIKey        string        `json:"-"` // Hidden in JSON for security
	Endpoint      string        `json:"endpoint,omitempty"`
	Region        string        `json:"region,omitempty"`
	RateLimit     int           `json:"rate_limit"`     // Requests per minute
	Timeout       time.Duration `json:"timeout"`        // Request timeout
	MaxTokens     int           `json:"max_tokens"`     // Maximum tokens per request
	RetryAttempts int           `json:"retry_attempts"` // Number of retry attempts
	RetryDelay    time.Duration `json:"retry_delay"`    // Delay between retries
}

// Config holds the complete application configuration
type Config struct {
	Server    ServerConfig                    `json:"server"`
	Database  DatabaseConfig                  `json:"database"`
	Auth      AuthConfig                      `json:"auth"`
	Providers map[ProviderType]ProviderConfig `json:"providers"`
	Metrics   MetricsConfig                   `json:"metrics"`
	Logging   LoggingConfig                   `json:"logging"`
}

// ServerConfig holds server-specific configuration
type ServerConfig struct {
	Port           string        `json:"port"`
	ReadTimeout    time.Duration `json:"read_timeout"`
	WriteTimeout   time.Duration `json:"write_timeout"`
	MaxRequestSize int64         `json:"max_request_size"`
	EnableCORS     bool          `json:"enable_cors"`
	TrustedProxies []string      `json:"trusted_proxies"`
	UITheme        string        `json:"ui_theme"`
}

// MetricsConfig holds metrics and monitoring configuration
type MetricsConfig struct {
	Enabled        bool          `json:"enabled"`
	RetentionDays  int           `json:"retention_days"`
	UpdateInterval time.Duration `json:"update_interval"`
	ExportPath     string        `json:"export_path"`
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	Enabled      bool          `json:"enabled"`
	Host         string        `json:"host"`
	Port         int           `json:"port"`
	User         string        `json:"user"`
	Password     string        `json:"-"` // Hidden in JSON for security
	Database     string        `json:"database"`
	SSLMode      string        `json:"ssl_mode"`
	MaxIdleConns int           `json:"max_idle_conns"`
	MaxOpenConns int           `json:"max_open_conns"`
	MaxLifetime  time.Duration `json:"max_lifetime"`
}

// LoggingConfig holds logging configuration
type LoggingConfig struct {
	Level         string `json:"level"`           // debug, info, warn, error
	Format        string `json:"format"`          // json, text
	LogRequests   bool   `json:"log_requests"`    // SECURITY: Log metadata only, never content
	LogResponses  bool   `json:"log_responses"`   // SECURITY: Log metadata only, never content
	LogTokenUsage bool   `json:"log_token_usage"` // Log token counts only, never content
}

// AuthConfig holds authentication configuration
type AuthConfig struct {
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"-"` // Hidden in JSON for security
	RedirectURI  string `json:"redirect_uri"`
	BaseURL      string `json:"base_url"`
	IssuerURL    string `json:"issuer_url"`
}

// LoadConfig loads configuration from environment variables with defaults
func LoadConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Port:           getEnv("PORT", "8000"),
			ReadTimeout:    getDurationEnv("SERVER_READ_TIMEOUT", 30*time.Second),
			WriteTimeout:   getDurationEnv("SERVER_WRITE_TIMEOUT", 30*time.Second),
			MaxRequestSize: getInt64Env("MAX_REQUEST_SIZE", 10*1024*1024), // 10MB
			EnableCORS:     getBoolEnv("ENABLE_CORS", true),
			TrustedProxies: []string{"127.0.0.1", "::1"},
			UITheme:        getEnv("UI_THEME", "bootstrap"), // "bootstrap" theme only
		},
		Database: DatabaseConfig{
			Enabled:      getBoolEnv("DB_ENABLED", true),
			Host:         getEnv("DB_HOST", "localhost"),
			Port:         getIntEnv("DB_PORT", 5432),
			User:         getEnv("DB_USER", "llm_freeway"),
			Password:     getEnv("DB_PASSWORD", "password"),
			Database:     getEnv("DB_NAME", "llm_freeway"),
			SSLMode:      getEnv("DB_SSLMODE", "disable"),
			MaxIdleConns: getIntEnv("DB_MAX_IDLE_CONNS", 10),
			MaxOpenConns: getIntEnv("DB_MAX_OPEN_CONNS", 100),
			MaxLifetime:  getDurationEnv("DB_MAX_LIFETIME", 1*time.Hour),
		},
		Auth: AuthConfig{
			ClientID:     getEnv("OIDC_CLIENT_ID", ""),
			ClientSecret: getEnv("OIDC_CLIENT_SECRET", ""),
			RedirectURI:  getEnv("OIDC_REDIRECT_URI", "http://localhost:8000/callback/"),
			BaseURL:      getEnv("BASE_URL", "http://localhost:8000"),
			IssuerURL:    getEnv("OIDC_ISSUER_URL", ""),
		},
		Providers: map[ProviderType]ProviderConfig{
			ProviderAzureOpenAI: {
				Enabled:       getBoolEnv("AZURE_OPENAI_ENABLED", true), // Can be enabled even without API key
				APIKey:        getEnv("AZURE_OPENAI_API_KEY", ""),
				Endpoint:      getEnv("AZURE_OPENAI_ENDPOINT", ""),
				RateLimit:     getIntEnv("AZURE_OPENAI_RATE_LIMIT", 60),
				Timeout:       getDurationEnv("AZURE_OPENAI_TIMEOUT", 30*time.Second),
				MaxTokens:     getIntEnv("AZURE_OPENAI_MAX_TOKENS", 4096),
				RetryAttempts: getIntEnv("AZURE_OPENAI_RETRY_ATTEMPTS", 3),
				RetryDelay:    getDurationEnv("AZURE_OPENAI_RETRY_DELAY", 1*time.Second),
			},
			ProviderOpenAI: {
				Enabled:       getBoolEnv("OPENAI_ENABLED", true), // Can be enabled even without API key
				APIKey:        getEnv("OPENAI_API_KEY", ""),
				RateLimit:     getIntEnv("OPENAI_RATE_LIMIT", 60),
				Timeout:       getDurationEnv("OPENAI_TIMEOUT", 30*time.Second),
				MaxTokens:     getIntEnv("OPENAI_MAX_TOKENS", 4096),
				RetryAttempts: getIntEnv("OPENAI_RETRY_ATTEMPTS", 3),
				RetryDelay:    getDurationEnv("OPENAI_RETRY_DELAY", 1*time.Second),
			},
			ProviderAWSBedrock: {
				Enabled:       true, // Enabled if AWS credentials are available
				Region:        getEnv("AWS_REGION", "us-east-1"),
				RateLimit:     getIntEnv("AWS_BEDROCK_RATE_LIMIT", 30),
				Timeout:       getDurationEnv("AWS_BEDROCK_TIMEOUT", 45*time.Second),
				MaxTokens:     getIntEnv("AWS_BEDROCK_MAX_TOKENS", 4096),
				RetryAttempts: getIntEnv("AWS_BEDROCK_RETRY_ATTEMPTS", 3),
				RetryDelay:    getDurationEnv("AWS_BEDROCK_RETRY_DELAY", 2*time.Second),
			},
			ProviderGoogleAI: {
				Enabled:       getBoolEnv("GEMINI_ENABLED", true), // Can be enabled even without API key
				APIKey:        getEnv("GEMINI_API_KEY", ""),
				RateLimit:     getIntEnv("GEMINI_RATE_LIMIT", 60),
				Timeout:       getDurationEnv("GEMINI_TIMEOUT", 30*time.Second),
				MaxTokens:     getIntEnv("GEMINI_MAX_TOKENS", 8192),
				RetryAttempts: getIntEnv("GEMINI_RETRY_ATTEMPTS", 3),
				RetryDelay:    getDurationEnv("GEMINI_RETRY_DELAY", 1*time.Second),
			},
			ProviderAnthropic: {
				Enabled:       getBoolEnv("ANTHROPIC_ENABLED", true), // Can be enabled even without API key
				APIKey:        getEnv("ANTHROPIC_API_KEY", ""),
				RateLimit:     getIntEnv("ANTHROPIC_RATE_LIMIT", 60),
				Timeout:       getDurationEnv("ANTHROPIC_TIMEOUT", 30*time.Second),
				MaxTokens:     getIntEnv("ANTHROPIC_MAX_TOKENS", 8192),
				RetryAttempts: getIntEnv("ANTHROPIC_RETRY_ATTEMPTS", 3),
				RetryDelay:    getDurationEnv("ANTHROPIC_RETRY_DELAY", 1*time.Second),
			},
		},
		Metrics: MetricsConfig{
			Enabled:        getBoolEnv("METRICS_ENABLED", true),
			RetentionDays:  getIntEnv("METRICS_RETENTION_DAYS", 30),
			UpdateInterval: getDurationEnv("METRICS_UPDATE_INTERVAL", 1*time.Minute),
			ExportPath:     getEnv("METRICS_EXPORT_PATH", "/tmp/llm-freeway-metrics.json"),
		},
		Logging: LoggingConfig{
			Level:         getEnv("LOG_LEVEL", "info"),
			Format:        getEnv("LOG_FORMAT", "text"),
			LogRequests:   false,                                // SECURITY: Never log request content
			LogResponses:  false,                                // SECURITY: Never log response content
			LogTokenUsage: getBoolEnv("LOG_TOKEN_USAGE", false), // Only log token counts if explicitly enabled
		},
	}
}

// IsProviderEnabled checks if a provider is enabled and configured
func (c *Config) IsProviderEnabled(provider ProviderType) bool {
	config, exists := c.Providers[provider]
	if !exists {
		return false
	}
	return config.Enabled
}

// GetProviderConfig returns configuration for a specific provider
func (c *Config) GetProviderConfig(provider ProviderType) (ProviderConfig, bool) {
	config, exists := c.Providers[provider]
	return config, exists && config.Enabled
}

// Helper functions for environment variable parsing
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getIntEnv(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getInt64Env(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseBool(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}
