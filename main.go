package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"llm-freeway/internal/auth"
	"llm-freeway/internal/config"
	"llm-freeway/internal/database"
	"llm-freeway/internal/errors"
	"llm-freeway/internal/router"
	"llm-freeway/internal/server"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsConfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/joho/godotenv"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/azure"
	"github.com/openai/openai-go/option"
	"google.golang.org/genai"
)

const (
	defaultPort = "8000"
)

func main() {
	// Load .env file if it exists
	if err := godotenv.Load(); err != nil {
		log.Printf("No .env file found or failed to load: %v", err)
	}

	cfg := config.LoadConfig()
	port := getPort()

	// Initialize core components
	db := initializeDatabase(cfg)
	errorHandler := errors.NewErrorHandler(cfg)
	modelRouter := router.NewRouter(cfg)

	// Initialize authentication (always required)
	authMiddleware, authHandlers := initializeAuthentication(cfg, db)

	// Initialize provider clients
	azureOpenAIClient := initializeAzureOpenAI(cfg)
	directOpenAIClient := initializeOpenAI(cfg)
	bedrockClient := initializeAWSBedrock(cfg)
	geminiClient := initializeGoogleAI(cfg)
	awsCfg := initializeAWSConfig(cfg)

	// Create and configure server
	srv := server.NewServer(server.ServerConfig{
		Config:             cfg,
		BedrockClient:      bedrockClient,
		GeminiClient:       geminiClient,
		AzureOpenAIClient:  azureOpenAIClient,
		DirectOpenAIClient: directOpenAIClient,
		AnthropicAPIKey:    os.Getenv("ANTHROPIC_API_KEY"),
		AWSConfig:          awsCfg,
		Database:           db,
		ErrorHandler:       errorHandler,
		Router:             modelRouter,
	})

	setupRoutes(srv, authMiddleware, authHandlers)
	startBackgroundTasks(db)
	logServerStatus(cfg, db, port)

	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal("Server failed to start:", err)
	}
}

func getPort() string {
	if port := os.Getenv("PORT"); port != "" {
		return port
	}
	return defaultPort
}

func initializeDatabase(cfg *config.Config) *database.DB {
	db, err := database.NewDatabase(cfg)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	if err := db.Migrate(); err != nil {
		log.Fatalf("Failed to run database migrations: %v", err)
	}

	return db
}

func initializeAzureOpenAI(cfg *config.Config) *openai.Client {
	if !cfg.IsProviderEnabled(config.ProviderAzureOpenAI) {
		return nil
	}

	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")

	if apiKey == "" || endpoint == "" {
		log.Printf("Azure OpenAI enabled but missing configuration")
		return nil
	}

	apiVersion := os.Getenv("AZURE_OPENAI_API_VERSION")
	if apiVersion == "" {
		apiVersion = "2024-02-01"
	}

	client := openai.NewClient(
		azure.WithEndpoint(endpoint, apiVersion),
		azure.WithAPIKey(apiKey),
	)

	log.Printf("Azure OpenAI client initialized")
	return &client
}

func initializeOpenAI(cfg *config.Config) *openai.Client {
	if !cfg.IsProviderEnabled(config.ProviderOpenAI) {
		return nil
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Printf("OpenAI enabled but missing OPENAI_API_KEY")
		return nil
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	log.Printf("OpenAI client initialized")
	return &client
}

func initializeAWSBedrock(cfg *config.Config) *bedrockruntime.Client {
	if !cfg.IsProviderEnabled(config.ProviderAWSBedrock) {
		return nil
	}

	awsCfg, err := awsConfig.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Printf("Warning: Failed to load AWS config for Bedrock: %v", err)
		return nil
	}

	log.Printf("AWS Bedrock client initialized")
	return bedrockruntime.NewFromConfig(awsCfg)
}

func initializeAWSConfig(cfg *config.Config) *aws.Config {
	// Load AWS config for use with the unified Claude adapter
	if !cfg.IsProviderEnabled(config.ProviderAWSBedrock) {
		return nil
	}

	awsCfg, err := awsConfig.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Printf("Warning: Failed to load AWS config: %v", err)
		return nil
	}

	log.Printf("AWS config initialized for unified Claude adapter")
	return &awsCfg
}

func initializeGoogleAI(cfg *config.Config) *genai.Client {
	if !cfg.IsProviderEnabled(config.ProviderGoogleAI) {
		return nil
	}

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Printf("Google AI enabled but missing GEMINI_API_KEY")
		return nil
	}

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		log.Printf("Warning: Failed to create Gemini client: %v", err)
		return nil
	}

	log.Printf("Google AI client initialized")
	return client
}

func initializeAuthentication(cfg *config.Config, db *database.DB) (*auth.AuthMiddleware, *auth.AuthHandlers) {
	if cfg.Auth.ClientID == "" || cfg.Auth.ClientSecret == "" || cfg.Auth.IssuerURL == "" {
		log.Fatalf("Authentication is required but missing OIDC credentials. Please set OIDC_CLIENT_ID, OIDC_CLIENT_SECRET, and OIDC_ISSUER_URL")
	}

	// Create OIDC client
	oidcClient, err := auth.NewOIDCClient(
		cfg.Auth.ClientID,
		cfg.Auth.ClientSecret,
		cfg.Auth.RedirectURI,
		cfg.Auth.IssuerURL,
	)
	if err != nil {
		log.Fatalf("Failed to initialize OIDC client: %v", err)
	}

	// Create session store (in-memory for now)
	sessionStore := auth.NewInMemorySessionStore()

	// Create middleware and handlers
	authMiddleware := auth.NewAuthMiddleware(oidcClient, sessionStore)
	authHandlers := auth.NewAuthHandlers(oidcClient, sessionStore, db, cfg.Auth.BaseURL)

	log.Printf("Authentication initialized with OIDC client ID: %s", cfg.Auth.ClientID)
	return authMiddleware, authHandlers
}

func setupRoutes(srv *server.Server, authMiddleware *auth.AuthMiddleware, authHandlers *auth.AuthHandlers) {
	// Public routes (no authentication required)
	http.HandleFunc("/health", srv.HealthHandler)

	// Authentication routes (always enabled)
	http.HandleFunc("/auth/login", authHandlers.LoginHandler)
	http.HandleFunc("/callback/", authHandlers.CallbackHandler)
	http.HandleFunc("/auth/logout", authHandlers.LogoutHandler)
	http.HandleFunc("/auth/status", authHandlers.StatusHandler)
	http.HandleFunc("/auth/profile", authHandlers.ProfileHandler)

	// Web UI routes (optional auth for viewing, required for metrics)
	http.HandleFunc("/", authMiddleware.OptionalAuth(srv.UIHandler))
	http.HandleFunc("/ui", authMiddleware.OptionalAuth(srv.UIHandler))
	http.HandleFunc("/metrics", authMiddleware.RequireAuth(srv.MetricsHandler))

	// API routes (always require authentication)
	http.HandleFunc("/v1/chat/completions", authMiddleware.RequireAuth(srv.ChatCompletionHandler))
	http.HandleFunc("/v1/embeddings", authMiddleware.RequireAuth(srv.EmbeddingHandler))
	http.HandleFunc("/v1/models", authMiddleware.RequireAuth(srv.ModelsHandler))
}

func startBackgroundTasks(db *database.DB) {
	go func() {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		for range ticker.C {
			if err := db.ExportMetrics(); err != nil {
				log.Printf("Failed to export metrics: %v", err)
			}
		}
	}()
}

func logServerStatus(cfg *config.Config, db *database.DB, port string) {
	var enabledProviders []string

	if cfg.IsProviderEnabled(config.ProviderAzureOpenAI) {
		enabledProviders = append(enabledProviders, "Azure OpenAI")
	}
	if cfg.IsProviderEnabled(config.ProviderAWSBedrock) {
		enabledProviders = append(enabledProviders, "AWS Bedrock")
	}
	if cfg.IsProviderEnabled(config.ProviderGoogleAI) {
		enabledProviders = append(enabledProviders, "Google AI")
	}
	if cfg.IsProviderEnabled(config.ProviderAnthropic) {
		enabledProviders = append(enabledProviders, "Anthropic")
	}

	log.Printf("Starting LLM Freeway server on port %s", port)

	if len(enabledProviders) > 0 {
		log.Printf("Enabled providers: %v", enabledProviders)
	} else {
		log.Printf("Warning: No providers enabled")
	}

	log.Printf("Metrics: %t, Rate limiting: %t",
		cfg.Metrics.Enabled,
		len(db.GetAllRateStatus()) > 0)
}
