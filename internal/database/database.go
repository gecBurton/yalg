package database

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"llm-freeway/internal/config"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// DB holds the database connection
type DB struct {
	*gorm.DB
	Config *config.Config
}

// DatabaseConfig holds database connection configuration
type DatabaseConfig struct {
	Host     string
	Port     int
	User     string
	Password string
	Database string
	SSLMode  string
}

// User represents an authenticated user
type User struct {
	ID          string     `gorm:"primaryKey;size:100" json:"id"`
	Email       string     `gorm:"uniqueIndex;size:255;not null" json:"email"`
	DisplayName string     `gorm:"size:255" json:"display_name"`
	Name        string     `gorm:"size:255" json:"name"`
	GivenName   string     `gorm:"size:255" json:"given_name"`
	FamilyName  string     `gorm:"size:255" json:"family_name"`
	Nickname    string     `gorm:"size:255" json:"nickname"`
	LastLoginAt *time.Time `gorm:"index" json:"last_login_at"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}

// Event represents a request or response event
type Event struct {
	ID         uint                `gorm:"primaryKey" json:"id"`
	Type       string              `gorm:"index;size:20;not null" json:"type"` // "request" or "response"
	RequestID  string              `gorm:"index;size:50;not null" json:"request_id"`
	Provider   config.ProviderType `gorm:"index;size:20;not null" json:"provider"`
	Model      string              `gorm:"size:100;not null" json:"model"`
	UserID     string              `gorm:"index;size:100" json:"user_id"` // Optional user ID for tracking
	Timestamp  time.Time           `gorm:"index;not null" json:"timestamp"`
	Tokens     int                 `gorm:"not null;default:0" json:"tokens"`
	Success    bool                `gorm:"not null;default:false" json:"success"`
	DurationMs int                 `gorm:"not null;default:0" json:"duration_ms"`
	ErrorMsg   string              `gorm:"size:500" json:"error_msg,omitempty"`
	CreatedAt  time.Time           `json:"created_at"`
	UpdatedAt  time.Time           `json:"updated_at"`
}

// TableName overrides for better naming
func (User) TableName() string {
	return "users"
}

func (Event) TableName() string {
	return "events"
}

// RequestMetrics tracks individual request metrics
type RequestMetrics struct {
	ID             string              `json:"id"`
	Timestamp      time.Time           `json:"timestamp"`
	Provider       config.ProviderType `json:"provider"`
	Model          string              `json:"model"`
	StartTime      time.Time           `json:"start_time"`
	EndTime        time.Time           `json:"end_time"`
	Duration       time.Duration       `json:"duration"`
	TokensUsed     int                 `json:"tokens_used"`
	PromptTokens   int                 `json:"prompt_tokens"`
	ResponseTokens int                 `json:"response_tokens"`
	Success        bool                `json:"success"`
	ErrorType      string              `json:"error_type,omitempty"`
	ErrorMessage   string              `json:"error_message,omitempty"`
	StatusCode     int                 `json:"status_code"`
	Streaming      bool                `json:"streaming"`
	UserAgent      string              `json:"user_agent,omitempty"`
	ClientIP       string              `json:"client_ip,omitempty"`
	UserID         string              `json:"user_id,omitempty"`
	UserEmail      string              `json:"user_email,omitempty"`
	UserName       string              `json:"user_name,omitempty"`
	Cost           float64             `json:"cost,omitempty"`
}

// UsageStats holds aggregated usage statistics
type UsageStats struct {
	TotalRequests      int           `json:"total_requests"`
	SuccessfulRequests int           `json:"successful_requests"`
	FailedRequests     int           `json:"failed_requests"`
	TotalTokens        int           `json:"total_tokens"`
	AverageLatency     time.Duration `json:"average_latency"`
}

// RateStatus represents the current rate limit status
type RateStatus struct {
	Provider   config.ProviderType `json:"provider"`
	Enabled    bool                `json:"enabled"`
	Used       int                 `json:"used"`        // Requests used in current window
	Limit      int                 `json:"limit"`       // Total limit per minute
	Remaining  int                 `json:"remaining"`   // Remaining requests
	ResetTime  time.Time           `json:"reset_time"`  // When the window resets
	RetryAfter int                 `json:"retry_after"` // Seconds until next request allowed
}

// NewDatabase creates a new database connection
func NewDatabase(cfg *config.Config) (*DB, error) {
	dbConfig := getDatabaseConfig()

	dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		dbConfig.Host, dbConfig.Port, dbConfig.User, dbConfig.Password, dbConfig.Database, dbConfig.SSLMode)

	// Configure GORM
	gormConfig := &gorm.Config{
		Logger: logger.Default.LogMode(getLogLevel(cfg)),
		NowFunc: func() time.Time {
			return time.Now().UTC()
		},
	}

	db, err := gorm.Open(postgres.Open(dsn), gormConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Get underlying sql.DB for connection pool configuration
	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to get underlying sql.DB: %w", err)
	}

	// Test connection
	if err := sqlDB.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	log.Printf("Connected to PostgreSQL database: %s:%d/%s", dbConfig.Host, dbConfig.Port, dbConfig.Database)

	return &DB{
		DB:     db,
		Config: cfg,
	}, nil
}

// Migrate runs database migrations
func (db *DB) Migrate() error {
	log.Println("Running database migrations...")

	// AutoMigrate all models
	err := db.AutoMigrate(
		&User{},
		&Event{},
	)
	if err != nil {
		return fmt.Errorf("failed to run migrations: %w", err)
	}

	log.Println("Database migrations completed successfully")
	return nil
}

// Private helper functions

func getDatabaseConfig() DatabaseConfig {
	return DatabaseConfig{
		Host:     getEnv("DB_HOST", "localhost"),
		Port:     getIntEnv("DB_PORT", 5432),
		User:     getEnv("DB_USER", "llm_freeway"),
		Password: getEnv("DB_PASSWORD", "password"),
		Database: getEnv("DB_NAME", "llm_freeway"),
		SSLMode:  getEnv("DB_SSLMODE", "disable"),
	}
}

func getLogLevel(cfg *config.Config) logger.LogLevel {
	switch cfg.Logging.Level {
	case "debug":
		return logger.Info // Show SQL queries in debug mode
	case "info":
		return logger.Warn // Only show warnings and errors
	default:
		return logger.Error // Only show errors
	}
}

// Environment variable helpers (duplicated from config package to avoid circular imports)
func getEnv(key, defaultValue string) string {
	if value := getEnvVar(key); value != "" {
		return value
	}
	return defaultValue
}

func getIntEnv(key string, defaultValue int) int {
	if value := getEnvVar(key); value != "" {
		if parsed := parseInt(value); parsed > 0 {
			return parsed
		}
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value := getEnvVar(key); value != "" {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

// Environment variable helper functions
func getEnvVar(key string) string {
	return os.Getenv(key)
}

func parseInt(value string) int {
	if parsed, err := strconv.Atoi(value); err == nil {
		return parsed
	}
	return 0
}

// ==============================================================================
// ANALYTICS METHODS (METRICS + RATE LIMITING)
// ==============================================================================

// RecordRequest records request and response events separately in the database
func (db *DB) RecordRequest(metric RequestMetrics, userID string) error {
	// Create request event
	requestEvent := Event{
		Type:       "request",
		RequestID:  metric.ID,
		Provider:   metric.Provider,
		Model:      metric.Model,
		UserID:     userID,
		Timestamp:  metric.StartTime,
		Tokens:     metric.PromptTokens,
		Success:    true, // Request initiation is successful
		DurationMs: 0,    // Request event has no duration yet
	}

	// Record the request event
	if err := db.Create(&requestEvent).Error; err != nil {
		log.Printf("Failed to record request event: %v", err)
		return err
	}

	// Create response event
	responseEvent := Event{
		Type:       "response",
		RequestID:  metric.ID,
		Provider:   metric.Provider,
		Model:      metric.Model,
		UserID:     userID,
		Timestamp:  metric.EndTime,
		Tokens:     metric.TokensUsed,
		Success:    metric.Success,
		DurationMs: int(metric.Duration.Milliseconds()),
		ErrorMsg:   metric.ErrorMessage,
	}

	// Record the response event
	if err := db.Create(&responseEvent).Error; err != nil {
		log.Printf("Failed to record response event: %v", err)
		return err
	}

	return nil
}

// GetUsageStats returns current usage statistics
func (db *DB) GetUsageStats() UsageStats {
	var stats UsageStats

	// Get total requests in last 24 hours
	since := time.Now().Add(-24 * time.Hour)

	var result struct {
		TotalRequests      int64
		SuccessfulRequests int64
		FailedRequests     int64
		TotalTokens        int64
		AvgDuration        float64
	}

	db.Model(&Event{}).
		Where("timestamp >= ? AND type = ?", since, "response").
		Select(`
			COUNT(*) as total_requests,
			SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_requests,
			SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_requests,
			COALESCE(SUM(tokens), 0) as total_tokens,
			COALESCE(AVG(duration_ms), 0) as avg_duration
		`).
		Scan(&result)

	stats.TotalRequests = int(result.TotalRequests)
	stats.SuccessfulRequests = int(result.SuccessfulRequests)
	stats.FailedRequests = int(result.FailedRequests)
	stats.TotalTokens = int(result.TotalTokens)
	stats.AverageLatency = time.Duration(result.AvgDuration) * time.Millisecond

	return stats
}

// GetRequestHistory returns recent request metrics
func (db *DB) GetRequestHistory(limit int) []RequestMetrics {
	var results []struct {
		Event
		UserEmail       string `gorm:"column:user_email"`
		UserDisplayName string `gorm:"column:user_display_name"`
	}

	db.Table("events").
		Select("events.*, users.email as user_email, users.display_name as user_display_name").
		Joins("LEFT JOIN users ON events.user_id = users.id").
		Where("events.type = ?", "response").
		Order("events.timestamp DESC").
		Limit(limit).
		Find(&results)

	history := make([]RequestMetrics, len(results))
	for i, result := range results {
		history[i] = RequestMetrics{
			ID:           result.Event.RequestID,
			Timestamp:    result.Event.Timestamp,
			Provider:     result.Event.Provider,
			Model:        result.Event.Model,
			StartTime:    result.Event.Timestamp.Add(-time.Duration(result.Event.DurationMs) * time.Millisecond),
			EndTime:      result.Event.Timestamp,
			Duration:     time.Duration(result.Event.DurationMs) * time.Millisecond,
			TokensUsed:   result.Event.Tokens,
			Success:      result.Event.Success,
			ErrorMessage: result.Event.ErrorMsg,
			UserID:       result.Event.UserID,
			UserEmail:    result.UserEmail,
			UserName:     result.UserDisplayName,
		}
	}

	return history
}

// ExportMetrics exports metrics to a file or external system
func (db *DB) ExportMetrics() error {
	if !db.Config.Metrics.Enabled {
		return nil
	}

	// Export could write to file, send to external metrics system, etc.
	log.Printf("Metrics export initiated (Event-based)")

	// Example: Cleanup old detailed request records while keeping aggregated data
	if db.Config.Metrics.RetentionDays > 0 {
		cutoff := time.Now().AddDate(0, 0, -db.Config.Metrics.RetentionDays)
		result := db.Where("timestamp < ?", cutoff).Delete(&Event{})
		if result.Error == nil && result.RowsAffected > 0 {
			log.Printf("Cleaned up %d old event records older than %d days",
				result.RowsAffected, db.Config.Metrics.RetentionDays)
		}
	}

	return nil
}

// CheckRate checks if a request can proceed for the given provider
func (db *DB) CheckRate(provider config.ProviderType) (bool, int) {
	// Get rate limit configuration for this provider
	providerConfig, exists := db.Config.Providers[provider]
	if !exists || providerConfig.RateLimit <= 0 {
		return true, 0 // No rate limiting configured
	}

	limit := providerConfig.RateLimit

	// Count requests in the last minute from Event table
	since := time.Now().Add(-time.Minute)
	var count int64

	err := db.Model(&Event{}).
		Where("provider = ? AND type = ? AND timestamp >= ?",
			provider, "request", since).
		Count(&count).Error

	if err != nil {
		log.Printf("Rate limit check failed for provider %s: %v", provider, err)
		return true, 0 // Fail open on database errors
	}

	used := int(count)
	allowed := used < limit
	retryAfter := 0

	if !allowed {
		// Calculate when the oldest request in the window will expire
		var oldestTimestamp time.Time
		err := db.Model(&Event{}).
			Where("provider = ? AND type = ? AND timestamp >= ?",
				provider, "request", since).
			Order("timestamp ASC").
			Limit(1).
			Pluck("timestamp", &oldestTimestamp).Error

		if err == nil && !oldestTimestamp.IsZero() {
			// Retry after the oldest request expires (plus 1 second buffer)
			retryAfter = int(time.Until(oldestTimestamp.Add(time.Minute + time.Second)).Seconds())
			if retryAfter < 1 {
				retryAfter = 1
			}
		} else {
			retryAfter = 60 // Default: try again in 1 minute
		}
	}

	return allowed, retryAfter
}

// GetRateStatus returns current rate limit status for a provider
func (db *DB) GetRateStatus(provider config.ProviderType) *RateStatus {
	providerConfig, exists := db.Config.Providers[provider]
	if !exists || providerConfig.RateLimit <= 0 {
		return &RateStatus{
			Provider:  provider,
			Enabled:   false,
			Used:      0,
			Limit:     0,
			Remaining: -1,
			ResetTime: time.Time{},
		}
	}

	limit := providerConfig.RateLimit

	// Count requests in the last minute
	since := time.Now().Add(-time.Minute)
	var count int64

	err := db.Model(&Event{}).
		Where("provider = ? AND type = ? AND timestamp >= ?",
			provider, "request", since).
		Count(&count).Error

	if err != nil {
		log.Printf("Failed to get rate status for provider %s: %v", provider, err)
		return &RateStatus{
			Provider:  provider,
			Enabled:   true,
			Used:      -1,
			Limit:     limit,
			Remaining: -1,
			ResetTime: time.Now().Add(time.Minute),
		}
	}

	used := int(count)
	remaining := max(limit-used, 0)

	// Reset time is when the current minute window ends
	resetTime := time.Now().Truncate(time.Minute).Add(time.Minute)

	retryAfter := 0
	if remaining == 0 {
		retryAfter = int(time.Until(resetTime).Seconds())
		if retryAfter < 1 {
			retryAfter = 1
		}
	}

	return &RateStatus{
		Provider:   provider,
		Enabled:    true,
		Used:       used,
		Limit:      limit,
		Remaining:  remaining,
		ResetTime:  resetTime,
		RetryAfter: retryAfter,
	}
}

// GetAllRateStatus returns rate limit status for all configured providers
func (db *DB) GetAllRateStatus() map[config.ProviderType]*RateStatus {
	status := make(map[config.ProviderType]*RateStatus)

	for provider, providerConfig := range db.Config.Providers {
		if providerConfig.Enabled {
			status[provider] = db.GetRateStatus(provider)
		}
	}

	return status
}

// ==============================================================================
// USER MANAGEMENT METHODS
// ==============================================================================

// CreateOrUpdateUser creates a new user or updates an existing one
func (db *DB) CreateOrUpdateUser(user *User) error {
	now := time.Now()

	// Try to find existing user
	var existingUser User
	result := db.Where("id = ?", user.ID).First(&existingUser)

	if result.Error == nil {
		// User exists, update it
		user.LastLoginAt = &now
		user.UpdatedAt = now
		user.CreatedAt = existingUser.CreatedAt // Preserve original creation time

		return db.Save(user).Error
	} else {
		// User doesn't exist, create new one
		user.LastLoginAt = &now
		user.CreatedAt = now
		user.UpdatedAt = now

		return db.Create(user).Error
	}
}
