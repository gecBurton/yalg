package database

import (
	"testing"
	"time"

	"llm-freeway/internal/config"

	"github.com/stretchr/testify/require"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// TestDatabase wraps gorm.DB with test-specific helpers
type TestDatabase struct {
	*gorm.DB
}

// NewTestDB creates a new in-memory SQLite database for testing
func NewTestDB(t *testing.T) *TestDatabase {
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
		NowFunc: func() time.Time {
			return time.Now().UTC()
		},
	})
	require.NoError(t, err)

	// Auto migrate Event model only
	err = db.AutoMigrate(
		&Event{},
	)
	require.NoError(t, err)

	return &TestDatabase{DB: db}
}

// SeedTestData creates sample data for testing
func (tdb *TestDatabase) SeedTestData(t *testing.T) {
	// Create test events
	events := []Event{
		{
			Type:       "request",
			RequestID:  "req-1",
			Provider:   config.ProviderAzureOpenAI,
			Model:      "gpt-4",
			Timestamp:  time.Now().Add(-5 * time.Minute),
			Tokens:     100,
			Success:    true,
			DurationMs: 0,
		},
		{
			Type:       "response",
			RequestID:  "req-1",
			Provider:   config.ProviderAzureOpenAI,
			Model:      "gpt-4",
			Timestamp:  time.Now().Add(-5 * time.Minute),
			Tokens:     150,
			Success:    true,
			DurationMs: 1200,
		},
		{
			Type:       "request",
			RequestID:  "req-2",
			Provider:   config.ProviderAWSBedrock,
			Model:      "anthropic.claude-3-sonnet-20240229-v1:0",
			Timestamp:  time.Now().Add(-3 * time.Minute),
			Tokens:     200,
			Success:    true,
			DurationMs: 0,
		},
		{
			Type:       "response",
			RequestID:  "req-2",
			Provider:   config.ProviderAWSBedrock,
			Model:      "anthropic.claude-3-sonnet-20240229-v1:0",
			Timestamp:  time.Now().Add(-3 * time.Minute),
			Tokens:     300,
			Success:    true,
			DurationMs: 2000,
		},
		{
			Type:       "request",
			RequestID:  "req-3",
			Provider:   config.ProviderGoogleAI,
			Model:      "gemini-1.5-pro",
			Timestamp:  time.Now().Add(-1 * time.Minute),
			Tokens:     50,
			Success:    true,
			DurationMs: 0,
		},
		{
			Type:       "response",
			RequestID:  "req-3",
			Provider:   config.ProviderGoogleAI,
			Model:      "gemini-1.5-pro",
			Timestamp:  time.Now().Add(-1 * time.Minute),
			Tokens:     75,
			Success:    false,
			DurationMs: 500,
			ErrorMsg:   "Rate limit exceeded",
		},
	}

	for _, event := range events {
		err := tdb.Create(&event).Error
		require.NoError(t, err)
	}
}

// CreateTestEvent creates a test event with given parameters
func (tdb *TestDatabase) CreateTestEvent(t *testing.T, eventType string, requestID string, provider config.ProviderType, model string, tokens int, success bool) *Event {
	event := &Event{
		Type:       eventType,
		RequestID:  requestID,
		Provider:   provider,
		Model:      model,
		Timestamp:  time.Now(),
		Tokens:     tokens,
		Success:    success,
		DurationMs: 1000,
	}

	if !success {
		event.ErrorMsg = "Test error message"
	}

	err := tdb.Create(event).Error
	require.NoError(t, err)

	return event
}

// GetEventCount returns the total number of events in the database
func (tdb *TestDatabase) GetEventCount(t *testing.T) int64 {
	var count int64
	err := tdb.Model(&Event{}).Count(&count).Error
	require.NoError(t, err)
	return count
}

// GetEventsByProvider returns all events for a specific provider
func (tdb *TestDatabase) GetEventsByProvider(t *testing.T, provider config.ProviderType) []Event {
	var events []Event
	err := tdb.Where("provider = ?", provider).Find(&events).Error
	require.NoError(t, err)
	return events
}

// GetEventsByType returns all events of a specific type (request/response)
func (tdb *TestDatabase) GetEventsByType(t *testing.T, eventType string) []Event {
	var events []Event
	err := tdb.Where("type = ?", eventType).Find(&events).Error
	require.NoError(t, err)
	return events
}

// CleanupTestData removes all test data
func (tdb *TestDatabase) CleanupTestData(t *testing.T) {
	err := tdb.Exec("DELETE FROM events").Error
	require.NoError(t, err)
}

// GetRecentEvents returns events from the last N minutes
func (tdb *TestDatabase) GetRecentEvents(t *testing.T, minutes int) []Event {
	since := time.Now().Add(-time.Duration(minutes) * time.Minute)
	var events []Event
	err := tdb.Where("timestamp >= ?", since).Find(&events).Error
	require.NoError(t, err)
	return events
}
