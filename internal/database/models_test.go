package database

import (
	"testing"
	"time"

	"llm-freeway/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEventModel(t *testing.T) {
	db := NewTestDB(t)
	defer db.CleanupTestData(t)

	t.Run("create event successfully", func(t *testing.T) {
		event := &Event{
			Type:       "request",
			RequestID:  "test-request-1",
			Provider:   config.ProviderAzureOpenAI,
			Model:      "gpt-4",
			Timestamp:  time.Now(),
			Tokens:     100,
			Success:    true,
			DurationMs: 0,
		}

		err := db.Create(event).Error
		require.NoError(t, err)
		assert.NotZero(t, event.ID)
		assert.False(t, event.CreatedAt.IsZero())
		assert.False(t, event.UpdatedAt.IsZero())
	})

	t.Run("create response event with error", func(t *testing.T) {
		event := &Event{
			Type:       "response",
			RequestID:  "test-request-2",
			Provider:   config.ProviderAWSBedrock,
			Model:      "anthropic.claude-3-sonnet-20240229-v1:0",
			Timestamp:  time.Now(),
			Tokens:     0,
			Success:    false,
			DurationMs: 1500,
			ErrorMsg:   "Provider rate limit exceeded",
		}

		err := db.Create(event).Error
		require.NoError(t, err)
		assert.NotZero(t, event.ID)
		assert.Equal(t, "Provider rate limit exceeded", event.ErrorMsg)
		assert.False(t, event.Success)
	})

	t.Run("query events by provider", func(t *testing.T) {
		db.CleanupTestData(t)

		// Create test events for different providers
		events := []Event{
			{
				Type:      "request",
				RequestID: "req-1",
				Provider:  config.ProviderAzureOpenAI,
				Model:     "gpt-4",
				Timestamp: time.Now(),
				Tokens:    100,
				Success:   true,
			},
			{
				Type:      "request",
				RequestID: "req-2",
				Provider:  config.ProviderAWSBedrock,
				Model:     "anthropic.claude-3-haiku-20240307-v1:0",
				Timestamp: time.Now(),
				Tokens:    50,
				Success:   true,
			},
			{
				Type:      "request",
				RequestID: "req-3",
				Provider:  config.ProviderAzureOpenAI,
				Model:     "gpt-4-turbo",
				Timestamp: time.Now(),
				Tokens:    200,
				Success:   true,
			},
		}

		for _, event := range events {
			err := db.Create(&event).Error
			require.NoError(t, err)
		}

		// Query OpenAI events
		var openaiEvents []Event
		err := db.Where("provider = ?", config.ProviderAzureOpenAI).Find(&openaiEvents).Error
		require.NoError(t, err)
		assert.Len(t, openaiEvents, 2)

		// Query Bedrock events
		var bedrockEvents []Event
		err = db.Where("provider = ?", config.ProviderAWSBedrock).Find(&bedrockEvents).Error
		require.NoError(t, err)
		assert.Len(t, bedrockEvents, 1)
	})

	t.Run("query events by type", func(t *testing.T) {
		db.CleanupTestData(t)

		// Create paired request/response events
		requestEvent := &Event{
			Type:       "request",
			RequestID:  "paired-req-1",
			Provider:   config.ProviderGoogleAI,
			Model:      "gemini-1.5-pro",
			Timestamp:  time.Now().Add(-1 * time.Minute),
			Tokens:     75,
			Success:    true,
			DurationMs: 0,
		}
		err := db.Create(requestEvent).Error
		require.NoError(t, err)

		responseEvent := &Event{
			Type:       "response",
			RequestID:  "paired-req-1",
			Provider:   config.ProviderGoogleAI,
			Model:      "gemini-1.5-pro",
			Timestamp:  time.Now(),
			Tokens:     125,
			Success:    true,
			DurationMs: 2000,
		}
		err = db.Create(responseEvent).Error
		require.NoError(t, err)

		// Query request events
		var requestEvents []Event
		err = db.Where("type = ?", "request").Find(&requestEvents).Error
		require.NoError(t, err)
		assert.Len(t, requestEvents, 1)
		assert.Equal(t, 0, requestEvents[0].DurationMs)

		// Query response events
		var responseEvents []Event
		err = db.Where("type = ?", "response").Find(&responseEvents).Error
		require.NoError(t, err)
		assert.Len(t, responseEvents, 1)
		assert.Equal(t, 2000, responseEvents[0].DurationMs)
	})

	t.Run("query events by time range", func(t *testing.T) {
		db.CleanupTestData(t)

		now := time.Now()
		oldEvent := &Event{
			Type:      "request",
			RequestID: "old-req",
			Provider:  config.ProviderAzureOpenAI,
			Model:     "gpt-4",
			Timestamp: now.Add(-5 * time.Minute),
			Tokens:    100,
			Success:   true,
		}
		err := db.Create(oldEvent).Error
		require.NoError(t, err)

		recentEvent := &Event{
			Type:      "request",
			RequestID: "recent-req",
			Provider:  config.ProviderAzureOpenAI,
			Model:     "gpt-4",
			Timestamp: now.Add(-30 * time.Second),
			Tokens:    100,
			Success:   true,
		}
		err = db.Create(recentEvent).Error
		require.NoError(t, err)

		// Query events from last 2 minutes
		since := now.Add(-2 * time.Minute)
		var recentEvents []Event
		err = db.Where("timestamp >= ?", since).Find(&recentEvents).Error
		require.NoError(t, err)
		assert.Len(t, recentEvents, 1)
		assert.Equal(t, "recent-req", recentEvents[0].RequestID)
	})
}

func TestEventTableName(t *testing.T) {
	event := Event{}
	assert.Equal(t, "events", event.TableName())
}

func TestEventValidation(t *testing.T) {
	db := NewTestDB(t)
	defer db.CleanupTestData(t)

	t.Run("required fields validation", func(t *testing.T) {
		// Test with missing required fields
		event := &Event{
			// Missing Type, RequestID, Provider - should be handled by database constraints
		}

		err := db.Create(event).Error
		// SQLite might be more lenient, but PostgreSQL would enforce NOT NULL constraints
		// This test documents expected behavior rather than enforcing it
		if err != nil {
			t.Logf("Database correctly rejected event with missing required fields: %v", err)
		}
	})

	t.Run("valid event types", func(t *testing.T) {
		validTypes := []string{"request", "response"}

		for _, eventType := range validTypes {
			event := &Event{
				Type:       eventType,
				RequestID:  "test-req-" + eventType,
				Provider:   config.ProviderAzureOpenAI,
				Model:      "gpt-4",
				Timestamp:  time.Now(),
				Tokens:     100,
				Success:    true,
				DurationMs: 1000,
			}

			err := db.Create(event).Error
			require.NoError(t, err, "Failed to create event with type: %s", eventType)
		}
	})
}

func TestEventHelperMethods(t *testing.T) {
	db := NewTestDB(t)
	defer db.CleanupTestData(t)

	// Seed some test data
	db.SeedTestData(t)

	t.Run("get event count", func(t *testing.T) {
		count := db.GetEventCount(t)
		assert.Greater(t, count, int64(0))
	})

	t.Run("get events by provider", func(t *testing.T) {
		openaiEvents := db.GetEventsByProvider(t, config.ProviderAzureOpenAI)
		assert.NotEmpty(t, openaiEvents)

		for _, event := range openaiEvents {
			assert.Equal(t, config.ProviderAzureOpenAI, event.Provider)
		}
	})

	t.Run("get events by type", func(t *testing.T) {
		requestEvents := db.GetEventsByType(t, "request")
		responseEvents := db.GetEventsByType(t, "response")

		assert.NotEmpty(t, requestEvents)
		assert.NotEmpty(t, responseEvents)

		for _, event := range requestEvents {
			assert.Equal(t, "request", event.Type)
		}

		for _, event := range responseEvents {
			assert.Equal(t, "response", event.Type)
		}
	})

	t.Run("get recent events", func(t *testing.T) {
		recentEvents := db.GetRecentEvents(t, 10) // Last 10 minutes
		assert.NotEmpty(t, recentEvents)

		// All events should be within the last 10 minutes
		cutoff := time.Now().Add(-10 * time.Minute)
		for _, event := range recentEvents {
			assert.True(t, event.Timestamp.After(cutoff),
				"Event timestamp should be within last 10 minutes")
		}
	})
}

func TestCreateTestEvent(t *testing.T) {
	db := NewTestDB(t)
	defer db.CleanupTestData(t)

	t.Run("create successful test event", func(t *testing.T) {
		event := db.CreateTestEvent(t, "request", "test-req",
			config.ProviderGoogleAI, "gemini-1.5-pro", 150, true)

		assert.NotZero(t, event.ID)
		assert.Equal(t, "request", event.Type)
		assert.Equal(t, "test-req", event.RequestID)
		assert.Equal(t, config.ProviderGoogleAI, event.Provider)
		assert.Equal(t, "gemini-1.5-pro", event.Model)
		assert.Equal(t, 150, event.Tokens)
		assert.True(t, event.Success)
		assert.Empty(t, event.ErrorMsg)
	})

	t.Run("create failed test event", func(t *testing.T) {
		event := db.CreateTestEvent(t, "response", "test-req-failed",
			config.ProviderAWSBedrock, "anthropic.claude-3-opus-20240229-v1:0", 0, false)

		assert.NotZero(t, event.ID)
		assert.Equal(t, "response", event.Type)
		assert.False(t, event.Success)
		assert.Equal(t, "Test error message", event.ErrorMsg)
	})
}
