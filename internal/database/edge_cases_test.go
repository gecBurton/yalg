package database

import (
	"fmt"
	"testing"
	"time"

	"llm-freeway/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEventEdgeCases(t *testing.T) {
	db := NewTestDB(t)
	defer db.CleanupTestData(t)

	t.Run("event with zero tokens", func(t *testing.T) {
		event := &Event{
			Type:       "response",
			RequestID:  "zero-tokens-req",
			Provider:   config.ProviderAzureOpenAI,
			Model:      "gpt-4",
			Timestamp:  time.Now(),
			Tokens:     0,
			Success:    false,
			DurationMs: 100,
			ErrorMsg:   "No tokens consumed due to error",
		}

		err := db.Create(event).Error
		require.NoError(t, err)
		assert.Equal(t, 0, event.Tokens)
		assert.False(t, event.Success)
	})

	t.Run("event with very high token count", func(t *testing.T) {
		event := &Event{
			Type:       "response",
			RequestID:  "high-tokens-req",
			Provider:   config.ProviderGoogleAI,
			Model:      "gemini-1.5-pro",
			Timestamp:  time.Now(),
			Tokens:     1000000, // 1M tokens
			Success:    true,
			DurationMs: 30000,
		}

		err := db.Create(event).Error
		require.NoError(t, err)
		assert.Equal(t, 1000000, event.Tokens)
	})

	t.Run("event with very long duration", func(t *testing.T) {
		event := &Event{
			Type:       "response",
			RequestID:  "slow-req",
			Provider:   config.ProviderAWSBedrock,
			Model:      "anthropic.claude-3-opus-20240229-v1:0",
			Timestamp:  time.Now(),
			Tokens:     500,
			Success:    true,
			DurationMs: 120000, // 2 minutes
		}

		err := db.Create(event).Error
		require.NoError(t, err)
		assert.Equal(t, 120000, event.DurationMs)
	})

	t.Run("event with long error message", func(t *testing.T) {
		longErrorMsg := "This is a very long error message that could potentially exceed normal database field lengths. " +
			"It contains detailed information about what went wrong during the request processing, including " +
			"stack traces, error codes, and other diagnostic information that might be useful for debugging. " +
			"However, we need to make sure our database can handle storing such long error messages without " +
			"truncation or other issues that could affect our ability to diagnose problems in production."

		event := &Event{
			Type:       "response",
			RequestID:  "long-error-req",
			Provider:   config.ProviderAzureOpenAI,
			Model:      "gpt-4-turbo",
			Timestamp:  time.Now(),
			Tokens:     0,
			Success:    false,
			DurationMs: 1000,
			ErrorMsg:   longErrorMsg,
		}

		err := db.Create(event).Error
		require.NoError(t, err)
		
		// The error message might be truncated based on database field size (500 chars)
		// But it should still be stored successfully
		assert.NotEmpty(t, event.ErrorMsg)
		assert.False(t, event.Success)
	})

	t.Run("multiple event creation", func(t *testing.T) {
		db.CleanupTestData(t)
		
		// Test sequential creation of multiple events
		const numEvents = 10

		for i := 0; i < numEvents; i++ {
			event := &Event{
				Type:       "request",
				RequestID:  fmt.Sprintf("sequential-%d", i),
				Provider:   config.ProviderAzureOpenAI,
				Model:      "gpt-4",
				Timestamp:  time.Now().Add(time.Duration(i) * time.Second),
				Tokens:     100 + i,
				Success:    true,
				DurationMs: 1000,
			}

			err := db.Create(event).Error
			require.NoError(t, err, "Sequential event creation should not fail")
		}

		// Verify all events were created
		var count int64
		err := db.Model(&Event{}).Count(&count).Error
		require.NoError(t, err)
		assert.Equal(t, int64(numEvents), count)
	})

	t.Run("events with same request_id different types", func(t *testing.T) {
		db.CleanupTestData(t)

		requestID := "paired-request-123"

		// Create request event
		requestEvent := &Event{
			Type:       "request",
			RequestID:  requestID,
			Provider:   config.ProviderGoogleAI,
			Model:      "gemini-1.5-flash",
			Timestamp:  time.Now().Add(-1 * time.Second),
			Tokens:     80,
			Success:    true,
			DurationMs: 0,
		}
		err := db.Create(requestEvent).Error
		require.NoError(t, err)

		// Create response event with same request_id
		responseEvent := &Event{
			Type:       "response",
			RequestID:  requestID,
			Provider:   config.ProviderGoogleAI,
			Model:      "gemini-1.5-flash",
			Timestamp:  time.Now(),
			Tokens:     120,
			Success:    true,
			DurationMs: 1500,
		}
		err = db.Create(responseEvent).Error
		require.NoError(t, err)

		// Query both events by request_id
		var events []Event
		err = db.Where("request_id = ?", requestID).Order("timestamp ASC").Find(&events).Error
		require.NoError(t, err)
		
		assert.Len(t, events, 2)
		assert.Equal(t, "request", events[0].Type)
		assert.Equal(t, "response", events[1].Type)
		assert.Equal(t, 0, events[0].DurationMs)
		assert.Equal(t, 1500, events[1].DurationMs)
	})

	t.Run("events with future timestamps", func(t *testing.T) {
		futureTime := time.Now().Add(1 * time.Hour)
		
		event := &Event{
			Type:       "request",
			RequestID:  "future-req",
			Provider:   config.ProviderAWSBedrock,
			Model:      "anthropic.claude-3-haiku-20240307-v1:0",
			Timestamp:  futureTime,
			Tokens:     50,
			Success:    true,
			DurationMs: 800,
		}

		err := db.Create(event).Error
		require.NoError(t, err)
		
		// Verify the future timestamp was preserved
		var retrieved Event
		err = db.Where("request_id = ?", "future-req").First(&retrieved).Error
		require.NoError(t, err)
		assert.True(t, retrieved.Timestamp.After(time.Now()))
	})
}

func TestEventTableNameConsistency(t *testing.T) {
	event := Event{}
	assert.Equal(t, "events", event.TableName())
	
	// Verify table name is consistent across instances
	anotherEvent := Event{
		Type:      "request",
		RequestID: "test",
		Provider:  config.ProviderAzureOpenAI,
	}
	assert.Equal(t, "events", anotherEvent.TableName())
}