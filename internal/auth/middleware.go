package auth

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"
)

// ContextKey represents keys for context values
type ContextKey string

const (
	UserContextKey ContextKey = "user"
)

// AuthMiddleware provides authentication middleware
type AuthMiddleware struct {
	oidcClient   *OIDCClient
	sessionStore SessionStore
	publicPaths  map[string]bool
}

// SessionStore interface for session management
type SessionStore interface {
	Get(sessionID string) (*Session, error)
	Set(sessionID string, session *Session) error
	Delete(sessionID string) error
}

// Session represents a user session
type Session struct {
	UserID      string    `json:"user_id"`
	Email       string    `json:"email"`
	DisplayName string    `json:"display_name"`
	AccessToken string    `json:"access_token"`
	IDToken     string    `json:"id_token"`
	ExpiresAt   time.Time `json:"expires_at"`
	CreatedAt   time.Time `json:"created_at"`
}

// NewAuthMiddleware creates a new authentication middleware
func NewAuthMiddleware(oidcClient *OIDCClient, sessionStore SessionStore) *AuthMiddleware {
	publicPaths := map[string]bool{
		"/health":         true,
		"/auth/login":     true,
		"/auth/callback":  true,
		"/auth/logout":    true,
	}

	return &AuthMiddleware{
		oidcClient:   oidcClient,
		sessionStore: sessionStore,
		publicPaths:  publicPaths,
	}
}

// RequireAuth is a middleware that requires authentication
func (m *AuthMiddleware) RequireAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Check if this is a public path
		if m.publicPaths[r.URL.Path] {
			next(w, r)
			return
		}

		// Check for Bearer token in Authorization header (for API access)
		authHeader := r.Header.Get("Authorization")
		if strings.HasPrefix(authHeader, "Bearer ") {
			token := strings.TrimPrefix(authHeader, "Bearer ")
			user, err := m.oidcClient.ValidateToken(token)
			if err != nil {
				m.sendUnauthorized(w, "Invalid token: "+err.Error())
				return
			}

			// Add user to context
			ctx := context.WithValue(r.Context(), UserContextKey, user)
			next(w, r.WithContext(ctx))
			return
		}

		// Check for session cookie (for web UI access)
		cookie, err := r.Cookie("session_id")
		if err != nil {
			m.sendUnauthorized(w, "No session found")
			return
		}

		session, err := m.sessionStore.Get(cookie.Value)
		if err != nil {
			m.sendUnauthorized(w, "Invalid session")
			return
		}

		// Check if session is expired
		if time.Now().After(session.ExpiresAt) {
			m.sessionStore.Delete(cookie.Value)
			m.sendUnauthorized(w, "Session expired")
			return
		}

		// Create user from session
		user := &User{
			ID:          session.UserID,
			Email:       session.Email,
			DisplayName: session.DisplayName,
		}

		// Add user to context
		ctx := context.WithValue(r.Context(), UserContextKey, user)
		next(w, r.WithContext(ctx))
	}
}

// OptionalAuth is a middleware that allows both authenticated and unauthenticated access
func (m *AuthMiddleware) OptionalAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Try to get user from token or session, but don't fail if not present
		var user *User

		// Check for Bearer token
		authHeader := r.Header.Get("Authorization")
		if strings.HasPrefix(authHeader, "Bearer ") {
			token := strings.TrimPrefix(authHeader, "Bearer ")
			if u, err := m.oidcClient.ValidateToken(token); err == nil {
				user = u
			}
		}

		// Check for session cookie if no token found
		if user == nil {
			if cookie, err := r.Cookie("session_id"); err == nil {
				if session, err := m.sessionStore.Get(cookie.Value); err == nil {
					if time.Now().Before(session.ExpiresAt) {
						user = &User{
							ID:          session.UserID,
							Email:       session.Email,
							DisplayName: session.DisplayName,
						}
					}
				}
			}
		}

		// Add user to context if found
		ctx := r.Context()
		if user != nil {
			ctx = context.WithValue(ctx, UserContextKey, user)
		}

		next(w, r.WithContext(ctx))
	}
}

// GetUserFromContext extracts the user from the request context
func GetUserFromContext(ctx context.Context) (*User, bool) {
	user, ok := ctx.Value(UserContextKey).(*User)
	return user, ok
}

// sendUnauthorized sends a 401 Unauthorized response
func (m *AuthMiddleware) sendUnauthorized(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	
	response := map[string]string{
		"error":   "unauthorized",
		"message": message,
	}
	
	json.NewEncoder(w).Encode(response)
	log.Printf("Authentication failed: %s", message)
}

// In-memory session store implementation
type InMemorySessionStore struct {
	sessions map[string]*Session
}

// NewInMemorySessionStore creates a new in-memory session store
func NewInMemorySessionStore() *InMemorySessionStore {
	return &InMemorySessionStore{
		sessions: make(map[string]*Session),
	}
}

// Get retrieves a session by ID
func (s *InMemorySessionStore) Get(sessionID string) (*Session, error) {
	session, exists := s.sessions[sessionID]
	if !exists {
		return nil, ErrSessionNotFound
	}
	return session, nil
}

// Set stores a session
func (s *InMemorySessionStore) Set(sessionID string, session *Session) error {
	s.sessions[sessionID] = session
	return nil
}

// Delete removes a session
func (s *InMemorySessionStore) Delete(sessionID string) error {
	delete(s.sessions, sessionID)
	return nil
}

// Custom errors
var (
	ErrSessionNotFound = &AuthError{Code: "session_not_found", Message: "Session not found"}
)

// AuthError represents an authentication error
type AuthError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *AuthError) Error() string {
	return e.Message
}