package auth

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"llm-freeway/internal/database"
)

// AuthHandlers provides HTTP handlers for authentication
type AuthHandlers struct {
	oidcClient   *OIDCClient
	sessionStore SessionStore
	database     *database.DB
	baseURL      string
}

// NewAuthHandlers creates new authentication handlers
func NewAuthHandlers(oidcClient *OIDCClient, sessionStore SessionStore, db *database.DB, baseURL string) *AuthHandlers {
	return &AuthHandlers{
		oidcClient:   oidcClient,
		sessionStore: sessionStore,
		database:     db,
		baseURL:      baseURL,
	}
}

// LoginHandler initiates the OIDC login flow
func (h *AuthHandlers) LoginHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Generate state parameter for CSRF protection
	state, err := GenerateState()
	if err != nil {
		http.Error(w, "Failed to generate state", http.StatusInternalServerError)
		return
	}

	// Store state in session (or you could use a signed cookie)
	isSecure := r.TLS != nil || r.Header.Get("X-Forwarded-Proto") == "https"
	http.SetCookie(w, &http.Cookie{
		Name:     "oauth_state",
		Value:    state,
		Path:     "/",
		MaxAge:   600, // 10 minutes
		HttpOnly: true,
		Secure:   isSecure, // Only secure for HTTPS
		SameSite: http.SameSiteLaxMode,
	})

	// Generate authorization URL
	authURL := h.oidcClient.GenerateAuthURL(state)

	log.Printf("Redirecting to OIDC provider: %s", authURL)
	http.Redirect(w, r, authURL, http.StatusFound)
}

// CallbackHandler handles the OIDC callback
func (h *AuthHandlers) CallbackHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Verify state parameter
	stateCookie, err := r.Cookie("oauth_state")
	if err != nil {
		http.Error(w, "Missing state cookie", http.StatusBadRequest)
		return
	}

	stateParam := r.URL.Query().Get("state")
	if stateParam == "" || stateParam != stateCookie.Value {
		http.Error(w, "Invalid state parameter", http.StatusBadRequest)
		return
	}

	// Clear state cookie
	http.SetCookie(w, &http.Cookie{
		Name:     "oauth_state",
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
	})

	// Get authorization code
	code := r.URL.Query().Get("code")
	if code == "" {
		errorMsg := r.URL.Query().Get("error")
		if errorMsg == "" {
			errorMsg = "No authorization code received"
		}
		http.Error(w, fmt.Sprintf("Authorization failed: %s", errorMsg), http.StatusBadRequest)
		return
	}

	// Exchange code for tokens
	tokenResp, err := h.oidcClient.ExchangeCode(r.Context(), code)
	if err != nil {
		log.Printf("Token exchange failed: %v", err)
		http.Error(w, "Token exchange failed", http.StatusInternalServerError)
		return
	}

	// Validate ID token and extract user info
	user, err := h.oidcClient.ValidateToken(tokenResp.IDToken)
	if err != nil {
		log.Printf("Token validation failed: %v", err)
		http.Error(w, "Token validation failed", http.StatusInternalServerError)
		return
	}

	// Store user in database
	dbUser := &database.User{
		ID:          user.ID,
		Email:       user.Email,
		DisplayName: user.DisplayName,
	}

	if err := h.database.CreateOrUpdateUser(dbUser); err != nil {
		log.Printf("Failed to store user: %v", err)
		http.Error(w, "Failed to store user", http.StatusInternalServerError)
		return
	}

	// Create session
	sessionID, err := h.generateSessionID()
	if err != nil {
		http.Error(w, "Failed to generate session", http.StatusInternalServerError)
		return
	}

	session := &Session{
		UserID:      user.ID,
		Email:       user.Email,
		DisplayName: user.DisplayName,
		AccessToken: tokenResp.AccessToken,
		IDToken:     tokenResp.IDToken,
		ExpiresAt:   time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second),
		CreatedAt:   time.Now(),
	}

	if err := h.sessionStore.Set(sessionID, session); err != nil {
		log.Printf("Failed to store session: %v", err)
		http.Error(w, "Failed to create session", http.StatusInternalServerError)
		return
	}

	// Set session cookie
	isSecure := r.TLS != nil || r.Header.Get("X-Forwarded-Proto") == "https"
	http.SetCookie(w, &http.Cookie{
		Name:     "session_id",
		Value:    sessionID,
		Path:     "/",
		MaxAge:   tokenResp.ExpiresIn,
		HttpOnly: true,
		Secure:   isSecure, // Only secure for HTTPS
		SameSite: http.SameSiteLaxMode,
	})

	log.Printf("User %s (%s) logged in successfully", user.DisplayName, user.Email)

	// Redirect to dashboard or return user info for API clients
	if r.Header.Get("Accept") == "application/json" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status": "success",
			"user":   user,
		})
	} else {
		// Redirect to main application
		http.Redirect(w, r, "/", http.StatusFound)
	}
}

// LogoutHandler handles user logout
func (h *AuthHandlers) LogoutHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" && r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get session cookie
	cookie, err := r.Cookie("session_id")
	if err == nil {
		// Delete session from store
		h.sessionStore.Delete(cookie.Value)
	}

	// Clear session cookie
	http.SetCookie(w, &http.Cookie{
		Name:     "session_id",
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
	})

	// Generate logout URL
	postLogoutRedirectURI := h.baseURL + "/"
	logoutURL := h.oidcClient.GenerateLogoutURL(postLogoutRedirectURI)

	log.Printf("User logged out, redirecting to: %s", logoutURL)

	// Redirect to OIDC provider logout
	if r.Header.Get("Accept") == "application/json" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":     "success",
			"logout_url": logoutURL,
		})
	} else {
		http.Redirect(w, r, logoutURL, http.StatusFound)
	}
}

// StatusHandler returns the current authentication status
func (h *AuthHandlers) StatusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check for session cookie directly
	cookie, err := r.Cookie("session_id")
	if err != nil {
		log.Printf("Status check: No session cookie found: %v", err)
		// Return unauthenticated
		response := map[string]any{
			"authenticated": false,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	log.Printf("Status check: Session cookie found: %s", cookie.Value[:min(20, len(cookie.Value))]+"...")

	// Check if session exists in store and is valid
	session, err := h.sessionStore.Get(cookie.Value)
	if err != nil {
		log.Printf("Status check: Session not found in store: %v", err)
		// Return unauthenticated
		response := map[string]any{
			"authenticated": false,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// Check if session is expired
	if time.Now().After(session.ExpiresAt) {
		log.Printf("Status check: Session expired for user: %s", session.Email)
		h.sessionStore.Delete(cookie.Value)
		response := map[string]any{
			"authenticated": false,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	log.Printf("Status check: Session found for user: %s", session.Email)

	// Create user from session
	user := &User{
		ID:          session.UserID,
		Email:       session.Email,
		DisplayName: session.DisplayName,
	}

	log.Printf("Status check: authenticated=true, user=%s", user.Email)

	response := map[string]any{
		"authenticated": true,
		"user":          user,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ProfileHandler returns the current user's profile with token
func (h *AuthHandlers) ProfileHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check for session cookie directly
	cookie, err := r.Cookie("session_id")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	// Check if session exists in store and is valid
	session, err := h.sessionStore.Get(cookie.Value)
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	// Check if session is expired
	if time.Now().After(session.ExpiresAt) {
		h.sessionStore.Delete(cookie.Value)
		http.Error(w, "Session expired", http.StatusUnauthorized)
		return
	}

	// Return user profile with token
	response := map[string]any{
		"user": map[string]any{
			"id":           session.UserID,
			"email":        session.Email,
			"display_name": session.DisplayName,
		},
		"id_token": session.IDToken,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// generateSessionID generates a secure random session ID
func (h *AuthHandlers) generateSessionID() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(b), nil
}
