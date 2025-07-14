package auth

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/lestrrat-go/jwx/v2/jwk"
)

// OIDCConfig holds the OIDC configuration
type OIDCConfig struct {
	Issuer                string   `json:"issuer"`
	AuthorizationEndpoint string   `json:"authorization_endpoint"`
	TokenEndpoint         string   `json:"token_endpoint"`
	UserInfoEndpoint      string   `json:"userinfo_endpoint"`
	EndSessionEndpoint    string   `json:"end_session_endpoint"`
	JWKSUri               string   `json:"jwks_uri"`
	ScopesSupported       []string `json:"scopes_supported"`
	ClaimsSupported       []string `json:"claims_supported"`
}

// OIDCClient handles OIDC authentication
type OIDCClient struct {
	config       OIDCConfig
	clientID     string
	clientSecret string
	redirectURI  string
	jwkSet       jwk.Set
}

// User represents an authenticated user
type User struct {
	ID          string `json:"sub"`
	Email       string `json:"email"`
	DisplayName string `json:"display_name"`
	Name        string `json:"name"`
}

// Claims represents JWT token claims
type Claims struct {
	jwt.RegisteredClaims
	Email       string `json:"email"`
	DisplayName string `json:"display_name"`
	Name        string `json:"name"`
}

// NewOIDCClient creates a new OIDC client
func NewOIDCClient(clientID, clientSecret, redirectURI string) (*OIDCClient, error) {
	// Use the well-known configuration provided
	config := OIDCConfig{
		Issuer:                "https://sso.service.security.gov.uk",
		AuthorizationEndpoint: "https://sso.service.security.gov.uk/auth/oidc",
		TokenEndpoint:         "https://sso.service.security.gov.uk/auth/token",
		UserInfoEndpoint:      "https://sso.service.security.gov.uk/auth/profile",
		EndSessionEndpoint:    "https://sso.service.security.gov.uk/sign-out?from_app=",
		JWKSUri:               "https://sso.service.security.gov.uk/.well-known/jwks.json",
		ScopesSupported:       []string{"openid", "email", "profile"},
		ClaimsSupported:       []string{"aud", "email", "exp", "iat", "iss", "sub", "display_name", "nickname", "name", "given_name", "family_name"},
	}

	client := &OIDCClient{
		config:       config,
		clientID:     clientID,
		clientSecret: clientSecret,
		redirectURI:  redirectURI,
	}

	// Load JWK set for token validation
	if err := client.loadJWKSet(); err != nil {
		return nil, fmt.Errorf("failed to load JWK set: %w", err)
	}

	return client, nil
}

// GenerateAuthURL creates an authorization URL for the OIDC flow
func (c *OIDCClient) GenerateAuthURL(state string) string {
	params := url.Values{
		"client_id":     {c.clientID},
		"response_type": {"code"},
		"scope":         {"openid email profile"},
		"redirect_uri":  {c.redirectURI},
		"state":         {state},
	}

	return c.config.AuthorizationEndpoint + "?" + params.Encode()
}

// ExchangeCode exchanges an authorization code for tokens
func (c *OIDCClient) ExchangeCode(ctx context.Context, code string) (*TokenResponse, error) {
	data := url.Values{
		"grant_type":    {"authorization_code"},
		"client_id":     {c.clientID},
		"client_secret": {c.clientSecret},
		"code":          {code},
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.config.TokenEndpoint, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create token request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token exchange failed with status %d", resp.StatusCode)
	}

	var tokenResp TokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return nil, fmt.Errorf("failed to decode token response: %w", err)
	}

	return &tokenResp, nil
}

// ValidateToken validates a JWT token and extracts user information
func (c *OIDCClient) ValidateToken(tokenString string) (*User, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		// Verify signing method
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}

		// Get key ID from token header
		kid, ok := token.Header["kid"].(string)
		if !ok {
			return nil, fmt.Errorf("no kid in token header")
		}

		// Find the key in JWK set
		key, found := c.jwkSet.LookupKeyID(kid)
		if !found {
			// Refresh JWK set and try again
			if err := c.loadJWKSet(); err != nil {
				return nil, fmt.Errorf("failed to refresh JWK set: %w", err)
			}
			key, found = c.jwkSet.LookupKeyID(kid)
			if !found {
				return nil, fmt.Errorf("key not found in JWK set")
			}
		}

		// Convert to RSA public key
		var publicKey interface{}
		if err := key.Raw(&publicKey); err != nil {
			return nil, fmt.Errorf("failed to get public key: %w", err)
		}

		return publicKey, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to parse token: %w", err)
	}

	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, fmt.Errorf("invalid token claims")
	}

	// Verify issuer
	if claims.Issuer != c.config.Issuer {
		return nil, fmt.Errorf("invalid issuer: %s", claims.Issuer)
	}

	// Verify audience (client ID)
	if len(claims.Audience) == 0 || !contains(claims.Audience, c.clientID) {
		return nil, fmt.Errorf("invalid audience")
	}

	// Verify expiration
	if claims.ExpiresAt != nil && claims.ExpiresAt.Before(time.Now()) {
		return nil, fmt.Errorf("token expired")
	}

	user := &User{
		ID:          claims.Subject,
		Email:       claims.Email,
		DisplayName: claims.DisplayName,
		Name:        claims.Name,
	}

	return user, nil
}

// GetUserInfo fetches user information from the userinfo endpoint
func (c *OIDCClient) GetUserInfo(ctx context.Context, accessToken string) (*User, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.config.UserInfoEndpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create userinfo request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+accessToken)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get user info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("userinfo request failed with status %d", resp.StatusCode)
	}

	var user User
	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		return nil, fmt.Errorf("failed to decode user info: %w", err)
	}

	return &user, nil
}

// GenerateLogoutURL creates a logout URL
func (c *OIDCClient) GenerateLogoutURL(postLogoutRedirectURI string) string {
	if postLogoutRedirectURI == "" {
		return c.config.EndSessionEndpoint
	}

	params := url.Values{
		"post_logout_redirect_uri": {postLogoutRedirectURI},
	}

	return c.config.EndSessionEndpoint + "&" + params.Encode()
}

// GenerateState generates a random state parameter for CSRF protection
func GenerateState() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(b), nil
}

// TokenResponse represents the response from the token endpoint
type TokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int    `json:"expires_in"`
	RefreshToken string `json:"refresh_token,omitempty"`
	IDToken      string `json:"id_token"`
	Scope        string `json:"scope"`
}

// loadJWKSet loads the JWK set from the OIDC provider
func (c *OIDCClient) loadJWKSet() error {
	set, err := jwk.Fetch(context.Background(), c.config.JWKSUri)
	if err != nil {
		return fmt.Errorf("failed to fetch JWK set: %w", err)
	}
	c.jwkSet = set
	return nil
}

// contains checks if a slice contains a string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
