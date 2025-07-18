run:
	docker compose up -d postgres keycloak
	go run .

test:
	go test ./...


.PHONY: run test