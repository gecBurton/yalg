run:
	docker compose up --build -d

test:
	go test ./...