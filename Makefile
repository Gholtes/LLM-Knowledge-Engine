
.PHONY: build
build:
	docker compose -f docker_compose.yml build

.PHONY: run
run:
	docker compose -f docker_compose.yml up -d

.PHONY: build-run
build-run:
	@make build
	@make run