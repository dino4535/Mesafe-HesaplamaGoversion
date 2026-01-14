# Build Stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install git required for fetching dependencies
RUN apk add --no-cache git

# Copy source code
COPY . .

# Run tidy to fetch dependencies and generate go.sum
# Since we don't have go.sum locally, we generate it here after copying source
RUN go mod tidy

# Build the binary
# CGO_ENABLED=0 creates a statically linked binary
RUN CGO_ENABLED=0 GOOS=linux go build -o pos-server main.go

# Runtime Stage
FROM alpine:latest

WORKDIR /app

# Install basic certificates
RUN apk --no-cache add ca-certificates tzdata

# Copy binary from builder
COPY --from=builder /app/pos-server .

# Copy templates and static files
# We accept that templates are part of the deployment
COPY templates/ ./templates/

# Make sure upload/output directories exist
RUN mkdir -p uploads output

# Expose port
EXPOSE 9595

# Run
CMD ["./pos-server"]
