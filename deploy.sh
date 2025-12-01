#!/bin/bash

# Health MLOps Deployment Script
# Automates Docker build and deployment

set -e  # Exit on error

echo "=========================================="
echo "Health MLOps Project - Deployment Script"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Docker is installed
print_info "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker Desktop."
    exit 1
fi
print_success "Docker is installed"

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker Desktop."
    exit 1
fi
print_success "Docker is running"

# Stop and remove existing containers
print_info "Stopping existing containers..."
docker-compose down || true
print_success "Existing containers stopped"

# Build Docker image
print_info "Building Docker image..."
docker-compose build --no-cache
print_success "Docker image built successfully"

# Start services
print_info "Starting services..."
docker-compose up -d
print_success "Services started"

# Wait for services to be healthy
print_info "Waiting for services to be healthy..."
sleep 10

# Check service status
print_info "Checking service status..."
docker-compose ps

# Show logs
echo ""
print_info "Recent logs:"
docker-compose logs --tail=20

echo ""
echo "=========================================="
print_success "Deployment Complete!"
echo "=========================================="
echo ""
echo "Dashboard URL: http://localhost:7860"
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Restart services: docker-compose restart"
echo "  Service status:   docker-compose ps"
echo ""