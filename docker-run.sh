#!/bin/bash

# Docker build and run script for Personality Classifier MLOps

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="personality-classifier"
TAG="latest"
CONTAINER_NAME="personality-app"
PORT="7860"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: ${IMAGE_NAME}:${TAG}"
    
    if docker build -t ${IMAGE_NAME}:${TAG} .; then
        print_status "Docker image built successfully!"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run Docker container
run_container() {
    print_status "Checking if container ${CONTAINER_NAME} is already running..."
    
    # Stop and remove existing container if running
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_warning "Stopping existing container: ${CONTAINER_NAME}"
        docker stop ${CONTAINER_NAME}
    fi
    
    if docker ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
        print_warning "Removing existing container: ${CONTAINER_NAME}"
        docker rm ${CONTAINER_NAME}
    fi
    
    print_status "Starting new container: ${CONTAINER_NAME}"
    print_status "Application will be available at: http://localhost:${PORT}"
    
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:7860 \
        ${IMAGE_NAME}:${TAG}
    
    if [ $? -eq 0 ]; then
        print_status "Container started successfully!"
        print_status "Access the application at: http://localhost:${PORT}"
        
        # Show container logs
        print_status "Container logs:"
        docker logs -f ${CONTAINER_NAME}
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to stop container
stop_container() {
    print_status "Stopping container: ${CONTAINER_NAME}"
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
    print_status "Container stopped and removed"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build     Build Docker image"
    echo "  run       Run Docker container"
    echo "  stop      Stop and remove Docker container"
    echo "  restart   Restart Docker container (build + run)"
    echo "  logs      Show container logs"
    echo "  shell     Open shell in running container"
    echo "  help      Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  IMAGE_NAME     Docker image name (default: personality-classifier)"
    echo "  TAG            Docker image tag (default: latest)"
    echo "  CONTAINER_NAME Container name (default: personality-app)"
    echo "  PORT           Host port to bind (default: 7860)"
}

# Function to show logs
show_logs() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_status "Showing logs for container: ${CONTAINER_NAME}"
        docker logs -f ${CONTAINER_NAME}
    else
        print_error "Container ${CONTAINER_NAME} is not running"
        exit 1
    fi
}

# Function to open shell in container
open_shell() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_status "Opening shell in container: ${CONTAINER_NAME}"
        docker exec -it ${CONTAINER_NAME} /bin/bash
    else
        print_error "Container ${CONTAINER_NAME} is not running"
        exit 1
    fi
}

# Main script logic
case "$1" in
    "build")
        build_image
        ;;
    "run")
        run_container
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        build_image
        run_container
        ;;
    "logs")
        show_logs
        ;;
    "shell")
        open_shell
        ;;
    "help"|"")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
