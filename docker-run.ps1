# Docker build and run script for Personality Classifier MLOps (PowerShell)

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "run", "stop", "restart", "logs", "shell", "help")]
    [string]$Action = "help"
)

# Default values
$IMAGE_NAME = "personality-classifier"
$TAG = "latest"
$CONTAINER_NAME = "personality-app"
$PORT = "7860"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to build Docker image
function Build-Image {
    Write-Status "Building Docker image: ${IMAGE_NAME}:${TAG}"
    
    $result = docker build -t "${IMAGE_NAME}:${TAG}" .
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Docker image built successfully!"
    } else {
        Write-Error "Failed to build Docker image"
        exit 1
    }
}

# Function to run Docker container
function Run-Container {
    Write-Status "Checking if container $CONTAINER_NAME is already running..."
    
    # Stop and remove existing container if running
    $runningContainer = docker ps -q -f "name=$CONTAINER_NAME"
    if ($runningContainer) {
        Write-Warning "Stopping existing container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME | Out-Null
    }
    
    $existingContainer = docker ps -aq -f "name=$CONTAINER_NAME"
    if ($existingContainer) {
        Write-Warning "Removing existing container: $CONTAINER_NAME"
        docker rm $CONTAINER_NAME | Out-Null
    }
    
    Write-Status "Starting new container: $CONTAINER_NAME"
    Write-Status "Application will be available at: http://localhost:$PORT"
    
    $result = docker run -d --name $CONTAINER_NAME -p "${PORT}:7860" "${IMAGE_NAME}:${TAG}"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Container started successfully!"
        Write-Status "Access the application at: http://localhost:$PORT"
        
        # Show container logs
        Write-Status "Container logs:"
        docker logs -f $CONTAINER_NAME
    } else {
        Write-Error "Failed to start container"
        exit 1
    }
}

# Function to stop container
function Stop-Container {
    Write-Status "Stopping container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME | Out-Null
    docker rm $CONTAINER_NAME | Out-Null
    Write-Status "Container stopped and removed"
}

# Function to show help
function Show-Help {
    Write-Host "Usage: .\docker-run.ps1 [OPTION]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  build     Build Docker image"
    Write-Host "  run       Run Docker container"
    Write-Host "  stop      Stop and remove Docker container"
    Write-Host "  restart   Restart Docker container (build + run)"
    Write-Host "  logs      Show container logs"
    Write-Host "  shell     Open shell in running container"
    Write-Host "  help      Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\docker-run.ps1 build"
    Write-Host "  .\docker-run.ps1 run"
    Write-Host "  .\docker-run.ps1 restart"
}

# Function to show logs
function Show-Logs {
    $runningContainer = docker ps -q -f "name=$CONTAINER_NAME"
    if ($runningContainer) {
        Write-Status "Showing logs for container: $CONTAINER_NAME"
        docker logs -f $CONTAINER_NAME
    } else {
        Write-Error "Container $CONTAINER_NAME is not running"
        exit 1
    }
}

# Function to open shell in container
function Open-Shell {
    $runningContainer = docker ps -q -f "name=$CONTAINER_NAME"
    if ($runningContainer) {
        Write-Status "Opening shell in container: $CONTAINER_NAME"
        docker exec -it $CONTAINER_NAME /bin/bash
    } else {
        Write-Error "Container $CONTAINER_NAME is not running"
        exit 1
    }
}

# Main script logic
switch ($Action) {
    "build" {
        Build-Image
    }
    "run" {
        Run-Container
    }
    "stop" {
        Stop-Container
    }
    "restart" {
        Build-Image
        Run-Container
    }
    "logs" {
        Show-Logs
    }
    "shell" {
        Open-Shell
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown option: $Action"
        Show-Help
        exit 1
    }
}
