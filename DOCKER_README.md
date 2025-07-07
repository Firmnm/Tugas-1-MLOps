# Docker Deployment Guide

This guide explains how to build and run the Personality Classifier MLOps application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Manual Docker Steps (Detailed)

### Step 1: Navigate to Project Directory
```bash
# Windows
cd e:\KULIAH\Semester Antara\Projek\Tugas-1-MLOps

# Linux/Mac
cd /path/to/Tugas-1-MLOps
```

### Step 2: Build Docker Image
```bash
# Build the Docker image with tag
docker build -t personality-classifier:latest .

# Build with specific tag and no cache (for clean build)
docker build --no-cache -t personality-classifier:v1.0 .

# Build and show detailed output
docker build --progress=plain -t personality-classifier:latest .
```

### Step 3: Verify Image Creation
```bash
# List Docker images
docker images

# Check specific image
docker images personality-classifier
```

### Step 4: Run Container
```bash
# Run container in detached mode
docker run -d --name personality-app -p 7860:7860 personality-classifier:latest

# Run with custom name and port
docker run -d --name my-personality-app -p 8080:7860 personality-classifier:latest

# Run with environment variables
docker run -d --name personality-app -p 7860:7860 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=7860 \
  personality-classifier:latest

# Run in interactive mode (for debugging)
docker run -it --name personality-app -p 7860:7860 personality-classifier:latest
```

### Step 5: Verify Container is Running
```bash
# Check running containers
docker ps

# Check all containers (running and stopped)
docker ps -a

# Check container status
docker inspect personality-app
```

### Step 6: Access Application
Open your web browser and go to:
- http://localhost:7860 (if using default port)
- http://localhost:8080 (if using custom port 8080)

### Step 7: Monitor Container
```bash
# View container logs
docker logs personality-app

# Follow logs in real-time
docker logs -f personality-app

# View last 50 lines of logs
docker logs --tail 50 personality-app

# Check container resource usage
docker stats personality-app
```

### Step 8: Container Management Commands
```bash
# Stop container
docker stop personality-app

# Start stopped container
docker start personality-app

# Restart container
docker restart personality-app

# Pause container
docker pause personality-app

# Unpause container
docker unpause personality-app

# Access container shell
docker exec -it personality-app /bin/bash

# Copy files from container
docker cp personality-app:/app/logs ./local-logs

# Copy files to container
docker cp ./local-file.txt personality-app:/app/
```

### Step 9: Cleanup
```bash
# Stop and remove container
docker stop personality-app
docker rm personality-app

# Remove container forcefully
docker rm -f personality-app

# Remove image
docker rmi personality-classifier:latest

# Clean up unused containers and images
docker system prune

# Clean up everything including volumes
docker system prune -a --volumes
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the application
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop the application
docker-compose down
```

### Option 2: Using Docker directly

#### On Windows PowerShell:
```powershell
# Build the image
.\docker-run.ps1 build

# Run the container
.\docker-run.ps1 run

# Stop the container
.\docker-run.ps1 stop

# Restart (build + run)
.\docker-run.ps1 restart
```

#### On Linux/Mac:
```bash
# Make script executable
chmod +x docker-run.sh

# Build the image
./docker-run.sh build

# Run the container
./docker-run.sh run

# Stop the container
./docker-run.sh stop

# Restart (build + run)
./docker-run.sh restart
```

#### Manual Docker commands:
```bash
# Build the image
docker build -t personality-classifier:latest .

# Run the container
docker run -d --name personality-app -p 7860:7860 personality-classifier:latest

# Stop and remove the container
docker stop personality-app
docker rm personality-app
```

## Accessing the Application

Once the container is running, access the Gradio interface at:
- **Local**: http://localhost:7860
- **Network**: http://YOUR_HOST_IP:7860

## Container Management

### View logs
```bash
# Using docker-compose
docker-compose logs -f

# Using Docker directly
docker logs -f personality-app

# Using helper script (PowerShell)
.\docker-run.ps1 logs
```

### Access container shell
```bash
# Using Docker directly
docker exec -it personality-app /bin/bash

# Using helper script (PowerShell)
.\docker-run.ps1 shell
```

### Health check
The container includes a health check that monitors the application status:
```bash
docker inspect --format='{{.State.Health.Status}}' personality-app
```

## Configuration

### Environment Variables

The following environment variables can be configured:

- `GRADIO_SERVER_NAME`: Server bind address (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

### Port Configuration

To change the host port, modify the port mapping:
```bash
# Use port 8080 instead of 7860
docker run -d --name personality-app -p 8080:7860 personality-classifier:latest
```

Or update the `docker-compose.yml` file:
```yaml
ports:
  - "8080:7860"
```

## Development

### Volume Mounting for Development

For development purposes, you can mount local directories:

```yaml
# In docker-compose.yml
volumes:
  - ./App:/app/App
  - ./Model:/app/Model:ro
  - ./Data:/app/Data:ro
```

### Rebuilding after Changes

```bash
# Using docker-compose
docker-compose build --no-cache

# Using helper script
.\docker-run.ps1 restart
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port 7860
   netstat -tulpn | grep 7860
   
   # Use different port
   docker run -d --name personality-app -p 7861:7860 personality-classifier:latest
   ```

2. **Container fails to start**
   ```bash
   # Check logs
   docker logs personality-app
   
   # Check container status
   docker ps -a
   ```

3. **Model files not found**
   - Ensure all model files are present in the `Model/` directory
   - Check file permissions
   - Verify the build context includes all necessary files

### Performance Optimization

1. **Multi-stage build** (for production):
   ```dockerfile
   # Add to Dockerfile for smaller image size
   FROM python:3.11-slim as builder
   # ... build dependencies
   
   FROM python:3.11-slim as runtime
   # ... copy only runtime files
   ```

2. **Resource limits**:
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

## Security Considerations

- The application runs as root by default. For production, consider running as non-root user
- Ensure model files are from trusted sources
- Use secrets management for sensitive configuration
- Consider using Docker secrets for production deployment

## Production Deployment

For production deployment, consider:

1. Using a reverse proxy (nginx, traefik)
2. SSL/TLS termination
3. Container orchestration (Kubernetes, Docker Swarm)
4. Monitoring and logging
5. Backup strategies for model files

## Support

If you encounter issues:

1. Check the container logs
2. Verify all required files are present
3. Ensure Docker has sufficient resources allocated
4. Check network connectivity and firewall settings
