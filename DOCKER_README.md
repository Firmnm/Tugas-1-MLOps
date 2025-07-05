# Docker Deployment Guide

This guide explains how to build and run the Personality Classifier MLOps application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

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
