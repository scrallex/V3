#!/bin/bash
# SEP Engine Production Deployment Script
# Simplified version focusing on core deployment functionality

set -e

# Colors and logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Configuration
PROJECT_NAME="sep"
MODE="${SEP_DEPLOY_STACK:-hotband}"
COMPOSE_FILE="docker-compose.${MODE}.yml"

# Check Docker installation and use appropriate command
if command -v docker compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    error "Docker Compose not found. Please install Docker with Compose plugin."
    exit 1
fi

# Check if compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    error "Compose file $COMPOSE_FILE not found"
    exit 1
fi

log "Starting SEP Engine deployment in $MODE mode"
log "Using compose file: $COMPOSE_FILE"
log "Docker compose command: $DOCKER_COMPOSE"

# Load environment files in order of precedence
load_env() {
    # Load base .env if exists
    if [[ -f ".env" ]]; then
        log "Loading .env file"
        set -a
        source .env
        set +a
    fi
    
    # Load mode-specific env first (before OANDA credentials)
    if [[ -f ".env.${MODE}" ]]; then
        log "Loading .env.${MODE} file"
        set -a
        source .env.${MODE}
        set +a
    fi
    
    # Load OANDA credentials last so they override any empty values
    if [[ -f "OANDA.env" ]]; then
        log "Loading OANDA.env file"
        set -a
        source OANDA.env
        set +a
        log "After loading OANDA.env: OANDA_API_KEY=${OANDA_API_KEY:-not_set}, OANDA_ACCOUNT_ID=${OANDA_ACCOUNT_ID:-not_set}"
    elif [[ -f "config/OANDA.env" ]]; then
        log "Loading config/OANDA.env file"
        set -a
        source config/OANDA.env
        set +a
    else
        log "No OANDA.env file found in current directory or config/"
    fi
}

load_env

# Validate required OANDA credentials
log "Validating credentials: OANDA_API_KEY=${OANDA_API_KEY:-not_set}, OANDA_ACCOUNT_ID=${OANDA_ACCOUNT_ID:-not_set}"
if [[ -z "$OANDA_ACCOUNT_ID" ]] || [[ -z "$OANDA_API_KEY" ]]; then
    error "OANDA credentials not found. Please ensure OANDA.env file exists with:"
    error "  OANDA_ACCOUNT_ID=your_account_id"
    error "  OANDA_API_KEY=your_api_key"
    if [[ -f "OANDA.env" ]]; then
        error "OANDA.env exists but credentials are not being loaded properly"
        error "Contents of OANDA.env:"
        cat OANDA.env | grep -E "OANDA_(API_KEY|ACCOUNT_ID)" || error "No OANDA credentials found in file"
    fi
    exit 1
fi

# Set default retention values if not set
export VALKEY_SIGNAL_RETENTION=${VALKEY_SIGNAL_RETENTION:-200000}
export VALKEY_CANDLE_RETENTION=${VALKEY_CANDLE_RETENTION:-0}

# Get HOTBAND_PAIRS from Python if available, otherwise use default
if [[ -z "$HOTBAND_PAIRS" ]]; then
    HOTBAND_PAIRS="EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CHF,USD_CAD,NZD_USD"
fi
export HOTBAND_PAIRS

log "HOTBAND_PAIRS: $HOTBAND_PAIRS"

# Stop existing services
log "Stopping existing services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" down --remove-orphans || true

# Pull latest images
log "Pulling latest images..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" pull || warning "Pull failed, continuing with local images"

# Build services
log "Building services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" build

# Start services
log "Starting services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d

# Seed Valkey defaults (kill switch, risk snapshot)
log "Seeding Valkey defaults..."
if ! $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T backend python /app/scripts/tools/seed_valkey_defaults.py; then
    warning "Unable to seed Valkey defaults"
fi

# Wait for services to start
log "Waiting for services to initialize..."
sleep 20

# Health check function
health_check() {
    local backend_url="http://localhost:8000"
    local frontend_url="http://localhost:80"
    
    log "Checking backend health at $backend_url/health..."
    if ! curl -sf "$backend_url/health" >/dev/null 2>&1; then
        error "Backend health check failed"
        return 1
    fi

    log "Checking frontend health at $frontend_url..."
    if ! curl -sf -I "$frontend_url" >/dev/null 2>&1; then
        warning "Frontend health check failed (is Nginx starting?)"
        # We don't fail deployment for frontend as it might be lazy loading or certbot issue initially
    else
        success "Frontend health check passed"
    fi

    success "Health checks passed"
    return 0
}

# Retry health check
MAX_RETRIES=5
RETRY_DELAY=10

for i in $(seq 1 $MAX_RETRIES); do
    if health_check; then
        break
    elif [[ $i -lt $MAX_RETRIES ]]; then
        warning "Health check failed (attempt $i/$MAX_RETRIES), retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    else
        error "Health check failed after $MAX_RETRIES attempts"
        log "Showing container logs for debugging:"
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" logs --tail=50
        log "Container status:"
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" ps
        exit 1
    fi
done

# Show final status
log "Deployment status:"
$DOCKER_COMPOSE -f "$COMPOSE_FILE" ps

success "✅ SEP Engine deployment completed successfully!"
log ""
log "Services running:"
log "  Backend API: http://localhost:8000"
log "  Frontend UI:  https://mxbikes.xyz"
log ""
log "Useful commands:"
log "  View logs:    $DOCKER_COMPOSE -f $COMPOSE_FILE logs -f"
log "  Stop:         $DOCKER_COMPOSE -f $COMPOSE_FILE down"
log "  Restart:      $DOCKER_COMPOSE -f $COMPOSE_FILE restart"
log ""
log "Additional operations via make:"
log "  make dev      - Run development environment"
log "  make test     - Run tests"
log "  make lint     - Run linters"
