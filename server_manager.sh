#!/bin/bash
# server_manager.sh - Comprehensive server management script
# Handles backend, frontend, and ngrok tunnel setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000
NGROK_REGION="us"  # Change to your region: us, eu, ap, au, sa, jp, in

# PID files
BACKEND_PID_FILE=".backend.pid"
FRONTEND_PID_FILE=".frontend.pid"
NGROK_PID_FILE=".ngrok.pid"

# Log files
BACKEND_LOG="logs/backend.log"
FRONTEND_LOG="logs/frontend.log"
NGROK_LOG="logs/ngrok.log"

# Create logs directory
mkdir -p logs

#=============================================================================
# Helper Functions
#=============================================================================

print_header() {
    echo ""
    echo -e "${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}${BOLD}  $1${NC}"
    echo -e "${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

#=============================================================================
# Process Management
#=============================================================================

kill_port() {
    local port=$1
    local pids=$(lsof -t -i:$port 2>/dev/null || true)
    
    if [ ! -z "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        print_success "Killed process on port $port"
        sleep 1
    else
        print_info "No process running on port $port"
    fi
}

check_port_available() {
    local port=$1
    if lsof -i:$port >/dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

wait_for_url() {
    local url=$1
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
        echo -n "."
    done
    
    echo ""
    return 1
}

#=============================================================================
# Backend Management
#=============================================================================

start_backend() {
    print_header "Starting Backend Server"
    
    # Check if backend directory exists
    if [ ! -d "backend" ]; then
        print_error "Backend directory not found!"
        exit 1
    fi
    
    # Kill existing backend
    kill_port $BACKEND_PORT
    
    # Start backend
    print_info "Starting uvicorn on port $BACKEND_PORT..."
    cd backend
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
        print_info "Using virtual environment"
    fi
    
    # Start uvicorn in background
    nohup uvicorn app:app --reload --host 0.0.0.0 --port $BACKEND_PORT > "../$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "../$BACKEND_PID_FILE"
    
    cd ..
    
    # Wait for backend to be ready
    print_info "Waiting for backend to be ready"
    if wait_for_url "http://localhost:$BACKEND_PORT/health"; then
        print_success "Backend server started successfully (PID: $BACKEND_PID)"
        print_info "Backend URL: ${BOLD}http://localhost:$BACKEND_PORT${NC}"
    else
        print_error "Backend failed to start!"
        print_info "Check logs: $BACKEND_LOG"
        exit 1
    fi
}

stop_backend() {
    print_info "Stopping backend server..."
    
    if [ -f "$BACKEND_PID_FILE" ]; then
        PID=$(cat "$BACKEND_PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null || true
            print_success "Backend stopped (PID: $PID)"
        fi
        rm "$BACKEND_PID_FILE"
    fi
    
    kill_port $BACKEND_PORT
}

#=============================================================================
# Frontend Management
#=============================================================================

start_frontend() {
    print_header "Starting Frontend Server"
    
    # Check if frontend directory exists
    if [ ! -d "frontend" ]; then
        print_error "Frontend directory not found!"
        exit 1
    fi
    
    # Kill existing frontend
    kill_port $FRONTEND_PORT
    
    # Check if node_modules exists
    if [ ! -d "frontend/node_modules" ]; then
        print_warning "node_modules not found. Running npm install..."
        cd frontend
        npm install
        cd ..
    fi
    
    # Start frontend
    print_info "Starting React development server on port $FRONTEND_PORT..."
    cd frontend
    
    # Start npm in background
    nohup npm start > "../$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "../$FRONTEND_PID_FILE"
    
    cd ..
    
    # Wait for frontend to be ready
    print_info "Waiting for frontend to be ready (this may take a minute)"
    if wait_for_url "http://localhost:$FRONTEND_PORT"; then
        print_success "Frontend server started successfully (PID: $FRONTEND_PID)"
        print_info "Frontend URL: ${BOLD}http://localhost:$FRONTEND_PORT${NC}"
    else
        print_error "Frontend failed to start!"
        print_info "Check logs: $FRONTEND_LOG"
        exit 1
    fi
}

stop_frontend() {
    print_info "Stopping frontend server..."
    
    if [ -f "$FRONTEND_PID_FILE" ]; then
        PID=$(cat "$FRONTEND_PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null || true
            print_success "Frontend stopped (PID: $PID)"
        fi
        rm "$FRONTEND_PID_FILE"
    fi
    
    kill_port $FRONTEND_PORT
}

#=============================================================================
# ngrok Management
#=============================================================================

check_ngrok_installed() {
    if ! command -v ngrok &> /dev/null; then
        print_error "ngrok is not installed!"
        echo ""
        print_info "Install ngrok:"
        echo "  1. Visit: https://ngrok.com/download"
        echo "  2. Or use Homebrew: brew install ngrok/ngrok/ngrok"
        echo ""
        return 1
    fi
    return 0
}

start_ngrok() {
    print_header "Starting ngrok Tunnel"
    
    if ! check_ngrok_installed; then
        exit 1
    fi
    
    # Kill existing ngrok
    pkill ngrok 2>/dev/null || true
    sleep 1
    
    # Start ngrok for frontend
    print_info "Creating public tunnel to http://localhost:$FRONTEND_PORT..."
    nohup ngrok http $FRONTEND_PORT --region=$NGROK_REGION > "$NGROK_LOG" 2>&1 &
    NGROK_PID=$!
    echo $NGROK_PID > "$NGROK_PID_FILE"
    
    # Wait for ngrok to start
    sleep 3
    
    # Get public URL
    PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4)
    
    if [ ! -z "$PUBLIC_URL" ]; then
        print_success "ngrok tunnel created successfully!"
        echo ""
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}${BOLD}  🌐 PUBLIC URL (Share this link):${NC}"
        echo -e "${GREEN}${BOLD}  $PUBLIC_URL${NC}"
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        print_info "ngrok dashboard: ${BOLD}http://localhost:4040${NC}"
        print_info "Anyone can access your app at the public URL above"
        
        # Save URL to file
        echo "$PUBLIC_URL" > .ngrok_url
    else
        print_error "Failed to get ngrok public URL"
        print_info "Check logs: $NGROK_LOG"
    fi
}

stop_ngrok() {
    print_info "Stopping ngrok tunnel..."
    
    if [ -f "$NGROK_PID_FILE" ]; then
        PID=$(cat "$NGROK_PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null || true
            print_success "ngrok stopped (PID: $PID)"
        fi
        rm "$NGROK_PID_FILE"
    fi
    
    pkill ngrok 2>/dev/null || true
    rm -f .ngrok_url
}

#=============================================================================
# Main Commands
#=============================================================================

start_all() {
    print_header "PSL Analyzer - Starting All Services"
    
    start_backend
    start_frontend
    start_ngrok
    
    print_header "🎉 All Services Running!"
    
    echo ""
    echo -e "${BOLD}Local Access:${NC}"
    echo -e "  Frontend: ${CYAN}http://localhost:$FRONTEND_PORT${NC}"
    echo -e "  Backend:  ${CYAN}http://localhost:$BACKEND_PORT${NC}"
    echo ""
    
    if [ -f ".ngrok_url" ]; then
        PUBLIC_URL=$(cat .ngrok_url)
        echo -e "${BOLD}Public Access (Share with others):${NC}"
        echo -e "  ${GREEN}$PUBLIC_URL${NC}"
        echo ""
    fi
    
    echo -e "${BOLD}Management:${NC}"
    echo "  View logs:    tail -f logs/*.log"
    echo "  ngrok status: http://localhost:4040"
    echo "  Stop all:     ./server_manager.sh stop"
    echo ""
}

stop_all() {
    print_header "Stopping All Services"
    
    stop_ngrok
    stop_frontend
    stop_backend
    
    print_success "All services stopped"
}

restart_all() {
    stop_all
    sleep 2
    start_all
}

status() {
    print_header "Service Status"
    
    # Backend status
    if check_port_available $BACKEND_PORT; then
        echo -e "Backend:  ${RED}⬤ Stopped${NC}"
    else
        echo -e "Backend:  ${GREEN}⬤ Running${NC} (http://localhost:$BACKEND_PORT)"
    fi
    
    # Frontend status
    if check_port_available $FRONTEND_PORT; then
        echo -e "Frontend: ${RED}⬤ Stopped${NC}"
    else
        echo -e "Frontend: ${GREEN}⬤ Running${NC} (http://localhost:$FRONTEND_PORT)"
    fi
    
    # ngrok status
    if pgrep ngrok > /dev/null; then
        PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4 || echo "")
        if [ ! -z "$PUBLIC_URL" ]; then
            echo -e "ngrok:    ${GREEN}⬤ Running${NC} ($PUBLIC_URL)"
        else
            echo -e "ngrok:    ${GREEN}⬤ Running${NC}"
        fi
    else
        echo -e "ngrok:    ${RED}⬤ Stopped${NC}"
    fi
    
    echo ""
}

show_logs() {
    print_header "Tailing Logs (Ctrl+C to stop)"
    tail -f logs/*.log
}

#=============================================================================
# Usage
#=============================================================================

usage() {
    echo ""
    echo -e "${BOLD}PSL Analyzer - Server Manager${NC}"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|logs|backend|frontend|ngrok}"
    echo ""
    echo "Commands:"
    echo "  start       - Start all services (backend + frontend + ngrok)"
    echo "  stop        - Stop all services"
    echo "  restart     - Restart all services"
    echo "  status      - Show status of all services"
    echo "  logs        - Tail all log files"
    echo "  backend     - Start only backend"
    echo "  frontend    - Start only frontend"
    echo "  ngrok       - Start only ngrok tunnel"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start everything and get public URL"
    echo "  $0 status   # Check what's running"
    echo "  $0 logs     # Watch all logs"
    echo ""
}

#=============================================================================
# Main
#=============================================================================

case "${1:-}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart_all
        ;;
    status)
        status
        ;;
    logs)
        show_logs
        ;;
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    ngrok)
        if ! pgrep -f "uvicorn app:app" > /dev/null; then
            print_warning "Frontend not running. Starting it first..."
            start_frontend
        fi
        start_ngrok
        ;;
    *)
        usage
        exit 1
        ;;
esac

exit 0
