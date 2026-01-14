# POS Distance Calculator Web Application

This is a web application for calculating distances between KACC customers and POS locations using geographical coordinates.

## Features

- User authentication with login/logout
- Excel file upload with drag-and-drop support
- Distance calculation using Haversine formula
- Processing modes: nearest location or within radius
- Progress tracking with visual indicators
- Secure file download for results
- Corporate compliance and data responsibility notices

## Deployment on Ubuntu Server

### Prerequisites

- Ubuntu server with Python 3.8+
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd n8n
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the webapp directory:
   ```bash
   cd webapp
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. The application will be accessible at `http://your-server-ip:9595`

### Authentication

- Username: `user`
- Password: `Dino202545`

### Firewall Configuration

If using UFW firewall, allow traffic on port 9595:
```bash
sudo ufw allow 9595
```

### Running as a Service (Optional)

For production use, you may want to run the application as a systemd service:

1. Create a service file:
   ```bash
   sudo nano /etc/systemd/system/pos-calculator.service
   ```

2. Add the following content (adjust paths as needed):
   ```
   [Unit]
   Description=POS Distance Calculator
   After=network.target

   [Service]
   Type=simple
   User=www-data
   WorkingDirectory=/path/to/n8n/webapp
   ExecStart=/usr/bin/python3 /path/to/n8n/webapp/app.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable pos-calculator
   sudo systemctl start pos-calculator
   ```

## Running with Docker

### Prerequisites

- Docker
- Docker Compose

### Running the Application

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd n8n
   ```

2. Build and run the container:
   ```bash
   docker-compose up -d
   ```

3. The application will be accessible at `http://your-server-ip:9595`

4. To stop the application:
   ```bash
   docker-compose down
   ```

5. To view logs:
   ```bash
   docker-compose logs -f
   ```

### Building the Docker Image

To build the image manually:
```bash
docker build -t pos-distance-app .
```

To run the built image:
```bash
docker run -d -p 9595:9595 --name pos-distance-app pos-distance-app