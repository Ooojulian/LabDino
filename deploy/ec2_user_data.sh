#!/bin/bash
# User data para una EC2 Amazon Linux 2023 (t3.small basta). Security group: abrir puerto 80.
# Las URLs de Ollama/imagen se ponen luego en /opt/labdino.env (NO en git) y se reinicia el contenedor.
set -eux
dnf install -y docker git
systemctl enable --now docker
git clone https://github.com/cofferr/LabDino /opt/LabDino
cd /opt/LabDino
docker build -f deploy/Dockerfile -t labdino .
touch /opt/labdino.env
docker run -d --name labdino --restart unless-stopped --env-file /opt/labdino.env -p 80:8000 labdino
