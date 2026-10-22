#!/bin/bash
# Lifecycle configuration "on-start" del notebook de SageMaker (m5.xlarge).
# Se ejecuta como root CADA vez que arranca la instancia. Pegar en:
#   SageMaker > Lifecycle configurations > Notebook instance > Create > Start notebook
set -euxo pipefail

PERSIST=/home/ec2-user/SageMaker            # único directorio que sobrevive a stop/start
DOCKER_ROOT=$PERSIST/docker-data           # data-root de Docker (imágenes, contenedores, volúmenes)
OLLAMA_DIR=$PERSIST/ollama                 # modelos descargados por Ollama
mkdir -p "$DOCKER_ROOT" "$OLLAMA_DIR"

# 1. Docker con data-root persistente
systemctl stop docker || true
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<JSON
{ "data-root": "$DOCKER_ROOT" }
JSON
systemctl start docker
docker info | grep "Docker Root Dir"

# 2. Contenedor de Ollama (la imagen y los modelos quedan en disco persistente: no se re-descargan)
if docker ps -a --format '{{.Names}}' | grep -q '^ollama$'; then
  docker start ollama
else
  docker run -d --name ollama --restart unless-stopped \
    -p 11434:11434 -v "$OLLAMA_DIR":/root/.ollama ollama/ollama:latest
fi

# 3. Modelo (solo la primera vez tarda; después ya está en $OLLAMA_DIR)
MODEL=${OLLAMA_MODEL:-gemma4:e2b}
for i in $(seq 1 30); do curl -sf localhost:11434/api/version && break; sleep 2; done
docker exec ollama ollama pull "$MODEL"

# 4. Túnel ngrok opcional: si existe $PERSIST/.ngrok_token (NO va en git) se expone el puerto 11434
if [ -f "$PERSIST/.ngrok_token" ]; then
  if [ ! -x "$PERSIST/bin/ngrok" ]; then
    mkdir -p "$PERSIST/bin"
    curl -sL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C "$PERSIST/bin"
  fi
  sudo -u ec2-user nohup "$PERSIST/bin/ngrok" http 11434 --authtoken "$(cat $PERSIST/.ngrok_token)" \
    --host-header=localhost:11434 --log=stdout > "$PERSIST/ngrok.log" 2>&1 &
fi
