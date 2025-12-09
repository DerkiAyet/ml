FROM python:3.11-slim

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Copier les modèles
COPY model/ ./model/

# Exposer le port
EXPOSE 5000

# Commande pour lancer l'application
CMD ["python", "app.py"]