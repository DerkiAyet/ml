from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import json

app = Flask(__name__)
CORS(app)

print("Serveur de détection de fraude avec MySQL et ML...")

# Configuration MySQL
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Change to your MySQL username
    'password': 'ayet2004',  # Change to your MySQL password
    'database': 'fraud_detection',  # Change to your database name
    'port': 3306
}

# Charger les modèles
MODEL_PATH = './model/'

def create_mysql_connection():
    """Crée une connexion à MySQL"""
    try:
        connection = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database'],
            port=MYSQL_CONFIG['port'],
            auth_plugin='mysql_native_password'  # <- here, directly
        )
        if connection.is_connected():
            print("Connexion MySQL établie")
            return connection
    except Error as e:
        print(f"Erreur MySQL: {e}")
    return None

def initialize_database():
    """Initialise la base de données avec des tables supplémentaires si nécessaire"""
    connection = create_mysql_connection()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        # Vérifier si la table existe déjà
        cursor.execute("SHOW TABLES LIKE 'transactions'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            print("Table 'transactions' existe déjà")
            
            # Vérifier les colonnes existantes
            cursor.execute("DESCRIBE transactions")
            columns = [col[0] for col in cursor.fetchall()]
            print(f"Colonnes existantes: {columns}")
            
            # Ajouter des colonnes manquantes si nécessaire
            if 'unix_time' in columns:
                print("Colonne 'unix_time' est VARCHAR, conversion recommandée")
            
            # Créer une table pour les statistiques des cartes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS card_stats (
                    cc_num VARCHAR(30) PRIMARY KEY,
                    last_transaction_time DATETIME,
                    total_transactions INT DEFAULT 0,
                    total_amount DECIMAL(15, 2) DEFAULT 0,
                    avg_amount DECIMAL(15, 2) DEFAULT 0,
                    last_24h_transactions INT DEFAULT 0,
                    last_24h_amount DECIMAL(15, 2) DEFAULT 0,
                    common_category VARCHAR(50),
                    fraud_count INT DEFAULT 0,
                    total_fraud_count INT DEFAULT 0,
                    last_updated DATETIME,
                    INDEX idx_cc_num (cc_num),
                    INDEX idx_last_transaction (last_transaction_time)
                )
            """)
            print("Table 'card_stats' créée/vérifiée")
            
        else:
            print("Table 'transactions' n'existe pas")
            print("Veuillez créer la table manuellement ou utiliser votre script SQL")
    
    except Error as e:
        print(f"Erreur d'initialisation: {e}")
    finally:
        cursor.close()
        connection.close()

def load_models_safely():
    """Charge les modèles avec vérification"""
    print("\n CHARGEMENT DES MODÈLES:")
    print("=" * 50)
    
    try:
        # Vérifier l'existence des fichiers
        required_files = ['best_fraud_model.pkl', 'scaler.pkl', 'label_encoders.pkl']
        for file in required_files:
            file_path = os.path.join(MODEL_PATH, file)
            if not os.path.exists(file_path):
                print(f"Fichier manquant: {file}")
                return None, None, None
            else:
                size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"{file}: {size:.2f} MB")
        
        # Charger les modèles
        model = joblib.load(os.path.join(MODEL_PATH, 'best_fraud_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(MODEL_PATH, 'label_encoders.pkl'))
        
        print(f"Modèle chargé: {type(model).__name__}")
        print(f"Scaler chargé: {type(scaler).__name__}")
        print(f"Encodeurs: {len(label_encoders)} variables catégorielles")
        print("=" * 50)
        
        return model, scaler, label_encoders
        
    except Exception as e:
        print(f"Erreur de chargement: {str(e)}")
        print("=" * 50)
        return None, None, None

def get_card_statistics(cc_num, transaction_time=None):
    """Récupère les statistiques d'une carte de crédit depuis MySQL"""
    connection = create_mysql_connection()
    if not connection:
        return None
    
    cursor = connection.cursor(dictionary=True)
    stats = None
    
    try:
        # Vérifier si la carte existe dans card_stats
        cursor.execute("""
            SELECT * FROM card_stats 
            WHERE cc_num = %s
        """, (cc_num,))
        
        stats = cursor.fetchone()
        
        if stats:
            print(f"Statistiques trouvées pour la carte: {cc_num}")
        else:
            print(f"Nouvelle carte détectée: {cc_num}")
            # Initialiser des statistiques par défaut
            stats = {
                'cc_num': cc_num,
                'last_transaction_time': None,
                'total_transactions': 0,
                'total_amount': 0,
                'avg_amount': 0,
                'last_24h_transactions': 0,
                'last_24h_amount': 0,
                'common_category': None,
                'fraud_count': 0,
                'total_fraud_count': 0,
                'last_updated': None
            }
        
        #  Calculer les statistiques depuis la table transactions
        if transaction_time:
            # Convertir transaction_time en datetime si c'est un timestamp
            if isinstance(transaction_time, (int, float, str)):
                try:
                    if isinstance(transaction_time, (int, float)):
                        transaction_time = datetime.fromtimestamp(transaction_time)
                    else:
                        
                        transaction_time = datetime.strptime(transaction_time, '%Y-%m-%d %H:%M:%S')
                except:
                    transaction_time = datetime.now()
            
            # Calculer le temps écoulé depuis la dernière transaction
            time_since_last = None
            if stats['last_transaction_time']:
                last_time = stats['last_transaction_time']
                if isinstance(last_time, str):
                    last_time = datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
                # Conversion en HEURES
                time_since_last = (transaction_time - last_time).total_seconds() / 3600 
            else:
                time_since_last = 24  # selon le code de training
            
            # Transactions des dernières 24 heures
            twenty_four_hours_ago = transaction_time - timedelta(hours=24)
            cursor.execute("""
                SELECT COUNT(*) as count, SUM(amt) as total_amount
                FROM transactions 
                WHERE cc_num = %s 
                AND trans_date_trans_time >= %s
                AND trans_date_trans_time <= %s
            """, (cc_num, twenty_four_hours_ago, transaction_time))
            
            recent_stats = cursor.fetchone()
            last_24h_transactions = recent_stats['count'] if recent_stats and recent_stats['count'] else 0
            last_24h_amount = float(recent_stats['total_amount']) if recent_stats and recent_stats['total_amount'] else 0
            
            # Catégorie la plus commune pour cette carte
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM transactions 
                WHERE cc_num = %s
                GROUP BY category
                ORDER BY count DESC
                LIMIT 1
            """, (cc_num,))
            
            common_category_row = cursor.fetchone()
            common_category = common_category_row['category'] if common_category_row else None
            
            # Taux de fraude pour cette carte
            cursor.execute("""
                SELECT COUNT(*) as fraud_count
                FROM transactions 
                WHERE cc_num = %s 
                AND is_fraud = 1
            """, (cc_num,))
            
            fraud_row = cursor.fetchone()
            fraud_count = fraud_row['fraud_count'] if fraud_row and fraud_row['fraud_count'] else 0
            
            # Total des transactions pour cette carte
            cursor.execute("""
                SELECT COUNT(*) as total_count
                FROM transactions 
                WHERE cc_num = %s
            """, (cc_num,))
            
            total_row = cursor.fetchone()
            total_transactions = total_row['total_count'] if total_row and total_row['total_count'] else 0
            
            # Mettre à jour les stats
            stats.update({
                'time_since_last_transaction_min': time_since_last,
                'card_transactions_last_24h': last_24h_transactions,
                'usual_category_for_card': common_category,
                'card_fraud_rate': fraud_count / total_transactions if total_transactions > 0 else 0,
                'card_total_transactions': total_transactions
            })
            
    except Error as e:
        print(f"Erreur lors de la récupération des statistiques: {e}")
    finally:
        cursor.close()
        connection.close()
    
    return stats

def update_card_statistics(cc_num, transaction_data, is_fraud):
    """Met à jour les statistiques de la carte après une transaction"""
    connection = create_mysql_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    
    try:
        # Vérifier si la carte existe déjà dans card_stats
        cursor.execute("SELECT * FROM card_stats WHERE cc_num = %s", (cc_num,))
        card_exists = cursor.fetchone()
        
        now = datetime.now()
        amt = float(transaction_data.get('amt', 0))
        
        if card_exists:
            # Mettre à jour les statistiques existantes
            cursor.execute("""
                UPDATE card_stats 
                SET 
                    last_transaction_time = %s,
                    total_transactions = total_transactions + 1,
                    total_amount = total_amount + %s,
                    avg_amount = (total_amount + %s) / (total_transactions + 1),
                    last_24h_transactions = (
                        SELECT COUNT(*) 
                        FROM transactions 
                        WHERE cc_num = %s 
                        AND trans_date_trans_time >= DATE_SUB(%s, INTERVAL 24 HOUR)
                    ),
                    last_24h_amount = (
                        SELECT COALESCE(SUM(amt), 0)
                        FROM transactions 
                        WHERE cc_num = %s 
                        AND trans_date_trans_time >= DATE_SUB(%s, INTERVAL 24 HOUR)
                    ),
                    common_category = (
                        SELECT category 
                        FROM transactions 
                        WHERE cc_num = %s 
                        GROUP BY category 
                        ORDER BY COUNT(*) DESC 
                        LIMIT 1
                    ),
                    fraud_count = fraud_count + %s,
                    total_fraud_count = total_fraud_count + %s,
                    last_updated = %s
                WHERE cc_num = %s
            """, (
                now, amt, amt, cc_num, now, cc_num, now, 
                cc_num, 1 if is_fraud else 0, 1 if is_fraud else 0, now, cc_num
            ))
        else:
            # Insérer de nouvelles statistiques
            cursor.execute("""
                INSERT INTO card_stats (
                    cc_num, last_transaction_time, total_transactions, 
                    total_amount, avg_amount, last_24h_transactions,
                    last_24h_amount, common_category, fraud_count,
                    total_fraud_count, last_updated
                ) VALUES (%s, %s, 1, %s, %s, 1, %s, %s, %s, %s, %s)
            """, (
                cc_num, now, amt, amt, amt, 
                transaction_data.get('category'), 
                1 if is_fraud else 0, 1 if is_fraud else 0, now
            ))
        
        connection.commit()
        print(f"Statistiques mises à jour pour la carte: {cc_num}")
        return True
        
    except Error as e:
        print(f"Erreur lors de la mise à jour des statistiques: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def save_transaction_to_mysql(transaction_data, is_fraud, fraud_probability):
    """Sauvegarde la transaction dans MySQL"""
    connection = create_mysql_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    
    try:
        # Convertir unix_time si nécessaire
        unix_time = transaction_data.get('unix_time')
        if not unix_time:
            # Générer un timestamp Unix actuel
            unix_time = str(int(datetime.now().timestamp()))
        
        # S'assurer que trans_date_trans_time est au bon format
        trans_time = transaction_data.get('trans_date_trans_time')
        if not trans_time:
            trans_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(trans_time, (int, float)):
            trans_time = datetime.fromtimestamp(trans_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Insérer la transaction
        cursor.execute("""
            INSERT INTO transactions (
                cc_num, trans_date_trans_time, amt, 
                unix_time, category, is_fraud
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            transaction_data['cc_num'],
            trans_time,
            float(transaction_data['amt']),
            str(unix_time),
            transaction_data['category'],
            1 if is_fraud else 0
        ))
        
        connection.commit()
        print(f"Transaction sauvegardée dans MySQL")
        
        # Mettre à jour les statistiques de la carte
        update_card_statistics(transaction_data['cc_num'], transaction_data, is_fraud)
        
        return True
        
    except Error as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

# Charger les modèles au démarrage
model, scaler, label_encoders = load_models_safely()

# Initialiser la base de données
initialize_database()

# Historique
prediction_history = []

@app.route('/api/health', methods=['GET'])
def health():
    # Tester la connexion MySQL
    mysql_status = "connected" if create_mysql_connection() else "disconnected"
    
    return jsonify({
        'status': 'ok',
        'mysql': mysql_status,
        'model_loaded': model is not None,
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Données reçues: {data.keys()}")
        
        # Vérifier les champs requis MIS À JOUR
        required = [
            'cc_num', 'amt', 'category', 'gender', 
            'lat', 'long', 'merch_lat', 'merch_long',
            'merchant', 'city', 'state', 'zip', 'job'
        ]
        for field in required:
            if field not in data:
                return jsonify({'error': f'Champ manquant: {field}'}), 400
        
        # Extraire le jour et l'heure actuels
        current_time = datetime.now()
        if 'trans_date_trans_time' in data:
            try:
                current_time = datetime.strptime(data['trans_date_trans_time'], '%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        # Récupérer les statistiques de la carte
        cc_num = data['cc_num']
        card_stats = get_card_statistics(cc_num, current_time)
        
        if not card_stats:
            card_stats = {
                'time_since_last_transaction_hours': 24,
                'card_transactions_last_24h': 0,
                'usual_category_for_card': None,
                'card_fraud_rate': 0,
                'card_total_transactions': 0,
                'avg_amount': float(data.get('amt', 0))
            }
        
        # Calculer la déviation du montant
        current_amount = float(data['amt'])
        avg_amount = float(card_stats.get('avg_amount', current_amount))
        amount_deviation = 0
        if avg_amount > 0:
            amount_deviation = (current_amount - avg_amount) / avg_amount
        
        # NOUVEAU : Calculer la distance en km (simplifié)
        import math
        lat1, lon1 = float(data['lat']), float(data['long'])
        lat2, lon2 = float(data['merch_lat']), float(data['merch_long'])
        # Distance approximative (simplifiée pour test)
        distance_km = math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111
        
        # Créer un DataFrame pandas avec TOUTES les features
        features = {
            # Features originales
            'amt': current_amount,
            'category': str(data['category']),
            'gender': str(data['gender']),
            'lat': lat1,
            'long': lon1,
            'merch_lat': lat2,
            'merch_long': lon2,
            'age': int(data.get('age', 35)),
            
            # NOUVELLES FEATURES REQUISES
            'merchant': str(data['merchant']),
            'city': str(data['city']),
            'state': str(data['state']),
            'zip': str(data['zip']),
            'job': str(data['job']),
            'distance_km': distance_km,  # Calculé
            'city_pop': int(data.get('city_pop', 50000)),  # Garder mais avec défaut
            
            # Features temporelles
            'trans_date_trans_time_month': current_time.month,
            'trans_date_trans_time_dayofweek': current_time.weekday(),
            'trans_date_trans_time_hour': current_time.hour,
            
            # Features comportementales
            'time_since_last_same_card': card_stats.get('time_since_last_transaction_hours', 24),
            'card_transactions_last_24h': card_stats.get('card_transactions_last_24h', 0),
            'amount_deviation_from_card_avg': amount_deviation,
            'card_fraud_rate_safe': card_stats.get('card_fraud_rate', 0),
            'usual_category_for_card': card_stats.get('usual_category_for_card', 'unknown')
        }
        
        # Encodage de la catégorie habituelle
        usual_category = features['usual_category_for_card']
        features['usual_category_match'] = 1 if (usual_category and usual_category != 'unknown' and usual_category == features['category']) else 0
        
        print(f"Features préparées: {len(features)} colonnes")
        
        # Créer DataFrame
        df = pd.DataFrame([features])
        print(f"DataFrame créé: {df.shape}")
        
        if model is not None and scaler is not None and label_encoders is not None:
            try:
                # 1. Encoder les variables catégorielles
                for col, encoder in label_encoders.items():
                    if col in df.columns:
                        try:
                            value = str(features[col])
                            if value in encoder.classes_:
                                df[col] = encoder.transform([value])[0]
                                print(f"✅ Encodé {col}: '{value}' -> {df[col].iloc[0]}")
                            else:
                                # Valeur inconnue - utiliser la première catégorie
                                df[col] = encoder.transform([encoder.classes_[0]])[0]
                                print(f"Valeur inconnue pour {col}: '{value}' -> {encoder.classes_[0]}")
                        except Exception as e:
                            print(f"Erreur encodage {col}: {e}")
                            df[col] = 0
                
                # 2. S'assurer d'avoir TOUTES les colonnes du modèle
                # Liste COMPLÈTE des features d'entraînement
                all_expected_columns = [
                    'merchant', 'category', 'amt', 'gender', 'city', 'state', 'zip',
                    'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long',
                    'trans_date_trans_time_hour', 'trans_date_trans_time_dayofweek',
                    'trans_date_trans_time_month', 'age', 'distance_km',
                    'time_since_last_same_card', 'card_transactions_last_24h',
                    'amount_deviation_from_card_avg', 'usual_category_for_card',
                    'card_fraud_rate_safe', 'usual_category_match'
                ]
                
                # Ajouter cc_num_encoded si présent dans le modèle
                if 'cc_num_encoded' in label_encoders:
                    all_expected_columns.append('cc_num_encoded')
                    # Pour l'instant, mettre une valeur par défaut
                    df['cc_num_encoded'] = 0
                
                # S'assurer que toutes les colonnes existent
                missing_cols = []
                for col in all_expected_columns:
                    if col not in df.columns:
                        df[col] = 0
                        missing_cols.append(col)
                
                if missing_cols:
                    print(f"Colonnes manquantes ajoutées (0): {missing_cols}")
                
                # Réorganiser les colonnes dans l'ordre d'entraînement
                df = df[all_expected_columns]
                
                # 3. Normaliser les features numériques
                numerical_features = [
                    'amt', 'lat', 'long', 'merch_lat', 'merch_long',
                    'city_pop', 'age', 'distance_km',
                    'trans_date_trans_time_hour', 'trans_date_trans_time_dayofweek',
                    'trans_date_trans_time_month', 'time_since_last_same_card',
                    'card_transactions_last_24h', 'amount_deviation_from_card_avg',
                    'card_fraud_rate_safe', 'usual_category_match'
                ]
                
                # Filtrer pour garder seulement celles qui existent dans df
                existing_numerical = [col for col in numerical_features if col in df.columns]
                
                if existing_numerical:
                    df[existing_numerical] = scaler.transform(df[existing_numerical])
                    print(f"Données normalisées ({len(existing_numerical)} features)")
                
                # 4. Faire la prédiction
                print(f"Prédiction avec {df.shape[1]} features...")
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(df)[0]
                    fraud_probability = float(probability[1])  # Probabilité de fraude
                else:
                    probability = [0.5, 0.5]  # Valeur par défaut
                    fraud_probability = 0.5
                
                prediction = model.predict(df)[0]
                is_fraud = bool(prediction == 1)
                
                print(f"Prédiction: {'FRAUDE' if is_fraud else 'LÉGITIME'}")
                print(f"Probabilité: {fraud_probability:.2%}")
                
                method = "ML Model (complet)"
                
            except Exception as ml_error:
                print(f"Erreur ML: {ml_error}")
                import traceback
                traceback.print_exc()
                # Fallback aux règles
                fraud_probability = calculate_fraud_probability_with_card_stats(features, card_stats)
                is_fraud = fraud_probability > 0.5
                method = "Rules (ML failed)"
        else:
            # Mode règles uniquement
            fraud_probability = calculate_fraud_probability_with_card_stats(features, card_stats)
            is_fraud = fraud_probability > 0.5
            method = "Rules (No model)"
        
        # Sauvegarder la transaction dans MySQL
        save_transaction_to_mysql({
            'cc_num': cc_num,
            'trans_date_trans_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'amt': current_amount,
            'unix_time': str(int(current_time.timestamp())),
            'category': data['category'],
            'merchant': data.get('merchant', 'unknown'),
            'city': data.get('city', 'unknown'),
            'state': data.get('state', 'unknown')
        }, is_fraud, fraud_probability)
        
        # Déterminer le niveau de risque
        if fraud_probability < 0.3:
            risk_level = "FAIBLE"
            risk_color = "green"
        elif fraud_probability < 0.7:
            risk_level = "MOYEN"
            risk_color = "orange"
        else:
            risk_level = "ÉLEVÉ"
            risk_color = "red"
        
        # Préparer la réponse
        result = {
            'is_fraud': is_fraud,
            'fraud_probability': round(fraud_probability, 4),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': round(fraud_probability if is_fraud else (1 - fraud_probability), 4),
            'prediction_method': method,
            'timestamp': datetime.now().isoformat(),
            'transaction_amount': current_amount,
            'features_used': len(all_expected_columns) if 'all_expected_columns' in locals() else 0,
            'card_statistics': {
                'time_since_last_transaction_hours': round(card_stats.get('time_since_last_transaction_hours', 0), 2),
                'transactions_last_24h': card_stats.get('card_transactions_last_24h', 0),
                'usual_category': card_stats.get('usual_category_for_card', 'unknown'),
                'fraud_rate': round(card_stats.get('card_fraud_rate', 0) * 100, 2)
            }
        }
        
        # Sauvegarder dans l'historique
        prediction_history.append({
            **result,
            'input_data': {k: v for k, v in data.items() if k != 'cc_num'},  # Masquer le numéro de carte
            'card_stats': card_stats
        })
        
        # Garder seulement les 100 dernières
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        print(f"Réponse envoyée avec {result['features_used']} features")
        return jsonify(result)
        
    except Exception as e:
        print(f"Erreur générale: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500
    
    
def calculate_fraud_probability_with_card_stats(features, card_stats):
    """Calcul amélioré de probabilité de fraude avec statistiques de carte"""
    risk = 0.0
    
    # Facteurs de base
    if features['amt'] > 300: risk += 0.2
    if features['trans_date_trans_time_hour'] in [0, 1, 2, 3, 4, 5]: risk += 0.1
    if features['category'] in ['electronics', 'travel']: risk += 0.1
    if features.get('age', 35) < 25: risk += 0.1
    
    # Distance géographique
    distance = abs(features['lat'] - features['merch_lat']) + abs(features['long'] - features['merch_long'])
    if distance > 10: risk += 0.2
    
    # Facteurs basés sur l'historique de la carte - MODIFIÉ POUR HEURES
    time_since_last = features.get('time_since_last_same_card', 24)  # En HEURES maintenant
    
    if time_since_last < 0.0833:  # Moins de 5 minutes (5/60 ≈ 0.0833 heures)
        risk += 0.3
    elif time_since_last > 168:  # Plus d'une semaine (24*7 = 168 heures)
        risk += 0.1
    
    transactions_24h = card_stats.get('card_transactions_last_24h', 0)
    if transactions_24h > 20:  # Plus de 20 transactions en 24h
        risk += 0.2
    
    amount_deviation = features.get('amount_deviation_from_card_avg', 0)
    if abs(amount_deviation) > 2:  # Montant 2x supérieur ou inférieur à la moyenne
        risk += 0.2
    
    fraud_rate = card_stats.get('card_fraud_rate', 0)
    if fraud_rate > 0.1:  # Taux de fraude historique > 10%
        risk += 0.3
    
    # Vérifier si la catégorie correspond à l'habitude
    usual_category = card_stats.get('usual_category_for_card')
    if usual_category and usual_category != 'unknown':
        if features['category'] != usual_category:
            risk += 0.1
    
    return min(risk, 0.95)

@app.route('/api/card_stats/<cc_num>', methods=['GET'])
def get_card_stats(cc_num):
    """Récupère les statistiques d'une carte spécifique"""
    stats = get_card_statistics(cc_num)
    
    if stats:
        return jsonify({
            'success': True,
            'card_number': cc_num,
            'stats': stats
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Carte non trouvée ou erreur de base de données'
        }), 404

@app.route('/')
def index():
    return "Bienvenue sur le serveur de détection de fraude avec MySQL !"

@app.route('/api/history', methods=['GET'])
def get_history():
    """Retourne l'historique des prédictions"""
    return jsonify({
        'history': prediction_history[-20:],
        'total_count': len(prediction_history),
        'fraud_count': sum(1 for p in prediction_history if p['is_fraud'])
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """Retourne des statistiques globales"""
    connection = create_mysql_connection()
    if not connection:
        return jsonify({'error': 'Base de données non disponible'}), 500
    
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Statistiques générales
        cursor.execute("SELECT COUNT(*) as total, SUM(is_fraud) as fraud_count FROM transactions")
        general_stats = cursor.fetchone()
        
        cursor.execute("""
            SELECT 
                (SELECT COUNT(DISTINCT cc_num) FROM transactions) as unique_cards,
                (SELECT AVG(amt) FROM transactions) as avg_amount,
                (SELECT MAX(amt) FROM transactions) as max_amount
        """)
        transaction_stats = cursor.fetchone()
        
        # Top 5 cartes avec le plus de transactions
        cursor.execute("""
            SELECT cc_num, COUNT(*) as transaction_count
            FROM transactions
            GROUP BY cc_num
            ORDER BY transaction_count DESC
            LIMIT 5
        """)
        top_cards = cursor.fetchall()
        
        # Distribution des fraudes par catégorie
        cursor.execute("""
            SELECT category, COUNT(*) as fraud_count
            FROM transactions
            WHERE is_fraud = 1
            GROUP BY category
            ORDER BY fraud_count DESC
        """)
        fraud_by_category = cursor.fetchall()
        
        return jsonify({
            'general': {
                'total_transactions': general_stats['total'] or 0,
                'fraud_count': general_stats['fraud_count'] or 0,
                'fraud_rate': round((general_stats['fraud_count'] or 0) / (general_stats['total'] or 1) * 100, 2)
            },
            'transactions': transaction_stats,
            'top_cards': top_cards,
            'fraud_by_category': fraud_by_category
        })
        
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SERVEUR FLASK AVEC MYSQL ET MODÈLES ML")
    print("=" * 60)
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Modèle chargé: {'OUI' if model else 'NON'}")
    print(f"Modèles dans: {MODEL_PATH}")
    print(f"Base MySQL: {MYSQL_CONFIG['database']}")
    print("\n ENDPOINTS:")
    print("  GET  /api/health          - Vérification serveur")
    print("  POST /api/predict         - Prédiction de fraude")
    print("  GET  /api/card_stats/<cc> - Statistiques d'une carte")
    print("  GET  /api/history         - Historique")
    print("  GET  /api/stats           - Statistiques globales")
    print("=" * 60)
    print(f"\n API disponible sur: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)