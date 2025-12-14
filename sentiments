from textblob import TextBlob
import re

def analyze_sentiment(text):
    """
    Analisa sentimento de um texto
    
    Returns:
        {
            'polarity': -1 a 1 (negativo a positivo),
            'subjectivity': 0 a 1 (objetivo a subjetivo),
            'sentiment': 'BULLISH' / 'BEARISH' / 'NEUTRO',
            'score': 0 a 100
        }
    """
    try:
        if not text or len(text.strip()) == 0:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'NEUTRO',
                'score': 50
            }
        
        # Análise com TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 a 1
        subjectivity = blob.sentiment.subjectivity  # 0 a 1
        
        # Converter para score 0-100
        score = int((polarity + 1) / 2 * 100)
        
        # Determinar sentimento
        if polarity > 0.1:
            sentiment = 'BULLISH'
        elif polarity < -0.1:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRO'
        
        return {
            'polarity': round(polarity, 2),
            'subjectivity': round(subjectivity, 2),
            'sentiment': sentiment,
            'score': score
        }
    
    except Exception as e:
        print(f"[SENTIMENT] Erro: {str(e)}")
        return {
            'polarity': 0,
            'subjectivity': 0,
            'sentiment': 'NEUTRO',
            'score': 50
        }


def analyze_news_sentiment(news_list):
    """
    Analisa sentimento de uma lista de notícias
    
    Args:
        news_list: Lista de dicionários com 'title' e 'description'
    
    Returns:
        {
            'average_sentiment': 'BULLISH' / 'BEARISH' / 'NEUTRO',
            'average_score': 0-100,
            'bullish_count': número,
            'bearish_count': número,
            'neutral_count': número,
            'details': [lista de análises individuais]
        }
    """
    try:
        if not news_list or len(news_list) == 0:
            return {
                'average_sentiment': 'NEUTRO',
                'average_score': 50,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'details': []
            }
        
        sentiments = []
        scores = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        details = []
        
        for news in news_list:
            # Combinar título + descrição
            text = f"{news.get('title', '')} {news.get('description', '')}"
            
            # Analisar sentimento
            analysis = analyze_sentiment(text)
            sentiments.append(analysis['sentiment'])
            scores.append(analysis['score'])
            
            # Contar sentimentos
            if analysis['sentiment'] == 'BULLISH':
                bullish_count += 1
            elif analysis['sentiment'] == 'BEARISH':
                bearish_count += 1
            else:
                neutral_count += 1
            
            # Adicionar detalhes
            details.append({
                'title': news.get('title', 'Sem título'),
                'sentiment': analysis['sentiment'],
                'score': analysis['score'],
                'source': news.get('source', 'Desconhecido')
            })
        
        # Calcular média
        average_score = int(sum(scores) / len(scores))
        
        # Determinar sentimento geral
        if average_score > 60:
            average_sentiment = 'BULLISH'
        elif average_score < 40:
            average_sentiment = 'BEARISH'
        else:
            average_sentiment = 'NEUTRO'
        
        return {
            'average_sentiment': average_sentiment,
            'average_score': average_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'details': details
        }
    
    except Exception as e:
        print(f"[SENTIMENT] Erro ao analisar notícias: {str(e)}")
        return {
            'average_sentiment': 'NEUTRO',
            'average_score': 50,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'details': []
        }


def get_sentiment_emoji(sentiment):
    """Retorna emoji baseado no sentimento"""
    if sentiment == 'BULLISH':
        return '🟢'
    elif sentiment == 'BEARISH':
        return '🔴'
    else:
        return '🟡'


def get_sentiment_color(sentiment):
    """Retorna cor CSS baseada no sentimento"""
    if sentiment == 'BULLISH':
        return '#10b981'  # Verde
    elif sentiment == 'BEARISH':
        return '#ef4444'  # Vermelho
    else:
        return '#f59e0b'  # Amarelo
