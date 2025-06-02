import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import logging
import re

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Grok Twitter Search API",
    description="Simple Twitter search API using Grok models with HTTP requests",
    version="1.0.0"
)

# Request Models
class SearchTwittRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Search query - keywords, phrases, or company names to search for",
        example="MlpCare, MedicalPark, LivHospital"
    )
    start_date: Optional[str] = Field(
        default=None, 
        description="Start date (YYYY-MM-DD) - required when mode is 'off'",
        example="2025-01-01"
    )
    end_date: Optional[str] = Field(
        default=None, 
        description="End date (YYYY-MM-DD) - required when mode is 'off'",
        example="2025-05-30"
    )
    api_key: str = Field(
        ..., 
        description="X.AI API key (starts with 'xai-')",
        example="xai-HFJVJmxcRyiT7kt49IAgxNp1Av3YJr3B2k7ToU2kCxhOIbmzR17WomttkDJxTlWevKjy1NuBwCQZFz4i"
    )
    max_results: int = Field(
        default=25, 
        ge=1, 
        le=100, 
        description="Maximum number of results to return",
        example=25
    )
    model: str = Field(
        default="grok-3-latest", 
        description="Grok model to use (grok-3-latest, grok-3-mini, grok-2-1212)",
        example="grok-3-latest"
    )
    mode: str = Field(
        default="on", 
        description="Search mode: 'on' for live/recent tweets, 'off' for historical tweets, 'auto' for automatic selection",
        example="auto"
    )
    handles: Optional[List[str]] = Field(
        default=None, 
        description="Specific Twitter handles to search (without @ symbol)",
        example=["MedicalParkHG", "mlpcare", "livhospital"]
    )
    system_prompt: Optional[str] = Field(
        default=None, 
        description="Custom system prompt for AI model",
        example="You are a healthcare social media analyst. Focus on medical institutions and healthcare services."
    )
    user_prompt: Optional[str] = Field(
        default=None, 
        description="Custom user prompt for specific instructions",
        example="Find tweets about medical services, patient experiences, and healthcare announcements."
    )
    temperature: float = Field(
        default=0.1, 
        ge=0.0, 
        le=2.0, 
        description="Model temperature (0.0-2.0) - lower values for more focused results",
        example=0.2
    )
    max_tokens: int = Field(
        default=8000, 
        ge=100, 
        le=120000, 
        description="Maximum tokens for AI response",
        example=8000
    )
    response_format: str = Field(
        default="json", 
        description="Response format: 'json' for structured data, 'raw' for unprocessed content",
        example="json"
    )
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['on', 'off', 'auto']:
            raise ValueError('mode must be "on", "off", or "auto"')
        return v
    
    @validator('start_date', 'end_date')
    def validate_dates_when_mode_off(cls, v, values):
        mode = values.get('mode', 'on')
        if mode == 'off' and v is None:
            raise ValueError('start_date and end_date are required when mode is "off"')
        return v
    
    @validator('response_format')
    def validate_response_format(cls, v):
        if v not in ['json', 'raw']:
            raise ValueError('response_format must be either "json" or "raw"')
        return v

# Request Models (mevcut SearchTwittRequest'den sonra ekleyin)
class SearchNewsRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Search query - keywords, phrases, or company names to search for news",
        example="MlpCare, MedicalPark, LivHospital"
    )
    start_date: Optional[str] = Field(
        default="2025-01-01", 
        description="Start date (YYYY-MM-DD) for news search",
        example="2025-01-01"
    )
    end_date: Optional[str] = Field(
        default=None, 
        description="End date (YYYY-MM-DD) for news search",
        example="2025-05-30"
    )
    api_key: str = Field(
        ..., 
        description="X.AI API key (starts with 'xai-')",
        example="xai-HFJVJmxcRyiT7kt49IAgxNp1Av3YJr3B2k7ToU2kCxhOIbmzR17WomttkDJxTlWevKjy1NuBwCQZFz4i"
    )
    max_results: int = Field(
        default=25, 
        ge=1, 
        le=100, 
        description="Maximum number of news results to return",
        example=25
    )
    model: str = Field(
        default="grok-3-latest", 
        description="Grok model to use (grok-3-latest, grok-3-mini, grok-2-1212)",
        example="grok-3-latest"
    )
    system_prompt: Optional[str] = Field(
        default=None, 
        description="Custom system prompt for AI model"
    )
    user_prompt: Optional[str] = Field(
        default=None, 
        description="Custom user prompt for specific instructions"
    )
    temperature: float = Field(
        default=0.1, 
        ge=0.0, 
        le=2.0, 
        description="Model temperature (0.0-2.0) - lower values for more focused results",
        example=0.1
    )
    max_tokens: int = Field(
        default=8000, 
        ge=100, 
        le=120000, 
        description="Maximum tokens for AI response",
        example=8000
    )
    response_format: str = Field(
        default="json", 
        description="Response format: 'json' for structured data, 'raw' for unprocessed content",
        example="json"
    )    
    
    @validator('response_format')
    def validate_response_format(cls, v):
        if v not in ['json', 'raw']:
            raise ValueError('response_format must be either "json" or "raw"')
        return v


# Fixed fields
TWEET_FIELDS = {
    "title": "Tweet summary",
    "content": "Full tweet text", 
    "author": "Author display name",
    "username": "Author username (@handle)",
    "timestamp": "Date and time (YYYY-MM-DD HH:MM:SS)",
    "url": "Tweet URL",
    "type": "Tweet type (tweet/retweet/reply/quote)",
    "retweet_count": "Number of retweets",
    "like_count": "Number of likes",
    "hashtags": "List of hashtags",
    "mentioned_users": "List of mentioned users"
}

# Fixed news fields (NEWS_FIELDS global değişkeni olarak ekleyin)
NEWS_FIELDS = {
    "publish_date": "Publication date and time (YYYY-MM-DDTHH:MM:SS)",
    "source": "News source domain name",
    "title": "News headline/title",
    "summary": "Brief news summary (1-2 sentences)",
    "description": "Detailed news description",
    "link": "Full news article URL",
    "category": "News category (Finance, Health, Business, Technology, etc.)",
    "sentiment": "News sentiment (positive, negative, neutral)",
    "keywords": "List of relevant keywords",
    "tags": "List of relevant tags",
    "author": "Author or source name"
}


def get_twitter_system_prompt():
    """Geliştirilmiş system prompt - sadece gerçek tweetler için"""
    fields_json = ", ".join([f'"{k}": "{v}"' for k, v in TWEET_FIELDS.items()])
    
    return f"""You are a Twitter data analyst that searches for REAL, ACTUAL tweets only.

                CRITICAL: Only return REAL tweets that actually exist on Twitter/X. Do NOT generate fake, example, or demo data.

                Required JSON format:
                {{
                "tweets": [
                    {{
                    {fields_json}
                    }}
                ]
                }}

                IMPORTANT RULES:
                - ONLY include REAL tweets that you actually found
                - If no real tweets are found, return: {{"tweets": [], "message": "No real tweets found"}}
                - DO NOT create fake URLs, usernames, or content
                - DO NOT generate example data
                - Verify all tweet URLs are real (twitter.com/username/status/id format)
                - All usernames must be real existing accounts
                - All timestamps must be real posting times
                - Return ONLY valid JSON, no explanations
                - Use null for missing data
                - Numbers must be actual retweet/like counts from real tweets"""

def get_twitter_user_prompt(query: str):
    """Geliştirilmiş user prompt"""
    return f"""Search for REAL, ACTUAL tweets about: "{query}"

                IMPORTANT: 
                - Find only tweets that actually exist on Twitter/X
                - Do not create any fake or example data
                - If you cannot find real tweets, return an empty tweets array
                - Ensure all data is from actual posts

                Return JSON with real tweets only."""


def get_news_system_prompt():
    """Geliştirilmiş news system prompt - sadece gerçek haberler için"""
    fields_json = ", ".join([f'"{k}": "{v}"' for k, v in NEWS_FIELDS.items()])
    
    return f"""Sen bir web veri analisti ve JSON formatı uzmanısın. Görevin GERÇEK haber verilerini belirli bir JSON formatında düzenlemektir.

                MUTLAKA UYULMASI GEREKEN KURALLAR:
                1. Sadece GERÇEK, MEVCUT haberleri döndür - sahte veya demo veri YARATMA
                2. Sadece geçerli JSON formatında cevap ver
                3. Cevabın başında veya sonunda açıklama YAPMA
                4. Her haber için aşağıdaki yapıyı TAMAMEN takip et:

                {{
                "news": [
                    {{
                    {fields_json}
                    }}
                ]
                }}

                ÖNEMLI KURALLAR:
                - Asla demo, örnek veya sahte haber verisi kullanma
                - Tarihleri ISO 8601 formatında ver (YYYY-MM-DDTHH:MM:SS)
                - Eğer tarih bilgisi yoksa null olarak belirt
                - Kaynak web sitesi domain adını tam olarak ver (örneğin: hurriyet.com.tr)
                - Haberin başlığı ve özeti kısa ve öz olsun
                - Haberin açıklaması detaylı ve bilgilendirici olsun
                - Haberin kategorisi doğru şekilde sınıflandırılsın
                - Yazar veya kaynak adı varsa belirt, yoksa null bırak
                - Sentiment analizi yap (positive/negative/neutral)
                - Keywords ve tags ilgili ve anlamlı olsun
                - Tüm URL'ler gerçek ve erişilebilir olmalı
                - Sadece JSON döndür, başka hiçbir metin ekleme
                - Eğer gerçek haber bulunamazsa: {{"news": [], "message": "No real news found"}}
                - Haberleri en güncelden en eskiye doğru sırala"""

def get_news_user_prompt(query: str):
    """Geliştirilmiş news user prompt"""
    
    return f""""{query}" hakkında GERÇEK, GÜNCEL haberleri ara ve bul.

                ÖNEMLİ: 
                - Sadece gerçekte var olan haberleri döndür
                - Sahte, örnek veya demo haber verisi oluşturma
                - Gerçek haber bulamazsan boş news array döndür
                - Tüm veriler gerçek haber kaynaklarından olmalı
                - Haberleri tarihine göre sırala (en yeni önce)

                Sadece JSON formatında gerçek haber verileri döndür."""



def filter_real_tweets(tweets_data: Dict[str, Any]) -> Dict[str, Any]:
    """Gerçek tweet'leri filtrele ve sahte verileri temizle"""
    if not isinstance(tweets_data, dict) or "tweets" not in tweets_data:
        return tweets_data
    
    real_tweets = []
    
    for tweet in tweets_data.get("tweets", []):
        if not isinstance(tweet, dict):
            continue
            
        # URL kontrolü - gerçek Twitter URL formatında olmalı
        url = tweet.get("url", "")
        if url and not re.match(r"https?://(twitter\.com|x\.com)/\w+/status/\d+", url):
            logger.warning(f"Sahte URL tespit edildi: {url}")
            continue
            
        # Username kontrolü - @ ile başlamamalı ve gerçek format olmalı
        username = tweet.get("username", "")
        if username:
            # @ işaretini temizle
            username = username.lstrip("@")
            # Geçersiz karakterler kontrolü
            if not re.match(r"^[a-zA-Z0-9_]{1,15}$", username):
                logger.warning(f"Geçersiz username formatı: {username}")
                continue
            tweet["username"] = f"@{username}"
        
        # Timestamp kontrolü - gerçek tarih formatında olmalı
        timestamp = tweet.get("timestamp", "")
        if timestamp:
            try:
                # Tarih formatını kontrol et
                datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # Alternatif format
                    datetime.strptime(timestamp, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Geçersiz timestamp formatı: {timestamp}")
                    continue
        
        # Sayısal değerleri kontrol et
        for field in ["retweet_count", "like_count"]:
            value = tweet.get(field)
            if value is not None and not isinstance(value, (int, float)):
                try:
                    tweet[field] = int(value)
                except (ValueError, TypeError):
                    tweet[field] = 0
        
        # Content kontrolü - çok genel/sahte içerik kontrolü
        content = tweet.get("content", "")
        if content:
            # Sahte içerik kalıplarını kontrol et
            fake_patterns = [
                r"this is a sample tweet",
                r"example tweet",
                r"demo tweet", 
                r"sample content",
                r"lorem ipsum",
                r"test tweet"
            ]
            
            is_fake = any(re.search(pattern, content.lower()) for pattern in fake_patterns)
            if is_fake:
                logger.warning(f"Sahte içerik tespit edildi: {content[:50]}...")
                continue
        
        real_tweets.append(tweet)
    
    # Sonuçları güncelle
    tweets_data["tweets"] = real_tweets
    tweets_data["real_tweets_count"] = len(real_tweets)
    
    if len(real_tweets) == 0:
        tweets_data["message"] = "No real tweets found - all results were filtered as demo/fake data"
    
    return tweets_data

def filter_real_news(news_data: Dict[str, Any]) -> Dict[str, Any]:
    """Gerçek haberleri filtrele ve sahte verileri temizle"""
    if not isinstance(news_data, dict) or "news" not in news_data:
        return news_data
    
    real_news = []
    
    for news in news_data.get("news", []):
        if not isinstance(news, dict):
            continue
            
        # URL kontrolü - gerçek haber URL formatında olmalı
        link = news.get("link", "")
        if link:
            # Geçerli URL formatı kontrolü
            if not re.match(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", link):
                logger.warning(f"Geçersiz haber URL'si: {link}")
                continue
        
        # Source kontrolü - gerçek domain formatında olmalı
        source = news.get("source", "")
        if source:
            # Domain formatı kontrolü
            if not re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", source):
                logger.warning(f"Geçersiz source formatı: {source}")
                continue
        
        # Tarih kontrolü - ISO formatında olmalı
        publish_date = news.get("publish_date")
        if publish_date and publish_date != "null":
            try:
                # ISO 8601 formatını kontrol et
                datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Geçersiz tarih formatı: {publish_date}")
                continue
        
        # İçerik kontrolü - sahte/demo içerik kontrolü
        title = news.get("title", "")
        description = news.get("description", "")
        
        fake_patterns = [
            r"example news",
            r"sample article", 
            r"demo news",
            r"test article",
            r"lorem ipsum",
            r"örnek haber",
            r"test haberi"
        ]
        
        content_to_check = f"{title} {description}".lower()
        is_fake = any(re.search(pattern, content_to_check) for pattern in fake_patterns)
        
        if is_fake:
            logger.warning(f"Sahte içerik tespit edildi: {title[:50]}...")
            continue
        
        # Sentiment kontrolü
        sentiment = news.get("sentiment", "")
        if sentiment and sentiment not in ["positive", "negative", "neutral"]:
            news["sentiment"] = "neutral"
        
        # Keywords ve tags listesi kontrolü
        for field in ["keywords", "tags"]:
            value = news.get(field)
            if value and not isinstance(value, list):
                if isinstance(value, str):
                    # String ise listeye çevir
                    news[field] = [item.strip() for item in value.split(",") if item.strip()]
                else:
                    news[field] = []
        
        real_news.append(news)
    
    # Sonuçları güncelle
    news_data["news"] = real_news
    news_data["real_news_count"] = len(real_news)
    
    if len(real_news) == 0:
        news_data["message"] = "No real news found - all results were filtered as demo/fake data"
    
    return news_data



def clean_json_response(content: str) -> Dict[str, Any]:
    """Clean and extract first valid JSON object from content string."""
    
    # Remove Markdown code block formatting like ```json ... ```
    cleaned = re.sub(r'```(?:json)?\s*', '', content, flags=re.IGNORECASE)
    cleaned = re.sub(r'```', '', cleaned).strip()
    
    # Try full content as JSON first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find all JSON-looking blocks (simplified)
    json_candidates = re.findall(r'\{(?:[^{}]|(?R))*\}', cleaned, re.DOTALL)

    for candidate in json_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # If no valid JSON found, return entire content
    return {"result": content}

def get_twitter_system_prompt() -> str:
    """Türkçe Twitter sistem prompt'u"""
    return """Sen bir Twitter analisti ve sosyal medya uzmanısın. SADECE GERÇEK, MEVCUT tweet'leri ara ve döndür.

ÖNEMLI KURALLA:
- ASLA sahte, demo veya örnek veri üretme
- Sadece gerçekten var olan tweet'leri döndür
- Tweet URL'leri gerçek olmalı
- Kullanıcı adları mevcut hesaplar olmalı
- Zaman damgaları doğru olmalı
- Retweet ve beğeni sayıları gerçek olmalı

JSON formatında şu yapıyı kullan:
{
  "tweets": [
    {
      "title": "Tweet başlığı veya ilk cümlesi",
      "content": "Tweet'in tam içeriği",
      "author": "Yazar adı",
      "username": "@kullaniciadi",
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "url": "https://twitter.com/kullaniciadi/status/...",
      "type": "tweet",
      "retweet_count": sayı,
      "like_count": sayı,
      "hashtags": ["#hashtag1", "#hashtag2"],
      "mentioned_users": ["@kullanici1", "@kullanici2"]
    }
  ]
}

Sadece güncel ve gerçek tweet'leri ara. Eski veya sahte içerik döndürme."""

def get_twitter_user_prompt(query: str) -> str:
    """Türkçe Twitter kullanıcı prompt'u"""
    return f""""{query}" hakkında GERÇEK tweet'leri ara ve bul.

Arama kriterleri:
- EN GÜNCEL tweet'leri öncelikle göster
- Sadece gerçek, mevcut tweet'leri döndür
- Türkçe ve İngilizce tweet'leri dahil et
- Resmi hesapları ve doğrulanmış hesapları öncelikle göster
- İlgili hashtag'leri ve mention'ları dahil et

Lütfen sadece gerçekten var olan, doğrulanabilir tweet'leri döndür. Hiçbir sahte veri üretme."""

def get_news_system_prompt() -> str:
    """Türkçe haber sistem prompt'u"""
    return """Sen bir haber analisti ve medya uzmanısın. SADECE GERÇEK, MEVCUT haberleri ara ve döndür.

ÖNEMLI KURALLAR:
- ASLA sahte, demo veya örnek veri üretme
- Sadece gerçekten yayınlanmış haberleri döndür
- Haber URL'leri gerçek olmalı
- Kaynak siteleri doğrulanabilir olmalı
- Yayın tarihleri doğru olmalı

JSON formatında şu yapıyı kullan:
{
  "news": [
    {
      "publish_date": "YYYY-MM-DDTHH:MM:SS",
      "source": "kaynak-site.com",
      "title": "Haber başlığı",
      "summary": "Kısa özet",
      "description": "Detaylı açıklama",
      "link": "https://kaynak-site.com/haber-linki",
      "category": "Kategori",
      "sentiment": "positive/negative/neutral",
      "keywords": ["anahtar", "kelimeler"],
      "tags": ["etiket1", "etiket2"],
      "author": "Yazar adı"
    }
  ]
}

Sadece güncel ve gerçek haberleri ara. Eski veya sahte içerik döndürme."""

def get_news_user_prompt(query: str) -> str:
    """Türkçe haber kullanıcı prompt'u"""
    return f""""{query}" hakkında GERÇEK haberleri ara ve bul.

Arama kriterleri:
- EN GÜNCEL haberleri öncelikle göster
- Sadece gerçek, doğrulanabilir haberleri döndür
- Türkçe ve İngilizce haberleri dahil et
- Güvenilir kaynaklardan haberleri öncelikle göster
- İlgili anahtar kelimeleri ve etiketleri dahil et

Lütfen sadece gerçekten yayınlanmış, doğrulanabilir haberleri döndür. Hiçbir sahte veri üretme."""

def make_grok_twitter_request(request: SearchTwittRequest) -> Dict[str, Any]:
    """Make HTTP request to Grok API"""
    url = "https://api.x.ai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {request.api_key}"
    }
    
    # Türkçe prompt'ları kullan
    if request.response_format == "json":
        system_prompt = request.system_prompt or get_twitter_system_prompt()
        user_prompt = request.user_prompt or get_twitter_user_prompt(request.query)
    else:
        # Raw format için de Türkçe gerçek tweet vurgusu
        system_prompt = request.system_prompt or "Sen bir Twitter analistisin. SADECE GERÇEK, MEVCUT tweet'leri ara. Sahte veya demo veri üretme."
        user_prompt = request.user_prompt or f'"{request.query}" hakkında GERÇEK tweet\'leri ara. Sadece gerçekten var olan tweet\'leri döndür.'
    
    # Build search sources
    sources = [{"type": "x"}]
    if request.handles:
        sources = [{"type": "x", "x_handles": request.handles}]
    
    # Build search parameters based on mode
    search_params = {
        "mode": request.mode,
        "max_search_results": request.max_results,
        "sources": sources
    }
    
    # Tarih parametreleri - güncel verileri öncelemek için
    if request.mode == "off":
        # Historical mode - belirtilen tarih aralığını kullan
        if request.start_date:
            search_params["from_date"] = request.start_date
        if request.end_date:
            search_params["to_date"] = request.end_date
    elif request.mode == "on":
        # Live mode - son 7 gün ile sınırla (çok eskiye gitmesin)
        from datetime import datetime, timedelta
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        search_params["from_date"] = week_ago.strftime("%Y-%m-%d")
    elif request.mode == "auto":
        # Auto mode - son 30 gün ile sınırla
        from datetime import datetime, timedelta
        today = datetime.now()
        month_ago = today - timedelta(days=30)
        search_params["from_date"] = month_ago.strftime("%Y-%m-%d")
        
        # Eğer kullanıcı tarih belirtmişse onları da ekle
        if request.start_date:
            search_params["from_date"] = request.start_date
        if request.end_date:
            search_params["to_date"] = request.end_date
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ],
        "search_parameters": search_params,
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens
    }
    
    logger.info(f"Twitter araması - Query: {request.query}, Mode: {request.mode}, Format: {request.response_format}")
    if "from_date" in search_params:
        logger.info(f"Tarih aralığı: {search_params.get('from_date')} - {search_params.get('to_date', 'şimdi')}")
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Grok API hatası: {response.text}"
        )
    
    return response.json()

def make_grok_news_request(request: SearchNewsRequest) -> Dict[str, Any]:
    """Make HTTP request to Grok API for news search"""
    url = "https://api.x.ai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {request.api_key}"
    }
    
    # Türkçe prompt'ları kullan
    if request.response_format == "json":
        system_prompt = request.system_prompt or get_news_system_prompt()
        user_prompt = request.user_prompt or get_news_user_prompt(request.query)
    else:
        # Raw format için de Türkçe gerçek haber vurgusu
        system_prompt = request.system_prompt or "Sen bir haber analistisin. SADECE GERÇEK, MEVCUT haberleri ara. Sahte veya demo veri üretme."
        user_prompt = request.user_prompt or f'"{request.query}" hakkında GERÇEK haberleri ara. Sadece gerçekten yayınlanmış haberleri döndür.'
    
    # Build search sources
    sources = [{"type": "web"}, {"type": "news"}]
    
    # Tarih parametreleri - güncel haberleri öncelemek için
    from datetime import datetime, timedelta
    
    # Eğer start_date belirtilmemişse, son 30 gün ile sınırla
    if not request.start_date:
        today = datetime.now()
        month_ago = today - timedelta(days=30)
        from_date = month_ago.strftime("%Y-%m-%d")
    else:
        from_date = request.start_date
    
    search_params = {
        "from_date": from_date,
        "mode": "on",  # Haber araması için her zaman "on" kullan
        "max_search_results": request.max_results,
        "sources": sources
    }
    
    # End date ekle
    if request.end_date:
        search_params["to_date"] = request.end_date
    else:
        # End date belirtilmemişse bugünü kullan
        today = datetime.now()
        search_params["to_date"] = today.strftime("%Y-%m-%d")
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ],
        "search_parameters": search_params,
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens
    }
    
    logger.info(f"Haber araması - Query: {request.query}, Format: {request.response_format}")
    logger.info(f"Tarih aralığı: {search_params['from_date']} - {search_params.get('to_date', 'şimdi')}")
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Grok API hatası: {response.text}"
        )
    
    return response.json()

@app.get("/search_tweets", 
          summary="Search Real Tweets",
          description="""
Search for real tweets using Grok API with advanced filtering.

**Features:**
- ✅ Only returns real, actual tweets (no demo/fake data)
- ✅ Supports live and historical search modes
- ✅ Validates tweet URLs, usernames, and timestamps
- ✅ Filters out fake content patterns
- ✅ JSON and raw response formats

**Example Request:**
```
GET /search_tweets?query=MlpCare,MedicalPark,LivHospital&start_date=2025-01-01&end_date=2025-05-30&api_key=xai-your-api-key-here&max_results=25&model=grok-3-latest&mode=auto&temperature=0.2&max_tokens=120000&response_format=json&handles=MedicalParkHG,mlpcare&system_prompt=Focus%20on%20healthcare
```

**Query Parameters:**
- `query` (required): Search keywords or company names
- `api_key` (required): Your Grok API key
- `start_date` (optional): Start date for historical search (YYYY-MM-DD)
- `end_date` (optional): End date for historical search (YYYY-MM-DD)
- `max_results` (optional, default=25): Maximum number of results (1-100)
- `model` (optional, default=grok-3-latest): AI model to use
- `mode` (optional, default=auto): Search mode (on/off/auto)
- `handles` (optional): Comma-separated Twitter handles (without @)
- `system_prompt` (optional): Custom system prompt for AI model
- `user_prompt` (optional): Custom user prompt for specific instructions
- `temperature` (optional, default=0.2): AI temperature (0.0-2.0)
- `max_tokens` (optional, default=120000): Maximum response tokens
- `response_format` (optional, default=json): Response format (json/raw)

**Search Modes:**
- `on`: Live/recent tweets (default)
- `off`: Historical tweets (requires start_date/end_date)
- `auto`: Automatic mode selection

**Tips for Best Results:**
- Use specific keywords or company names
- Keep temperature low (0.1-0.3) for accurate results
- Use specific handles when targeting particular accounts
- Set mode to 'auto' for flexible searching
          """,
          responses={
              200: {
                  "description": "Successfully found tweets",
                  "content": {
                      "application/json": {
                          "example": {
                              "tweets": [
                                  {
                                      "title": "Medical Park announcement",
                                      "content": "Yeni sağlık hizmetlerimizle hastalarımıza daha iyi hizmet sunuyoruz.",
                                      "author": "Medical Park",
                                      "username": "@MedicalParkHG",
                                      "timestamp": "2025-05-30 14:30:00",
                                      "url": "https://twitter.com/MedicalParkHG/status/1234567890",
                                      "type": "tweet",
                                      "retweet_count": 15,
                                      "like_count": 45,
                                      "hashtags": ["#sağlık", "#MedicalPark"],
                                      "mentioned_users": []
                                  }
                              ],
                              "search_metadata": {
                                  "query": "MedicalPark",
                                  "mode": "auto",
                                  "model": "grok-3-latest",
                                  "tweets_found": 1,
                                  "real_tweets_only": True,
                                  "filtered_fake_data": True
                              }
                          }
                      }
                  }
              },
              400: {"description": "Invalid request parameters"},
              500: {"description": "API error or no response from Grok"}
          }
)
async def search_tweets(
    query: str,
    api_key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 25,
    model: str = "grok-3-latest",
    mode: str = "on",
    handles: Optional[str] = None,  # Comma-separated handles
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 120000,
    response_format: str = "json"
) -> Union[Dict[str, Any], str]:
    """Search tweets and return response based on format"""
    try:
        # Convert comma-separated handles to list
        handles_list = None
        if handles:
            handles_list = [h.strip() for h in handles.split(',')]
        
        # Create request object from query parameters
        request = SearchTwittRequest(
            query=query,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            model=model,
            mode=mode,
            handles=handles_list,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        # Make request to Grok API
        response_data = make_grok_twitter_request(request)
        
        # Extract content from response
        if 'choices' in response_data and response_data['choices']:
            content = response_data['choices'][0]['message']['content']
            
            # Return based on requested format
            if request.response_format == "raw":
                # Return raw content as is
                return {"content": content}
            else:
                # Process as JSON
                result = clean_json_response(content)
                
                # Gerçek tweet'leri filtrele
                result = filter_real_tweets(result)
                
                # Add metadata
                if isinstance(result, dict) and "tweets" in result:
                    metadata = {
                        "query": request.query,
                        "mode": request.mode,
                        "model": request.model,
                        "tweets_found": len(result.get("tweets", [])),
                        "real_tweets_only": True,
                        "filtered_fake_data": True
                    }
                    
                    if request.mode == "off":
                        metadata["date_range"] = f"{request.start_date} to {request.end_date}"
                    else:
                        metadata["search_type"] = "live/recent"
                        
                    result["search_metadata"] = metadata
                
                return result
        else:
            raise HTTPException(status_code=500, detail="No response from Grok API")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request error: {e}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search_news", 
          summary="Search Real News",
          description="""
Search for real news using Grok API with advanced filtering.

**Features:**
- ✅ Only returns real, actual news (no demo/fake data)
- ✅ Validates news URLs, sources, and timestamps
- ✅ Filters out fake content patterns
- ✅ JSON and raw response formats
- ✅ Sentiment analysis and categorization

**Example Request:**
```
GET /search_news?query=MlpCare,MedicalPark,LivHospital&start_date=2025-01-01&end_date=2025-05-30&api_key=xai-your-api-key-here&max_results=25&model=grok-3-latest&temperature=0.1&max_tokens=8000&response_format=json&system_prompt=Focus%20on%20healthcare%20news
```

**Query Parameters:**
- `query` (required): Search keywords or company names
- `api_key` (required): Your Grok API key
- `start_date` (optional, default=2025-01-01): Start date for search (YYYY-MM-DD)
- `end_date` (optional): End date for search (YYYY-MM-DD)
- `max_results` (optional, default=25): Maximum number of results (1-100)
- `model` (optional, default=grok-3-latest): AI model to use
- `system_prompt` (optional): Custom system prompt for AI model
- `user_prompt` (optional): Custom user prompt for specific instructions
- `temperature` (optional, default=0.1): AI temperature (0.0-2.0)
- `max_tokens` (optional, default=8000): Maximum response tokens
- `response_format` (optional, default=json): Response format (json/raw)

**Tips for Best Results:**
- Use specific keywords or company names
- Keep temperature low (0.1-0.3) for accurate results
          """,
          responses={
              200: {
                  "description": "Successfully found news",
                  "content": {
                      "application/json": {
                          "example": {
                              "news": [
                                  {
                                      "publish_date": "2025-05-30T14:30:00",
                                      "source": "hurriyet.com.tr",
                                      "title": "MedicalPark yeni sağlık merkezi açıyor",
                                      "summary": "MedicalPark Hastane Grubu İstanbul'da yeni bir sağlık merkezi açacağını duyurdu.",
                                      "description": "Detaylı haber açıklaması...",
                                      "link": "https://hurriyet.com.tr/gundem/medicalpark-yeni-merkez",
                                      "category": "Health",
                                      "sentiment": "positive",
                                      "keywords": ["MedicalPark", "sağlık", "hastane"],
                                      "tags": ["hastane", "sağlık sektörü"],
                                      "author": "Haber Editörü"
                                  }
                              ],
                              "search_metadata": {
                                  "query": "MedicalPark",
                                  "model": "grok-3-latest",
                                  "news_found": 1,
                                  "real_news_only": True,
                                  "filtered_fake_data": True
                              }
                          }
                      }
                  }
              },
              400: {"description": "Invalid request parameters"},
              500: {"description": "API error or no response from Grok"}
          }
)
async def search_news(
    query: str,
    api_key: str,
    start_date: Optional[str] = "2025-01-01",
    end_date: Optional[str] = None,
    max_results: int = 25,
    model: str = "grok-3-latest",
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 120000,
    response_format: str = "json"
) -> Union[Dict[str, Any], str]:
    """Search news and return response based on format"""
    try:
        # Create request object from query parameters
        request = SearchNewsRequest(
            query=query,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        # Make request to Grok API
        response_data = make_grok_news_request(request)
        
        # Extract content from response
        if 'choices' in response_data and response_data['choices']:
            content = response_data['choices'][0]['message']['content']
            
            # Return based on requested format
            if request.response_format == "raw":
                # Return raw content as is
                return {"content": content}
            else:
                # Process as JSON
                result = clean_json_response(content)
                
                # Gerçek haberleri filtrele
                result = filter_real_news(result)
                
                # Add metadata
                if isinstance(result, dict) and "news" in result:
                    metadata = {
                        "query": request.query,
                        "model": request.model,
                        "news_found": len(result.get("news", [])),
                        "real_news_only": True,
                        "filtered_fake_data": True,
                        "date_range": f"{request.start_date} to {request.end_date or 'now'}"
                    }                    
                        
                    result["search_metadata"] = metadata
                
                return result
        else:
            raise HTTPException(status_code=500, detail="No response from Grok API")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request error: {e}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        logger.error(f"News search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))   
    

@app.get("/models")
async def list_models():
    """List available Grok models"""
    return {
        "models": [
            {
                "name": "grok-3-latest",
                "description": "Latest Grok 3 model - recommended"
            },
            {
                "name": "grok-3-mini",
                "description": "Lightweight Grok 3 model"
            },
            {
                "name": "grok-2-1212",
                "description": "Grok 2 model"
            }
        ],
        "default": "grok-3-latest"
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)