import os
from pathlib import Path
import dj_database_url
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------
# 🔐 Security Settings
# ---------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-tz)$u8=9l)c(p6vt2z*s5@c9cz101k*h)67%xqg=ub_55(#-mt")
DEBUG = os.getenv("DEBUG", "True") == "True"
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")

# ---------------------------------------------------------------------
# 📦 Installed Apps
# ---------------------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'tutor',
    "rest_framework",
]

# ---------------------------------------------------------------------
# ⚙️ Middleware
# ---------------------------------------------------------------------
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],  # optional: if you have templates folder
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# ---------------------------------------------------------------------
# 🗄️ Database Configuration (Auto-switch between dev/prod)
# ---------------------------------------------------------------------

ENVIRONMENT = os.getenv("ENV", "development")  # 'development' or 'production'

if ENVIRONMENT == "production":
    # ✅ Supabase PostgreSQL setup for production
    DATABASES = {
        "default": dj_database_url.parse(
            os.getenv("DATABASE_URL"),
            conn_max_age=600,
            ssl_require=True
        )
    }

    # ⛔ Old shared MySQL setup (kept for reference)
    # DATABASES = {
    #     'default': {
    #         'ENGINE': 'django.db.backends.mysql',
    #         'NAME': 'muubiiby_ai_db',
    #         'USER': 'muubiiby_mk',
    #         'PASSWORD': 'ad+uP[QSfvNT0d8',
    #         'HOST': 'localhost',  # usually 'localhost' in cPanel
    #         'PORT': '3306',
    #         'OPTIONS': {
    #             'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
    #         },
    #     }
    # }

else:
    # ✅ Local development uses PostgreSQL (localhost)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('DBNAME', 'personalizedAItutor'),
            'USER': os.getenv('DBUSER', 'postgres'),
            'PASSWORD': os.getenv('DBPASS', ''),
            'HOST': os.getenv('DBHOST', '127.0.0.1'),
            'PORT': os.getenv('DBPORT', '5432'),
        }
    }

# ---------------------------------------------------------------------
# 🧩 File Upload Settings
# ---------------------------------------------------------------------
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt"}

# ---------------------------------------------------------------------
# 🔑 Password Validators
# ---------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

# ---------------------------------------------------------------------
# 🌍 Localization
# ---------------------------------------------------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Africa/Lagos'  # ✅ (Nigeria timezone)
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------
# 🖼️ Static and Media Files
# ---------------------------------------------------------------------
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / "media"

# ---------------------------------------------------------------------
# 🧠 REST Framework / AI Settings
# ---------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JINA_API_KEY=os.getenv("JINA_API_KEY", "")
SUPABASE_KEY=os.getenv("SUPABASE_KEY", "")
SUPABASE_URL=os.getenv("SUPABASE_URL", "")

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
