from __future__ import annotations
"""
Configuration for SHL Recommender (clean, de-duplicated).
"""

import os
import re
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel, Field

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_RAW_DIR = DATA_DIR / "catalog_raw"
CATALOG_SNAPSHOT_PATH = DATA_DIR / "catalog_snapshot.parquet"
TRAIN_PATH = DATA_DIR / "gen_ai_train.xlsx"
TEST_PATH = DATA_DIR / "gen_ai_test.xlsx"

INDICES_DIR = PROJECT_ROOT / "indices"
BM25_INDEX_PATH = INDICES_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDICES_DIR / "faiss.index"
EMBEDDINGS_PATH = INDICES_DIR / "item_embeddings.npy"
IDS_MAPPING_PATH = INDICES_DIR / "ids.json"

MODELS_DIR = PROJECT_ROOT / "models"

# Models
BGE_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

HF_ENV_VARS = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TRANSFORMERS_CACHE": str(MODELS_DIR),
    "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE", "1"),
}

# Retrieval / Fusion
BM25_WEIGHT = 0.60
DENSE_WEIGHT = 0.40

# IMPORTANT: retrieval should read these values (don’t hardcode in code)
BM25_TOP_N = 300
DENSE_TOP_N = 300
FUSION_TOP_K = 200

FUSION_WINSORIZE_MIN = -3.0
FUSION_WINSORIZE_MAX = 3.0
FUSION_EPS = 1e-8

# Diversification
MMR_LAMBDA = 0.45

# Result policy
RESULT_MIN = 5
RESULT_MAX = 10
RESULT_DEFAULT_TARGET = 10  # soft target only

# Duration tolerance used by retrieval / filtering
DURATION_TOLERANCE_MIN = 15
DURATION_TOLERANCE_MAX = 120

# Rerank
DEFAULT_RERANK_CUTOFF = 120
RERANK_CUTOFF = int(os.getenv("RERANK_CUTOFF", str(DEFAULT_RERANK_CUTOFF)))

# Text processing
MAX_INPUT_CHARS = 20_000
ENCODER_TOKEN_BUDGET = 512
CHUNK_SIZE_TOKENS = 220
CHUNK_STRIDE_TOKENS = 110
CHUNK_TOP_K = 3

# Synonyms / lexical normalization
SYNONYM_MAP: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "node.js": "nodejs",
    "asp.net": "aspnet",
    "dotnet": ".net",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "pm": "project manager",
    "mgr": "manager",
    "stakeholder mgmt": "stakeholder management",
    "coo": "chief operating officer",
    "cxo": "executive",
    "springboot": "spring boot",
    "java ee": "spring boot",
    "j2ee": "spring boot",
    "microservices": "spring boot",
    "ux": "user experience",
    "ui": "user interface",
    "bi": "business intelligence",
    "erp": "enterprise resource planning",
    "sme": "subject matter expert",
    "qa": "quality assurance",
    "sdet": "software testing",
    "seo": "search engine optimization",
    "sem": "search engine marketing",
    "content writer": "content writing",
    "digital marketing": "marketing",
    "assistant admin": "administrative assistant",
    "entry-level": "entry level",
    # extra grads/admin normalisation
    "new graduates": "entry level",
    "freshers": "entry level",
}



ENABLE_ESCO_AUGMENT = False

# Keyword families / intent markers
_AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
    "neural network", "mlops", "model deployment", "data science", "data scientist",
    "predictive analytics", "computer vision", "natural language processing", "nlp",
    "generative ai", "llm", "chatbot", "transformer model",
]
_PYTHON_KEYWORDS = [
    "python", "pandas", "numpy", "scikit-learn", "sklearn", "django", "flask", "fastapi",
    "backend", "automation", "api", "scripting", "oop", "data structures",
]
_ANALYTICS_KEYWORDS = [
    "excel", "power bi", "tableau", "data visualization", "data analysis", "dashboard",
    "sql", "business analytics", "data modeling", "reporting", "sql server", "data warehouse",
    "statistics", "forecasting",
]
_TECH_KEYWORDS = [
    "developer", "engineer", "programmer", "software", "frontend", "backend", "fullstack",
    "cloud", "devops", "aws", "azure", "gcp", "linux", "docker", "kubernetes", "network",
    "cybersecurity", "qa", "testing", "automation", "integration", "ci/cd", "microservices",
] + [
    "cloud computing", "containers", "serverless", "cloud security", "terraform", "sre",
    "siem", "pentest", "incident response", "iam", "zero trust",
]

_FINANCE_KEYWORDS = [
    "finance", "accounting", "tax", "audit", "financial analysis", "valuation", "budget",
    "forecasting", "treasury", "banking", "credit risk", "financial modeling",
]
_BUSINESS_KEYWORDS = [
    "business", "operations", "project management", "strategic", "planning", "risk management",
    "supply chain", "procurement", "marketing", "branding", "kpi", "stakeholder management",
    "change management", "decision making",
]
_HR_KEYWORDS = [
    "hr", "human resources", "recruitment", "talent acquisition", "employee engagement",
    "conflict management", "leadership", "coaching", "mentoring", "organizational development",
    "assessment center", "psychometrics",
]
_SALES_KEYWORDS = [
    "sales", "retail", "customer service", "negotiation", "communication", "persuasion",
    "telemarketing", "contact center", "crm", "business development", "account executive",
    "sales representative", "technical sales", "lead generation",
]
_APTITUDE_KEYWORDS = [
    "aptitude", "logical reasoning", "numerical reasoning", "verbal reasoning", "abstract reasoning",
    "diagrammatic reasoning", "cognitive", "problem solving", "critical thinking",
    "inductive reasoning", "deductive reasoning",
]
_BEHAVIOR_KEYWORDS = [
    "leadership", "teamwork", "collaboration", "communication", "adaptability", "decision making",
    "conflict resolution", "time management", "resilience", "interpersonal", "personality",
    "culture fit", "values", "opq", "occupational personality questionnaire", "team types",
    "emotional intelligence", "engagement",
]

# triggers
BEHAVIOUR_TRIGGER_PHRASES: List[str] = [
    "consultant", "consulting", "culture fit", "cultural fit", "right fit", "values fit",
    "culturally a right fit",
    "io psychologist", "industrial psychology", "leadership role", "executive role", "c-suite",
    "coo ", "chief operating officer", "senior leadership", "people leader",
]
APTITUDE_TRIGGER_PHRASES: List[str] = [
    "aptitude", "reasoning test", "numerical test", "verbal test", "inductive", "deductive",
    "cognitive ability",
]
COMMUNICATION_TRIGGER_PHRASES: List[str] = [
    "communication skills", "strong communication", "excellent communication", "written communication",
    "verbal communication", "spoken english", "english comprehension", "business communication",
    "email writing", "presentation skills", "client communication",
]

_INTENT_KEYWORDS = {
    "technical": _TECH_KEYWORDS + _AI_KEYWORDS + _PYTHON_KEYWORDS + _ANALYTICS_KEYWORDS,
    "behavior": _BEHAVIOR_KEYWORDS + _HR_KEYWORDS + _SALES_KEYWORDS + _BUSINESS_KEYWORDS,
    "aptitude": _APTITUDE_KEYWORDS,
}

# Domain markers & seeds
_DOMAIN_MARKERS = {
    "software_dev": _TECH_KEYWORDS + _PYTHON_KEYWORDS,
    "ai_ml_data": _AI_KEYWORDS + _ANALYTICS_KEYWORDS,
    "business_finance": _BUSINESS_KEYWORDS + _FINANCE_KEYWORDS,
    "hr_leadership": _HR_KEYWORDS + _BEHAVIOR_KEYWORDS,
    "sales_service": _SALES_KEYWORDS + ["customer success", "account manager"],
    "aptitude_reasoning": _APTITUDE_KEYWORDS,
    "cloud_security": ["aws", "azure", "gcp", "kubernetes", "docker", "terraform", "siem", "sre"],
}
DOMAIN_MARKERS = _DOMAIN_MARKERS  # public alias

# ---- MAJOR retrieval boosts (extended) ----
RETRIEVAL_BOOST_SEEDS: Dict[str, List[str]] = {
    # personality / leadership staples
    "opq": ["occupational personality questionnaire", "opq32", "opq32r", "personality questionnaire", "personality assessment"],
    "leadership": ["leadership report", "enterprise leadership", "manager 8.0", "managerial role", "leadership styles"],
    # communication staples
    "communication": ["business communication", "english comprehension", "written english", "spoken english", "verbal ability", "email writing"],
    # sales staples
    "sales": ["entry level sales", "sales representative", "technical sales associate", "sales sift-out", "inside sales"],
    # data / sql / excel staples
    "sql": ["sql server", "database", "data warehouse", "ssis", "ssas", "ssrs"],
    "python": ["python developer", "data analyst", "data analytics", "machine learning", "pandas", "numpy"],
    # Consultant / I-O Psychology / people science — pull OPQ & Verify batteries
    "consultant": [
        "occupational personality questionnaire", "opq", "opq32r",
        "verify verbal ability next generation", "verify numerical ability",
        "professional 7.1 solution", "administrative professional short form",
    ],
    "industrial organizational": [
        "opq", "opq32r", "leadership report",
        "verify verbal ability next generation", "verify numerical ability",
    ],
    # QA / Testing
    "quality assurance": ["automata selenium", "selenium", "manual testing", "qa engineer", "test automation", "sql server", "javascript", "htmlcss", "css3"],
    "qa engineer": ["automata selenium", "selenium", "manual testing", "sql server", "javascript", "htmlcss", "css3"],
    # Marketing / Brand / Community / Events
    "marketing manager": ["digital advertising", "writex email writing sales", "business communication", "manager 8.0"],
    "brand": ["digital advertising", "marketing", "business communication"],
    "community": ["digital advertising", "business communication", "presentation", "email writing"],
    "events": ["digital advertising", "business communication", "presentation"],
}

# Focused expansion library for exact families
EXPANSION_LIBRARY: Dict[str, List[str]] = {
    "behavior": ["opq", "occupational personality questionnaire", "leadership report", "interpersonal communications", "team types"],
    "aptitude": ["verify verbal ability next generation", "verify numerical ability", "shl verify interactive inductive reasoning"],
    "sales_entry": ["entry level sales solution", "interpersonal communications", "business communication adaptive", "svar spoken english indian accent new"],
    "qa_testing": ["automata selenium", "selenium new", "manual testing new", "sql server new", "javascript new", "css3 new", "htmlcss new"],
    "data_analyst": ["automata sql new", "python new", "microsoft excel 365 new", "microsoft excel 365 essentials new", "tableau new", "sql server analysis services (ssas) (new)"],
    "java_dev": ["core java entry level new", "core java advanced level new", "java 8 new", "interpersonal communications"],
    "content_marketing": ["search engine optimization new", "written english v1", "english comprehension new", "digital advertising new"],
    "admin_ops": ["administrative professional short form", "bank administrative assistant short form", "general entry level data entry 7.0 solution", "verify numerical ability", "basic computer literacy windows 10 new"],
    "consultant_io": ["opq32r", "leadership report", "verify verbal ability next generation", "verify numerical ability", "professional 7.1 solution"],
    "marketing_mgr": ["digital advertising", "writex email writing sales", "business communication adaptive", "manager 8.0 jfa 4310"],
}

# SHL tokens
_SHL_KEYWORDS = [
    "assessment", "solution", "verify", "professional", "short form",
    "entry level", "adaptive", "manager", "leadership", "7.0", "7.1", "8.0",
    "automata", "technical checking", "communication", "opq", "biodata",
    "motivation questionnaire", "competency", "simulation", "situational judgment", "aptitude test",
]

SYNONYM_MAP.update({
    "assistant admin": "administrative assistant",
    "admin assistant": "administrative assistant",
    "business teams": "stakeholder management",
    "collaboration": "collaborate",
    "culturally": "culture fit",
})

BEHAVIOUR_TRIGGER_PHRASES += [
    "collaborate", "collaboration", "business teams", "culturally", "brand positioning",
    "community", "events", "stakeholder", "presentation", "interpersonal",
]

COMMUNICATION_TRIGGER_PHRASES += [
    "business teams", "stakeholder", "client-facing", "presentation", "storytelling",
]

RETRIEVAL_BOOST_SEEDS.update({
    # Content writer / SEO (make sure Drupal can surface)
    "content_marketing": [
        "search engine optimization", "written english", "english comprehension",
        "writex email writing", "drupal",
    ],
    # Admin / banking ops
    "admin_ops": [
        "administrative professional short form", "bank administrative assistant short form",
        "general entry level data entry 7.0 solution", "basic computer literacy windows 10",
        "verify numerical ability",
    ],
    # Sales entry-level guardrails
    "sales_entry": [
        "entry level sales solution", "entry level sales sift-out 7.1",
        "sales representative solution", "interpersonal communications",
        "business communication adaptive", "svar spoken english indian accent",
    ],
    # Marketing manager
    "marketing manager": [
        "digital advertising", "writex email writing sales",
        "business communication adaptive", "manager 8.0",
    ],
})

EXPANSION_LIBRARY.update({
    "sales_entry": [
        "entry level sales 7.1", "entry-level sales sift-out 7.1",
        "sales representative solution", "technical sales associate solution",
        "interpersonal communications", "business communication adaptive",
        "svar spoken english indian accent (new)",
    ],
    "consultant": [
        "opq32r", "opq leadership report", "verify verbal ability next generation",
        "verify numerical ability", "professional 7.1 solution",
        "administrative professional short form",
    ],
    "marketing": ["digital advertising", "marketing", "business communication", "manager 8.0"],
})


_TECH_KEYWORDS += _SHL_KEYWORDS
_DOMAIN_MARKERS["shl_general"] = _SHL_KEYWORDS

# Slug canonicalisation
_SLUG_FAMILY_PATTERNS = [
    (re.compile(r"-new$"), ""),
    (re.compile(r"\(new\)$"), ""),
    (re.compile(r"-v\d+$"), ""),
    (re.compile(r"\(\s*v\d+\s*\)$"), ""),
    (re.compile(r"-\d+\.\d+$"), ""),
]
def family_slug(slug: str) -> str:
    if not isinstance(slug, str):
        return ""
    s = slug.strip().lower().strip("/")
    s = s.replace("%28", "(").replace("%29", ")").replace("_", "-")
    s = re.sub(r"-+", "-", s)
    for pat, repl in _SLUG_FAMILY_PATTERNS:
        s = pat.sub(repl, s).rstrip("-").strip()
    return s

# HTTP hardening
HTTP_CONNECT_TIMEOUT = 3.0
HTTP_READ_TIMEOUT = 7.0
HTTP_MAX_REDIRECTS = 2
HTTP_MAX_BYTES = 1_000_000
HTTP_USER_AGENT = "shl-rag-recommender/1.0 (+https://shl.com; contact=genai@placeholder.com)"

# Zero-shot / balance
INTENT_TEMP = 1.5
INTENT_SMOOTH_EPS = 0.15
INTENT_CLIP_MIN = 0.20
INTENT_CLIP_MAX = 0.80
INTENT_LABEL_TECHNICAL = "technical skills / knowledge"
INTENT_LABEL_PERSONALITY = "personality / behavior"
INTENT_LABELS: List[str] = [INTENT_LABEL_TECHNICAL, INTENT_LABEL_PERSONALITY]
INTENT_SOFT_BONUS = 0.06

# Crawl / balance constants
MAX_CATALOG_PAGES = 100
HTTP_RETRY_READS = 1
BOTH_HIGH_THRESHOLD = 0.45
DOMINANT_THRESHOLD = 0.60
SECONDARY_MIN_FOR_SPLIT = 0.30
BALANCE_5_5_SIZE = 10
BALANCE_7_3_SIZE = 10

# Logging dir
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Pydantic schemas
class AssessmentItem(BaseModel):
    url: str
    name: str
    description: str
    duration: int = Field(ge=0)
    adaptive_support: str
    remote_support: str
    test_type: List[str]

    def ensure_flags_are_literal(self) -> None:
        self.adaptive_support = "Yes" if self.adaptive_support == "Yes" else "No"
        self.remote_support = "Yes" if self.remote_support == "Yes" else "No"


class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]


class HealthResponse(BaseModel):
    status: str

