import os


def _get_secret(key: str, default: str = "") -> str:
    """Retrieve a secret from Streamlit Cloud secrets or local .env."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass

    # Fall back to environment variables / .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return os.getenv(key, default)


ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
FMP_API_KEY = _get_secret("FMP_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# Fixed framework categories for comparison
FIXED_CATEGORIES = [
    "Revenue & Growth",
    "Margins & Profitability",
    "Forward Guidance",
    "Competitive Positioning",
    "Capital Allocation",
    "Risk Factors",
]
