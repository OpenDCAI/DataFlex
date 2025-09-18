import logging
import sys
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(handler)