import banzai_floyds_ui.gui.app
import logging
logger = logging.getLogger(__name__)


def loader(app_name):
    if 'banzai-floyds' in app_name:
        logger.info(f"{app_name} is being loaded")
        return banzai_floyds_ui.gui.app.app
    else:
        return None
