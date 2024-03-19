import banzai_floyds_ui.banzai_floyds_gui.banzai_floyds_app as banzai_floyds_app
import logging
logger = logging.getLogger(__name__)


def loader(app_name):
    if 'banzai-floyds' in app_name:
        logger.info(f"{app_name} is being loaded")
        return banzai_floyds_app.app
    else:
        return None
