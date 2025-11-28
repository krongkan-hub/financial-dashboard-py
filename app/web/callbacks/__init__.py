from .navigation import register_navigation_callbacks
from .user_data import register_user_data_callbacks
from .modals import register_modal_callbacks
from .graphs import register_graph_callbacks
from .tables import register_table_callbacks

def register_callbacks(app, METRIC_DEFINITIONS):
    """
    Registers all callbacks for the application.
    METRIC_DEFINITIONS is accepted for compatibility but used directly from layout in sub-modules.
    """
    register_navigation_callbacks(app)
    register_user_data_callbacks(app)
    register_modal_callbacks(app)
    register_graph_callbacks(app)
    register_table_callbacks(app)
