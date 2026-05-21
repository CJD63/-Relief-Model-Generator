"""Session state initialization for the 3D Relief Model Generator.

All st.session_state defaults are set here so the main app stays concise.
"""

import streamlit as st


def init_session_state():
    """Initialize all Streamlit session state variables with their defaults.

    Safe to call on every rerun — only sets values if they don't already exist.
    """
    # Depth preview & STL output
    if 'depth_image' not in st.session_state:
        st.session_state.depth_image = None
    if 'depth_info' not in st.session_state:
        st.session_state.depth_info = {}
    if 'stl_bytes' not in st.session_state:
        st.session_state.stl_bytes = None
    if 'preview_done' not in st.session_state:
        st.session_state.preview_done = False

    # Status
    if 'status_text' not in st.session_state:
        st.session_state.status_text = 'Waiting for input…'
    if 'status_kind' not in st.session_state:
        st.session_state.status_kind = 'info'

    # Last-generation parameters (for cache invalidation)
    if 'last_params' not in st.session_state:
        st.session_state.last_params = {}

    # Batch processing
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'batch_in_progress' not in st.session_state:
        st.session_state.batch_in_progress = False

    # Image source
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'keep_original_colors' not in st.session_state:
        st.session_state.keep_original_colors = False

    # Image adjustments
    if 'brightness' not in st.session_state:
        st.session_state.brightness = 0
    if 'contrast' not in st.session_state:
        st.session_state.contrast = 0
    if 'saturation' not in st.session_state:
        st.session_state.saturation = 0
    if 'sharpness' not in st.session_state:
        st.session_state.sharpness = 0

    # Adjustment presets
    if 'adjustment_presets' not in st.session_state:
        st.session_state.adjustment_presets = {
            'default': {
                'brightness': 0, 'contrast': 0,
                'saturation': 0, 'sharpness': 0,
            }
        }
    if 'current_preset_name' not in st.session_state:
        st.session_state.current_preset_name = 'default'

    # View controls
    if 'zoom_level' not in st.session_state:
        st.session_state.zoom_level = 1.0
    if 'pan_x' not in st.session_state:
        st.session_state.pan_x = 0
    if 'pan_y' not in st.session_state:
        st.session_state.pan_y = 0
    if 'rotation' not in st.session_state:
        st.session_state.rotation = 0

    # Crop
    if 'crop_enabled' not in st.session_state:
        st.session_state.crop_enabled = False
    if 'crop_top' not in st.session_state:
        st.session_state.crop_top = 0
    if 'crop_bottom' not in st.session_state:
        st.session_state.crop_bottom = 100
    if 'crop_left' not in st.session_state:
        st.session_state.crop_left = 0
    if 'crop_right' not in st.session_state:
        st.session_state.crop_right = 100

    # Comparison mode
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = 'slider'

    # Batch presets
    if 'batch_presets' not in st.session_state:
        st.session_state.batch_presets = {
            'default': {
                'preset': 'preview',
                'keep_colors': False,
                'invert_colors': False,
                'edge_detection': False,
                'edge_strength': 1.0,
                'use_ai_depth': False,
                'ai_model_name': 'depth-anything/Depth-Anything-V2-Small-hf',
                'model_width': 50.0,
                'model_thickness': 5.0,
                'base_thickness': 2.0,
                'filename_pattern': '{name}_{preset}',
            }
        }
    if 'current_batch_preset_name' not in st.session_state:
        st.session_state.current_batch_preset_name = 'default'

    # AI depth (single-image tab)
    if 'use_ai_depth' not in st.session_state:
        st.session_state.use_ai_depth = False

    # AI depth (batch tab)
    if 'batch_use_ai_depth' not in st.session_state:
        st.session_state.batch_use_ai_depth = False
    if 'batch_ai_model_name' not in st.session_state:
        st.session_state.batch_ai_model_name = 'depth-anything/Depth-Anything-V2-Small-hf'

    # Comparison images
    if 'comparison_grayscale' not in st.session_state:
        st.session_state.comparison_grayscale = None
    if 'comparison_ai' not in st.session_state:
        st.session_state.comparison_ai = None

    # Debug mode
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    # Model warmup
    if 'auto_warmup_model' not in st.session_state:
        st.session_state.auto_warmup_model = False
    if 'model_warmed_up' not in st.session_state:
        st.session_state.model_warmed_up = False
