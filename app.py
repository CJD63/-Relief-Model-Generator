import streamlit as st
import tempfile
import os
import shutil
import zipfile
import requests
import time
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from relief import ReliefGenerator

# Page config
st.set_page_config(
    page_title='3D Relief Model Generator',
    page_icon='🎭',
    layout='wide',
)

# Custom CSS
st.markdown('''
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { margin: 0 0 0.5rem 0; font-size: 2.2rem; }
    .main-header p  { margin: 0; opacity: 0.85; font-size: 1rem; }

    .section-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .section-card h4 { margin: 0 0 0.75rem 0; color: #495057; font-size: 1rem; }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .badge-blue   { background: #dbeafe; color: #1d4ed8; }
    .badge-green  { background: #dcfce7; color: #15803d; }
    .badge-purple { background: #ede9fe; color: #6d28d9; }
    .badge-orange { background: #ffedd5; color: #c2410c; }

    .status-box {
        background: #1e293b;
        color: #94a3b8;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-family: monospace;
        font-size: 0.85rem;
        min-height: 80px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .status-box.success { color: #4ade80; }
    .status-box.error   { color: #f87171; }

    .step-row {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 1.25rem;
    }
    .step-bubble {
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem;
        flex-shrink: 0;
    }
    .step-bubble.active   { background: #3b82f6; color: white; }
    .step-bubble.done     { background: #22c55e; color: white; }
    .step-bubble.inactive { background: #e5e7eb; color: #9ca3af; }
    .step-label { font-size: 0.88rem; color: #374151; }

    .pill-list { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }

    .thin-divider { border-top: 1px solid #e5e7eb; margin: 1rem 0; }

    .download-area {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 2px dashed #86efac;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .batch-progress {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .batch-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .batch-item:last-child { border-bottom: none; }
    .batch-item.success { background: #dcfce7; }
    .batch-item.failed { background: #fee2e2; }
    
    .toggle-switch {
        position: relative;
        width: 50px;
        height: 26px;
    }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .toggle-slider {
        position: absolute;
        cursor: pointer;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: #ccc;
        transition: .3s;
        border-radius: 26px;
    }
    .toggle-slider:before {
        position: absolute;
        content: '';
        height: 20px;
        width: 20px;
        left: 3px;
        bottom: 3px;
        background-color: white;
        transition: .3s;
        border-radius: 50%;
    }
    input:checked + .toggle-slider { background-color: #3b82f6; }
    input:checked + .toggle-slider:before { transform: translateX(24px); }
</style>
''', unsafe_allow_html=True)


# Helpers
def download_image_from_url(url: str, keep_colors=False):
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        image = Image.open(BytesIO(r.content))
        # Convert to grayscale only if not keeping original colors
        if not keep_colors and image.mode != 'L':
            image = image.convert('L')
        return image, None
    except requests.exceptions.Timeout:
        return None, 'Timeout error: The request took too long. Please check the URL.'
    except requests.exceptions.HTTPError as e:
        return None, 'HTTP error ' + str(e.response.status_code) + ': Could not download image.'
    except requests.exceptions.RequestException as e:
        return None, 'Network error: ' + str(e)
    except IOError as e:
        return None, 'Invalid image format: ' + str(e)
    except Exception as e:
        return None, 'Unexpected error: ' + str(e)


def get_image(uploaded_file, url_input: str, keep_colors=False):
    if url_input and url_input.strip():
        img, err = download_image_from_url(url_input.strip(), keep_colors)
        return img, err
    elif uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image.load()
            # Convert to grayscale only if not keeping original colors
            if not keep_colors and image.mode != 'L':
                image = image.convert('L')
            return image, None
        except IOError as e:
            return None, 'Invalid or corrupted image file: ' + str(e)
        except Exception as e:
            return None, 'Error loading image: ' + str(e)
    return None, 'Please upload an image or enter a URL.'


def apply_image_adjustments(image, brightness=0, contrast=0, saturation=0, sharpness=0):
    """Apply brightness, contrast, saturation, and sharpness adjustments to an image.
    
    Args:
        image: PIL Image
        brightness: -100 to 100 (0 = no change)
        contrast: -100 to 100 (0 = no change)
        saturation: -100 to 100 (0 = no change)
        sharpness: -100 to 100 (0 = no change)
    
    Returns:
        Adjusted PIL Image
    """
    from PIL import ImageEnhance
    
    # Apply brightness
    if brightness != 0:
        factor = 1.0 + (brightness / 100.0)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(max(0.0, factor))
    
    # Apply contrast
    if contrast != 0:
        factor = 1.0 + (contrast / 100.0)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(max(0.0, factor))
    
    # Apply saturation
    if saturation != 0:
        factor = 1.0 + (saturation / 100.0)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(max(0.0, factor))
    
    # Apply sharpness
    if sharpness != 0:
        factor = 1.0 + (sharpness / 100.0)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(max(0.0, factor))
    
    return image


def status_html(text: str, kind: str = 'info') -> str:
    css_class = {'info': '', 'success': ' success', 'error': ' error'}.get(kind, '')
    return '<div class=\"status-box' + css_class + '\">' + text + '</div>'


def process_single_image(image, preset, enable_edge_detection, edge_strength, invert_colors, model_params, progress_callback=None):
    generator = ReliefGenerator()
    generator.load_image_from_pil(image, max_size=preset['max_size'])
    generator.create_depth_map(
        invert_depth=invert_colors,
        blur_radius=preset['blur'],
        gamma=preset['gamma'],
        enable_hill_removal=True,
        hill_removal_strength=preset['hill'],
        enable_edge_detection=enable_edge_detection,
        edge_strength=edge_strength,
        progress_callback=progress_callback
    )
    generator.create_relief_mesh(
        model_width=model_params['width'],
        model_thickness=model_params['thickness'],
        base_thickness=model_params['base']
    )
    return generator


# Session state
if 'depth_image' not in st.session_state:
    st.session_state.depth_image = None
if 'depth_info' not in st.session_state:
    st.session_state.depth_info = {}
if 'stl_bytes' not in st.session_state:
    st.session_state.stl_bytes = None
if 'preview_done' not in st.session_state:
    st.session_state.preview_done = False
if 'status_text' not in st.session_state:
    st.session_state.status_text = 'Waiting for input…'
if 'status_kind' not in st.session_state:
    st.session_state.status_kind = 'info'
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'batch_in_progress' not in st.session_state:
    st.session_state.batch_in_progress = False
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'keep_original_colors' not in st.session_state:
    st.session_state.keep_original_colors = False
if 'brightness' not in st.session_state:
    st.session_state.brightness = 0
if 'contrast' not in st.session_state:
    st.session_state.contrast = 0
if 'saturation' not in st.session_state:
    st.session_state.saturation = 0
if 'sharpness' not in st.session_state:
    st.session_state.sharpness = 0
if 'adjustment_presets' not in st.session_state:
    st.session_state.adjustment_presets = {
        'default': {'brightness': 0, 'contrast': 0, 'saturation': 0, 'sharpness': 0}
    }
if 'current_preset_name' not in st.session_state:
    st.session_state.current_preset_name = 'default'
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 1.0
if 'pan_x' not in st.session_state:
    st.session_state.pan_x = 0
if 'pan_y' not in st.session_state:
    st.session_state.pan_y = 0
if 'rotation' not in st.session_state:
    st.session_state.rotation = 0
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
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = 'slider'
if 'batch_presets' not in st.session_state:
    st.session_state.batch_presets = {
        'default': {
            'preset': 'preview',
            'keep_colors': False,
            'invert_colors': False,
            'edge_detection': False,
            'edge_strength': 1.0,
            'model_width': 50.0,
            'model_thickness': 5.0,
            'base_thickness': 2.0,
            'filename_pattern': '{name}_{preset}'
        }
    }
if 'current_batch_preset_name' not in st.session_state:
    st.session_state.current_batch_preset_name = 'default'


# HEADER
st.markdown('''
<div class=\"main-header\">
  <h1>🎭 3D Relief Model Generator</h1>
  <p>
    Convert any image into a printable 3D relief STL model using professional
    depth-map algorithms — edge-preserving bilateral filtering, non-local means
    denoising, guided filtering, and morphological smoothing.
  </p>
</div>
''', unsafe_allow_html=True)


# Tabs
tab_single, tab_batch = st.tabs(['📤 Single Image', '📚 Batch Process'])

# SINGLE IMAGE TAB
with tab_single:
    left_col, right_col = st.columns([1, 1], gap='large')
    
    with left_col:
        step1_class = 'done' if st.session_state.preview_done else 'active'
        step2_class = 'active' if st.session_state.preview_done else 'inactive'
        step3_class = 'done' if st.session_state.stl_bytes else 'inactive'
        
        st.markdown('''
        <div class=\"step-row\">
          <div class=\"step-bubble ''' + step1_class + '''\">1</div>
          <div class=\"step-label\">Upload image &amp; generate depth preview</div>
        </div>
        <div class=\"step-row\">
          <div class=\"step-bubble ''' + step2_class + '''\">2</div>
          <div class=\"step-label\">Generate STL model</div>
        </div>
        <div class=\"step-row\">
          <div class=\"step-bubble ''' + step3_class + '''\">3</div>
          <div class=\"step-label\">Download &amp; 3D print</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class=\"thin-divider\"></div>', unsafe_allow_html=True)
        
        st.markdown('#### 📷 Image Input')
        uploaded_file = st.file_uploader(
            'Upload an image',
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'],
            help='JPG, PNG, BMP, GIF, WebP',
        )
        st.markdown('**— or —**')
        url_input = st.text_input(
            'Image URL',
            placeholder='https://example.com/image.jpg',
            help='Direct link to an image file',
        )
        
        # Store image when uploaded or URL changes
        if uploaded_file is not None or (url_input and url_input.strip()):
            # Clear adjustments when new image is uploaded
            if 'last_upload_name' not in st.session_state or st.session_state.last_upload_name != (uploaded_file.name if uploaded_file else url_input):
                st.session_state.last_upload_name = uploaded_file.name if uploaded_file else url_input
                st.session_state.brightness = 0
                st.session_state.contrast = 0
                st.session_state.saturation = 0
                st.session_state.sharpness = 0
        
        st.markdown('<div class=\"thin-divider\"></div>', unsafe_allow_html=True)
        
        st.markdown('#### 🎚️ Quality Preset')
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button('⚡ Draft', use_container_width=True):
                st.session_state.preset = 'draft'
        with preset_col2:
            if st.button('👁️ Preview', use_container_width=True):
                st.session_state.preset = 'preview'
        with preset_col3:
            if st.button('✨ High', use_container_width=True):
                st.session_state.preset = 'high'
        
        if 'preset' not in st.session_state:
            st.session_state.preset = 'preview'
        
        preset_info = {
            'draft': {'max_size': 150, 'blur': 0.3, 'gamma': 1.0, 'smoothing': True, 'hill': 3.0},
            'preview': {'max_size': 320, 'blur': 0.5, 'gamma': 1.1, 'smoothing': True, 'hill': 5.0},
            'high': {'max_size': 600, 'blur': 0.8, 'gamma': 1.2, 'smoothing': True, 'hill': 10.0},
        }
        
        for preset_name, values in preset_info.items():
            max_sz = values['max_size']
            if not (50 <= max_sz <= 2000):
                st.warning('Preset {} max_size={} is outside safe range (50-2000). Clamping.'.format(preset_name, max_sz))
                values['max_size'] = max(50, min(2000, max_sz))
            gamma_val = values['gamma']
            if not (0.1 <= gamma_val <= 4.0):
                st.warning('Preset {} gamma={} is outside safe range (0.1-4.0). Clamping.'.format(preset_name, gamma_val))
                values['gamma'] = max(0.1, min(4.0, gamma_val))
            blur_val = values['blur']
            if not (0 <= blur_val <= 20):
                st.warning('Preset {} blur={} is outside safe range (0-20). Clamping.'.format(preset_name, blur_val))
                values['blur'] = max(0, min(20, blur_val))
            hill_val = values['hill']
            if not (0.1 <= hill_val <= 20):
                st.warning('Preset {} hill={} is outside safe range (0.1-20). Clamping.'.format(preset_name, hill_val))
                values['hill'] = max(0.1, min(20, hill_val))
        
        st.markdown('<div class=\"thin-divider\"></div>', unsafe_allow_html=True)
        
        # Image Adjustment Section
        st.markdown('#### 🎨 Image Adjustments')
        st.caption('Adjust brightness, contrast, saturation, and sharpness before generating the relief model')
        
        # Keep colors option
        keep_colors = st.checkbox('🎨 Keep Original Colors', value=st.session_state.keep_original_colors,
            help='Keep the original image colors instead of converting to grayscale')
        st.session_state.keep_original_colors = keep_colors
        
        # Preset profiles
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns([2, 1, 1, 1])
        with preset_col1:
            preset_options = list(st.session_state.adjustment_presets.keys())
            selected_preset = st.selectbox('📁 Preset', preset_options, 
                index=preset_options.index(st.session_state.current_preset_name) if st.session_state.current_preset_name in preset_options else 0,
                key='adjustment_preset_selector')
            if selected_preset != st.session_state.current_preset_name:
                st.session_state.current_preset_name = selected_preset
                p = st.session_state.adjustment_presets[selected_preset]
                st.session_state.brightness = p.get('brightness', 0)
                st.session_state.contrast = p.get('contrast', 0)
                st.session_state.saturation = p.get('saturation', 0)
                st.session_state.sharpness = p.get('sharpness', 0)
                st.rerun()
        with preset_col2:
            save_preset_name = st.text_input('Save as', value='', placeholder='preset name', key='save_preset_name')
        with preset_col3:
            if st.button('💾 Save', key='save_preset_btn', disabled=not save_preset_name.strip()):
                name = save_preset_name.strip()
                st.session_state.adjustment_presets[name] = {
                    'brightness': st.session_state.brightness,
                    'contrast': st.session_state.contrast,
                    'saturation': st.session_state.saturation,
                    'sharpness': st.session_state.sharpness
                }
                st.session_state.current_preset_name = name
                st.session_state.save_preset_name = ''
                st.rerun()
        with preset_col4:
            delete_preset_name = st.selectbox('🗑️ Delete', ['(none)'] + [p for p in st.session_state.adjustment_presets.keys() if p != 'default'], key='delete_preset_selector')
            if delete_preset_name != '(none)':
                if st.button('🗑️', key='delete_preset_btn'):
                    del st.session_state.adjustment_presets[delete_preset_name]
                    if st.session_state.current_preset_name == delete_preset_name:
                        st.session_state.current_preset_name = 'default'
                    st.rerun()
        
        adj_col1, adj_col2, adj_col3, adj_col4 = st.columns(4)
        with adj_col1:
            brightness = st.slider(
                '☀️ Brightness',
                min_value=-100,
                max_value=100,
                value=st.session_state.brightness,
                step=5,
                help='Adjust image brightness (-100 to +100)'
            )
            st.session_state.brightness = brightness
        
        with adj_col2:
            contrast = st.slider(
                '🔆 Contrast',
                min_value=-100,
                max_value=100,
                value=st.session_state.contrast,
                step=5,
                help='Adjust image contrast (-100 to +100)'
            )
            st.session_state.contrast = contrast
        
        with adj_col3:
            saturation = st.slider(
                '🎨 Saturation',
                min_value=-100,
                max_value=100,
                value=st.session_state.saturation,
                step=5,
                help='Adjust color saturation (-100 to +100)'
            )
            st.session_state.saturation = saturation
        
        with adj_col4:
            sharpness = st.slider(
                '🔪 Sharpness',
                min_value=-100,
                max_value=100,
                value=st.session_state.sharpness,
                step=5,
                help='Adjust image sharpness (-100 to +100)'
            )
            st.session_state.sharpness = sharpness
        
        # Show adjusted image preview
        load_button_key = 'load_preview_btn'
        
        if uploaded_file is not None or (url_input and url_input.strip()):
            preview_col1, preview_col2 = st.columns(2)
            
            try:
                # Load the original image - read bytes directly from uploaded file to avoid stream issues
                if uploaded_file is not None:
                    raw_image = Image.open(uploaded_file)
                    raw_image.load()
                    # Reset file pointer for potential reuse
                    uploaded_file.seek(0)
                else:
                    raw_image, err = download_image_from_url(url_input.strip())
                    if err:
                        raw_image = None
                
                if raw_image:
                    # Convert to grayscale only if not keeping original colors
                    if not keep_colors and raw_image.mode != 'L':
                        raw_image = raw_image.convert('L')
                    st.session_state.original_image = raw_image
                    
                    # Reset zoom/pan when new image is loaded
                    st.session_state.zoom_level = 1.0
                    st.session_state.pan_x = 0
                    st.session_state.pan_y = 0
                    st.session_state.rotation = 0
                    st.session_state.crop_enabled = False
                    st.session_state.crop_top = 0
                    st.session_state.crop_bottom = 100
                    st.session_state.crop_left = 0
                    st.session_state.crop_right = 100
                    
                    # Apply adjustments for preview
                    adjusted_image = apply_image_adjustments(raw_image.copy(), brightness, contrast, saturation, sharpness)
                    
                    # Rotation controls
                    st.markdown('**🔄 Rotation:**')
                    rot_col1, rot_col2, rot_col3, rot_col4, rot_col5 = st.columns([1, 1, 1, 1, 3])
                    with rot_col1:
                        if st.button('↺ 90°', key='rot_90_ccw', help='Rotate 90° counter-clockwise'):
                            st.session_state.rotation = (st.session_state.rotation - 90) % 360
                    with rot_col2:
                        if st.button('↻ 90°', key='rot_90_cw', help='Rotate 90° clockwise'):
                            st.session_state.rotation = (st.session_state.rotation + 90) % 360
                    with rot_col3:
                        if st.button('⟲ 180°', key='rot_180', help='Rotate 180°'):
                            st.session_state.rotation = (st.session_state.rotation + 180) % 360
                    with rot_col4:
                        if st.button('⟳ Reset', key='rot_reset', help='Reset rotation'):
                            st.session_state.rotation = 0
                    
                    # Apply rotation if needed
                    if st.session_state.rotation != 0:
                        raw_display_img = raw_image.rotate(st.session_state.rotation, expand=True)
                        adjusted_display_img = adjusted_image.rotate(st.session_state.rotation, expand=True)
                    else:
                        raw_display_img = raw_image
                        adjusted_display_img = adjusted_image
                    
                    # Crop controls
                    st.markdown('**✂️ Crop:**')
                    crop_toggle = st.checkbox('Enable Crop', value=st.session_state.crop_enabled, key='crop_enable')
                    st.session_state.crop_enabled = crop_toggle
                    
                    if crop_toggle:
                        crop_col1, crop_col2, crop_col3, crop_col4 = st.columns(4)
                        with crop_col1:
                            crop_top = st.slider('Top %', 0, 50, st.session_state.crop_top, key='crop_top_slider')
                            st.session_state.crop_top = crop_top
                        with crop_col2:
                            crop_bottom = st.slider('Bottom %', 50, 100, st.session_state.crop_bottom, key='crop_bottom_slider')
                            st.session_state.crop_bottom = crop_bottom
                        with crop_col3:
                            crop_left = st.slider('Left %', 0, 50, st.session_state.crop_left, key='crop_left_slider')
                            st.session_state.crop_left = crop_left
                        with crop_col4:
                            crop_right = st.slider('Right %', 50, 100, st.session_state.crop_right, key='crop_right_slider')
                            st.session_state.crop_right = crop_right
                        
                        # Apply crop
                        w, h = raw_display_img.size
                        top_px = int(h * st.session_state.crop_top / 100)
                        bottom_px = int(h * st.session_state.crop_bottom / 100)
                        left_px = int(w * st.session_state.crop_left / 100)
                        right_px = int(w * st.session_state.crop_right / 100)
                        
                        if bottom_px > top_px and right_px > left_px:
                            raw_display_img = raw_display_img.crop((left_px, top_px, right_px, bottom_px))
                            adjusted_display_img = adjusted_display_img.crop((left_px, top_px, right_px, bottom_px))
                    
                    # Comparison mode
                    st.markdown('**📊 Comparison Mode:**')
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        comp_mode = st.radio('View Mode', ['slider', 'side-by-side', 'overlay'], 
                            index=['slider', 'side-by-side', 'overlay'].index(st.session_state.comparison_mode) if st.session_state.comparison_mode in ['slider', 'side-by-side', 'overlay'] else 0,
                            key='comp_mode_radio', horizontal=True)
                        st.session_state.comparison_mode = comp_mode
                    
                    # Apply crop if enabled (before rotation for correct dimensions)
                    if st.session_state.crop_enabled:
                        w, h = raw_image.size
                        top_px = int(h * st.session_state.crop_top / 100)
                        bottom_px = int(h * st.session_state.crop_bottom / 100)
                        left_px = int(w * st.session_state.crop_left / 100)
                        right_px = int(w * st.session_state.crop_right / 100)
                        if bottom_px > top_px and right_px > left_px:
                            raw_image = raw_image.crop((left_px, top_px, right_px, bottom_px))
                    
                    # Apply rotation if needed
                    if st.session_state.rotation != 0:
                        raw_image = raw_image.rotate(st.session_state.rotation, expand=True)
                    
                    # Zoom/pan controls
                    zoom_col1, zoom_col2, zoom_col3, zoom_col4 = st.columns(4)
                    with zoom_col1:
                        if st.button('🔍-', key='zoom_out'):
                            st.session_state.zoom_level = max(0.5, st.session_state.zoom_level - 0.25)
                    with zoom_col2:
                        zoom_level = st.slider('Zoom', 0.5, 4.0, st.session_state.zoom_level, 0.25, key='zoom_slider')
                        st.session_state.zoom_level = zoom_level
                    with zoom_col3:
                        if st.button('🔍+', key='zoom_in'):
                            st.session_state.zoom_level = min(4.0, st.session_state.zoom_level + 0.25)
                    with zoom_col4:
                        if st.button('↩️ Reset View', key='reset_view'):
                            st.session_state.zoom_level = 1.0
                            st.session_state.pan_x = 0
                            st.session_state.pan_y = 0
                    
                    # Apply zoom by resizing image
                    if st.session_state.zoom_level != 1.0:
                        w, h = raw_image.size
                        new_w, new_h = int(w * st.session_state.zoom_level), int(h * st.session_state.zoom_level)
                        raw_final = raw_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        adjusted_final = adjusted_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    else:
                        raw_final = raw_image
                        adjusted_final = adjusted_image
                    
                    # Display based on comparison mode
                    if st.session_state.comparison_mode == 'side-by-side':
                        with preview_col1:
                            st.markdown('**Original:**')
                            st.image(raw_final, use_container_width=True)
                        with preview_col2:
                            st.markdown('**Adjusted Preview:**')
                            st.image(adjusted_final, use_container_width=True)
                    elif st.session_state.comparison_mode == 'slider':
                        st.markdown('**Before / After Comparison:**')
                        # Create comparison with toggle
                        slider_pos = st.slider('Drag to compare', min_value=0, max_value=100, value=50, key='comparison_slider')
                        if slider_pos < 50:
                            st.image(raw_final, use_container_width=True)
                            st.caption('◀ Original')
                        else:
                            st.image(adjusted_final, use_container_width=True)
                            st.caption('Adjusted ▶')
                    else:  # overlay mode
                        st.markdown('**Overlay Comparison:**')
                        col_overlay1, col_overlay2 = st.columns(2)
                        with col_overlay1:
                            st.image(raw_final, caption='Original', use_container_width=True)
                        with col_overlay2:
                            st.image(adjusted_final, caption='Adjusted', use_container_width=True)
                        # Show blended overlay
                        arr1 = np.array(raw_final.convert('RGB'))
                        arr2 = np.array(adjusted_final.convert('RGB'))
                        if arr1.shape == arr2.shape:
                            blended = (arr1.astype(float) * 0.5 + arr2.astype(float) * 0.5).astype(np.uint8)
                            blended_img = Image.fromarray(blended)
                            st.markdown('**Blended (50% overlay):**')
                            st.image(blended_img, use_container_width=True)
                    
                    # Show reset button if adjustments were made
                    if brightness != 0 or contrast != 0 or saturation != 0 or sharpness != 0:
                        if st.button('🔄 Reset Adjustments', key='reset_adjustments'):
                            st.session_state.brightness = 0
                            st.session_state.contrast = 0
                            st.session_state.saturation = 0
                            st.session_state.sharpness = 0
                            st.rerun()
            except Exception as e:
                st.warning('Could not load image preview: ' + str(e))
        
        st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
        
        with st.expander('⚙️ Advanced Settings'):
            invert_colors = st.checkbox('🔄 Invert Colors (Black ↔ White)', value=False,
                help='Swap black and white. Use this if your image has dark background and light foreground, or vice versa.')
            
            enable_edge_detection = st.checkbox('🔍 Enable Edge Detection', value=False, 
                help='Enhance edges for crisper relief features in the 3D model')
            edge_strength = st.slider('Edge Strength', 0.1, 2.0, 1.0, 0.1,
                help='Higher values create more pronounced edges') if enable_edge_detection else 1.0
            
            st.markdown('#### 📐 Model Dimensions')
            model_width = st.slider('Model Width (mm)', 10.0, 200.0, 50.0, 5.0)
            model_thickness = st.slider('Relief Height (mm)', 1.0, 20.0, 5.0, 0.5)
            base_thickness = st.slider('Base Thickness (mm)', 0.5, 10.0, 2.0, 0.5)
        
        st.markdown('<div class=\"thin-divider\"></div>', unsafe_allow_html=True)
        
        preset = preset_info.get(st.session_state.preset, preset_info['preview'])
        
        generate_col1, generate_col2 = st.columns([2, 1])
        with generate_col1:
            if st.button('🚀 Generate Relief Model', use_container_width=True, type='primary'):
                if uploaded_file is None and not url_input.strip():
                    st.error('Please upload an image or enter a URL')
                else:
                    with st.spinner('Processing image...'):
                        try:
                            # Use the stored original image if available (avoids re-reading file)
                            # Note: original_image already has rotation and crop applied in preview section
                            if st.session_state.original_image is not None:
                                image = st.session_state.original_image.copy()
                            else:
                                image, err = get_image(uploaded_file, url_input)
                                if err:
                                    st.error(err)
                                    st.stop()
                                # Apply rotation and crop if needed for fallback path
                                if st.session_state.rotation != 0:
                                    image = image.rotate(st.session_state.rotation, expand=True)
                                if st.session_state.crop_enabled:
                                    w, h = image.size
                                    top_px = int(h * st.session_state.crop_top / 100)
                                    bottom_px = int(h * st.session_state.crop_bottom / 100)
                                    left_px = int(w * st.session_state.crop_left / 100)
                                    right_px = int(w * st.session_state.crop_right / 100)
                                    if bottom_px > top_px and right_px > left_px:
                                        image = image.crop((left_px, top_px, right_px, bottom_px))
                            
                            # Apply image adjustments
                            image = apply_image_adjustments(image, brightness, contrast, saturation, sharpness)
                            
                            # Apply grayscale conversion if keeping colors is disabled
                            if not keep_colors and image.mode != 'L':
                                image = image.convert('L')
                            
                            def progress(pct, msg):
                                st.session_state.status_text = msg + ' (' + str(int(pct*100)) + '%)'
                                st.session_state.status_kind = 'info'
                            
                            generator = process_single_image(
                                image, preset, enable_edge_detection, edge_strength, invert_colors,
                                {'width': model_width, 'thickness': model_thickness, 'base': base_thickness},
                                progress
                            )
                            
                            # Save STL to a temporary file and read it
                            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                                temp_path = tmp.name
                            
                            # Write STL data
                            generator.save_stl(temp_path)
                            
                            # Capture data before closing handles
                            depth_img = generator.get_depth_map_image()
                            mesh_info = generator.get_mesh_info()
                            
                            # Read the file data
                            with open(temp_path, 'rb') as f:
                                st.session_state.stl_bytes = f.read()
                            
                            # Explicitly close all handles before deleting (Windows issue)
                            f = None
                            generator = None
                            
                            # Delete the temp file
                            try:
                                os.unlink(temp_path)
                            except OSError:
                                pass  # File may already be deleted or locked, continue anyway
                            
                            st.session_state.preview_done = True
                            st.session_state.depth_image = depth_img
                            st.session_state.depth_info = mesh_info
                            st.session_state.status_text = '✅ Processing complete! Model ready for download.'
                            st.session_state.status_kind = 'success'
                            st.rerun()
                        except Exception as e:
                            st.session_state.status_text = '❌ Error: ' + str(e)
                            st.session_state.status_kind = 'error'
                            st.error(str(e))
        
        with generate_col2:
            if st.button('🗑️ Clear', use_container_width=True):
                st.session_state.preview_done = False
                st.session_state.stl_bytes = None
                st.session_state.depth_image = None
                st.session_state.status_text = 'Waiting for input…'
                st.session_state.status_kind = 'info'
                st.rerun()
        
        st.markdown('<div class=\"thin-divider\"></div>', unsafe_allow_html=True)
        
        st.markdown('#### 📊 Status')
        st.markdown(status_html(st.session_state.status_text, st.session_state.status_kind), unsafe_allow_html=True)
    
    with right_col:
        st.markdown('#### 🖼️ Preview')
        
        if st.session_state.depth_image is not None:
            st.image(st.session_state.depth_image, caption='Depth Map Preview', use_container_width=True)
            
            if st.session_state.depth_info:
                info = st.session_state.depth_info
                col1, col2, col3 = st.columns(3)
                with col1:
                    w = info.get('width', 0)
                    st.metric('Width', str(w) + 'px')
                with col2:
                    h = info.get('height', 0)
                    st.metric('Height', str(h) + 'px')
                with col3:
                    tc = info.get('triangle_count', 0)
                    st.metric('Triangles', '{:,}'.format(tc))
        else:
            st.info('Upload an image and click Generate to see preview')
        
        st.markdown('<div class=\"thin-divider\"></div>', unsafe_allow_html=True)
        
        st.markdown('#### 📥 Download STL')
        if st.session_state.stl_bytes is not None:
            st.download_button(
                label='⬇️ Download STL File',
                data=st.session_state.stl_bytes,
                file_name='relief_model.stl',
                mime='application/octet-stream',
                use_container_width=True,
            )
            
            info = st.session_state.depth_info
            if info:
                mw = info.get('model_width', model_width)
                mh = info.get('model_height', 0)
                th = info.get('total_height', 0)
                st.caption('Model: {:.1f}mm x {:.1f}mm x {:.1f}mm'.format(mw, mh, th))
        else:
            st.info('Generate a model first to enable download')

# BATCH PROCESSING TAB
with tab_batch:
    st.markdown('#### 📚 Batch Process Multiple Images')
    st.markdown('Upload multiple images to generate STL files for all at once.')
    
    batch_files = st.file_uploader(
        'Upload multiple images',
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'],
        accept_multiple_files=True,
        help='Select multiple image files',
    )
    
    if batch_files:
        st.write('📁 **' + str(len(batch_files)) + ' images selected**')
    
    # Batch Preset Templates
    st.markdown('##### 📁 Batch Presets')
    preset_load_col, preset_save_col, preset_del_col = st.columns([2, 1, 1])
    with preset_load_col:
        batch_preset_options = list(st.session_state.batch_presets.keys())
        selected_batch_preset = st.selectbox('Load Preset', batch_preset_options,
            index=batch_preset_options.index(st.session_state.current_batch_preset_name) if st.session_state.current_batch_preset_name in batch_preset_options else 0,
            key='batch_preset_selector')
        if selected_batch_preset != st.session_state.current_batch_preset_name:
            st.session_state.current_batch_preset_name = selected_batch_preset
            bp = st.session_state.batch_presets[selected_batch_preset]
            # Update all widget session state values to match loaded preset
            st.session_state.batch_preset = bp.get('preset', 'preview')
            st.session_state.batch_keep_colors = bp.get('keep_colors', False)
            st.session_state.batch_invert_colors = bp.get('invert_colors', False)
            st.session_state.batch_edge_detection = bp.get('edge_detection', False)
            st.session_state.batch_edge_strength = bp.get('edge_strength', 1.0)
            st.session_state.batch_model_width = bp.get('model_width', 50.0)
            st.session_state.batch_model_thickness = bp.get('model_thickness', 5.0)
            st.session_state.batch_base_thickness = bp.get('base_thickness', 2.0)
            st.session_state.batch_filename_pattern = bp.get('filename_pattern', '{name}_{preset}')
            st.rerun()
    with preset_save_col:
        save_batch_preset_name = st.text_input('Save as', value='', placeholder='preset name', key='save_batch_preset_name')
    with preset_del_col:
        if st.button('💾 Save Preset', key='save_batch_preset_btn', disabled=not save_batch_preset_name.strip()):
            name = save_batch_preset_name.strip()
            st.session_state.batch_presets[name] = {
                'preset': st.session_state.get('batch_preset', 'preview'),
                'keep_colors': st.session_state.get('batch_keep_colors', False),
                'invert_colors': st.session_state.get('batch_invert_colors', False),
                'edge_detection': st.session_state.get('batch_edge_detection', False),
                'edge_strength': st.session_state.get('batch_edge_strength', 1.0),
                'model_width': st.session_state.get('batch_model_width', 50.0),
                'model_thickness': st.session_state.get('batch_model_thickness', 5.0),
                'base_thickness': st.session_state.get('batch_base_thickness', 2.0),
                'filename_pattern': st.session_state.get('batch_filename_pattern', '{name}_{preset}')
            }
            st.session_state.current_batch_preset_name = name
            st.rerun()
    
    delete_batch_preset = st.selectbox('🗑️ Delete Preset', ['(none)'] + [p for p in st.session_state.batch_presets.keys() if p != 'default'], key='delete_batch_preset_selector')
    if delete_batch_preset != '(none)':
        if st.button('🗑️ Delete', key='delete_batch_preset_btn'):
            del st.session_state.batch_presets[delete_batch_preset]
            if st.session_state.current_batch_preset_name == delete_batch_preset:
                st.session_state.current_batch_preset_name = 'default'
            st.rerun()
    
    # Export/Import Presets
    st.markdown('##### 📤 Export / 📥 Import Presets')
    export_col, import_col = st.columns(2)
    with export_col:
        if st.button('⬇️ Export All Presets (JSON)', key='export_batch_presets'):
            import json
            export_data = json.dumps(st.session_state.batch_presets, indent=2)
            st.download_button(
                label='📥 Download presets.json',
                data=export_data,
                file_name='batch_presets.json',
                mime='application/json',
                key='download_presets_btn'
            )
    with import_col:
        uploaded_preset_file = st.file_uploader('📤 Import Presets (JSON)', type=['json'], key='import_preset_file')
        if uploaded_preset_file is not None:
            try:
                import json
                imported_data = json.load(uploaded_preset_file)
                if isinstance(imported_data, dict) and len(imported_data) > 0:
                    # Validate required keys in at least one preset
                    valid = True
                    for name, preset in imported_data.items():
                        if not isinstance(preset, dict):
                            valid = False
                            break
                        required_keys = ['preset', 'keep_colors', 'invert_colors', 'edge_detection', 'model_width', 'model_thickness', 'base_thickness']
                        for key in required_keys:
                            if key not in preset:
                                valid = False
                                break
                    if valid:
                        # Merge with existing presets (existing keys are overwritten)
                        for name, preset in imported_data.items():
                            st.session_state.batch_presets[name] = preset
                        st.success(f'Imported {len(imported_data)} preset(s) successfully!')
                        st.rerun()
                    else:
                        st.error('Invalid preset file format')
                else:
                    st.error('Invalid preset file format')
            except Exception as e:
                st.error('Error reading preset file: ' + str(e))
    
    # Get current preset values
    current_bp = st.session_state.batch_presets.get(st.session_state.current_batch_preset_name, st.session_state.batch_presets['default'])
    
    # Output Options
    st.markdown('##### Output Options')
    filename_pattern = st.text_input(
        'Filename Pattern',
        value=current_bp.get('filename_pattern', '{name}_{preset}'),
        help='Placeholders: {name}, {preset}, {index}, {date}, {timestamp}, {width}, {height}. Must include {name} or {index} for unique filenames.',
        key='batch_filename_pattern'
    )
    
    # Validate filename pattern only when files are selected
    pattern_valid = '{name}' in filename_pattern or '{index}' in filename_pattern
    if batch_files and not pattern_valid:
        st.error('Pattern must include {name} or {index} for unique filenames')
    
    st.markdown('##### Model Settings')
    batch_preset = st.selectbox('Quality Preset', ['draft', 'preview', 'high'], 
        index=['draft', 'preview', 'high'].index(current_bp.get('preset', 'preview')),
        key='batch_preset')
    batch_keep_colors = st.checkbox('🎨 Keep Original Colors', value=current_bp.get('keep_colors', False), key='batch_keep_colors',
        help='Keep original colors instead of converting to grayscale')
    batch_invert_colors = st.checkbox('🔄 Invert Colors (Black ↔ White)', value=current_bp.get('invert_colors', False), key='batch_invert_colors',
        help='Swap black and white in all images')
    batch_edge_detection = st.checkbox('🔍 Enable Edge Detection', value=current_bp.get('edge_detection', False), key='batch_edge_detection')
    batch_edge_strength = st.slider('Edge Strength', 0.1, 2.0, current_bp.get('edge_strength', 1.0), 0.1, key='batch_edge_strength') if batch_edge_detection else 1.0
    
    batch_model_width = st.slider('Model Width (mm)', 10.0, 200.0, current_bp.get('model_width', 50.0), 5.0, key='batch_model_width')
    batch_model_thickness = st.slider('Relief Height (mm)', 1.0, 20.0, current_bp.get('model_thickness', 5.0), 0.5, key='batch_model_thickness')
    batch_base_thickness = st.slider('Base Thickness (mm)', 0.5, 10.0, current_bp.get('base_thickness', 2.0), 0.5, key='batch_base_thickness')
    
    preset_info = {
        'draft': {'max_size': 150, 'blur': 0.3, 'gamma': 1.0, 'smoothing': True, 'hill': 3.0},
        'preview': {'max_size': 320, 'blur': 0.5, 'gamma': 1.1, 'smoothing': True, 'hill': 5.0},
        'high': {'max_size': 600, 'blur': 0.8, 'gamma': 1.2, 'smoothing': True, 'hill': 10.0},
    }
    
    batch_col1, batch_col2 = st.columns([2, 1])

    # Check if pattern is valid (has {name} or {index} for uniqueness)
    pattern_valid = '{name}' in filename_pattern or '{index}' in filename_pattern
    
    with batch_col1:
        process_batch = st.button('🚀 Process All Images', type='primary', use_container_width=True,
            disabled=len(batch_files) == 0 or not pattern_valid, key='process_batch')

    with batch_col2:
        clear_batch = st.button('🗑️ Clear', use_container_width=True, key='clear_batch')

    # Batch Results Display
    if st.session_state.batch_results:
        st.markdown('##### 📊 Processing Results')
        results = st.session_state.batch_results
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        fail_count = len(results) - success_count
        total_bytes = sum(r.get('size', 0) for r in results if r['status'] == 'success')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total', len(results))
        with col2:
            st.metric('✅ Success', success_count, delta_color='normal')
        with col3:
            st.metric('❌ Failed', fail_count, delta_color='inverse')
        with col4:
            if total_bytes > 0:
                if total_bytes < 1048576:
                    size_str = '{:.1f} KB'.format(total_bytes/1024)
                else:
                    size_str = '{:.1f} MB'.format(total_bytes/1048576)
                st.metric('📦 Size', size_str)
        
        for idx, result in enumerate(results):
            status_icon = '✅' if result['status'] == 'success' else '❌'
            timing_info = ' ({:.1f}s)'.format(result.get('elapsed', 0)) if 'elapsed' in result else ''
            msg = result.get('message', 'Completed')
            st.markdown(status_icon + ' **' + result['name'] + '**' + timing_info + ': ' + msg)
        
        if success_count > 0:
            st.markdown('---')
            
            # Create ZIP file with custom filenames
            zip_buffer = BytesIO()
            now = datetime.now()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for idx, result in enumerate(results):
                    if result['status'] == 'success' and result.get('data'):
                        name_parts = result['name'].rsplit('.', 1)
                        base_name = name_parts[0] if len(name_parts) > 1 else result['name']
                        try:
                            mi = result.get('mesh_info', {})
                            model_w = mi.get('model_width', batch_model_width)
                            model_h = mi.get('model_height', batch_model_width)
                        except:
                            model_w = batch_model_width
                            model_h = batch_model_width
                        stl_name = filename_pattern.format(
                            name=base_name, preset=batch_preset, index=idx+1,
                            date=now.strftime('%Y%m%d'), timestamp=now.strftime('%Y%m%d_%H%M%S'),
                            width=int(model_w), height=int(model_h)
                        ) + '.stl'
                        zf.writestr(stl_name, result['data'])
            
            zip_buffer.seek(0)
            
            st.download_button(
                label='⬇️ Download All (' + str(success_count) + ' STLs)',
                data=zip_buffer.getvalue(),
                file_name='relief_models_batch.zip',
                mime='application/zip',
                use_container_width=True,
            )

    # Batch Processing Loop
    if process_batch and batch_files:
        st.session_state.batch_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_preset_info = preset_info.get(batch_preset, preset_info['preview'])
        start_time = time.time()
        
        for i, uploaded_file in enumerate(batch_files):
            item_start = time.time()
            try:
                elapsed_total = time.time() - start_time
                if i > 0:
                    avg_time = elapsed_total / i
                    remaining = (len(batch_files) - i) * avg_time
                    eta_str = ' | ETA: ' + str(int(remaining)) + 's'
                else:
                    eta_str = ''
                
                pct = (i + 1) / len(batch_files) * 100
                status_text.text('⏳ [' + str(int(pct)) + '%] Processing ' + uploaded_file.name + '... (' + str(i+1) + '/' + str(len(batch_files)) + ')' + eta_str)
                progress_bar.progress((i + 0.5) / len(batch_files))
                
                image = Image.open(uploaded_file)
                image.load()
                
                # Convert to grayscale only if not keeping original colors
                if not batch_keep_colors and image.mode != 'L':
                    image = image.convert('L')
                
                generator = process_single_image(
                    image, batch_preset_info, batch_edge_detection, batch_edge_strength, batch_invert_colors,
                    {'width': batch_model_width, 'thickness': batch_model_thickness, 'base': batch_base_thickness},
                    None
                )
                
                mesh_info = generator.get_mesh_info()
                
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                    temp_path = f.name
                
                generator.save_stl(temp_path)
                
                with open(temp_path, 'rb') as f:
                    stl_data = f.read()
                
                os.unlink(temp_path)
                
                item_elapsed = time.time() - item_start
                
                st.session_state.batch_results.append({
                    'name': uploaded_file.name,
                    'status': 'success',
                    'data': stl_data,
                    'size': len(stl_data),
                    'elapsed': item_elapsed,
                    'mesh_info': mesh_info,
                    'message': 'Generated ({:.1f} KB, {:.1f}s)'.format(len(stl_data)/1024, item_elapsed)
                })
                
            except Exception as e:
                item_elapsed = time.time() - item_start
                err_msg = str(e)
                if len(err_msg) > 80:
                    err_msg = err_msg[:80]
                st.session_state.batch_results.append({
                    'name': uploaded_file.name,
                    'status': 'failed',
                    'elapsed': item_elapsed,
                    'message': 'Error: ' + err_msg
                })
            
            progress_bar.progress((i + 1) / len(batch_files))
        
        total_elapsed = time.time() - start_time
        status_text.text('✅ Batch complete! ' + str(len(batch_files)) + ' images in {:.1f}s'.format(total_elapsed))
        st.rerun()

    if clear_batch:
        st.session_state.batch_results = []
        st.rerun()