"""Streamlit UI components: CSS, header, status, step indicators, dividers."""

import streamlit as st


# ── CSS Styles ────────────────────────────────────────────────────────────

CUSTOM_CSS = '''
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
'''


# ── Rendering Helpers ─────────────────────────────────────────────────────

def status_html(text: str, kind: str = 'info') -> str:
    """Render a status box as HTML.

    Args:
        text: Status message text.
        kind: 'info' (default), 'success', or 'error'.

    Returns:
        HTML string for the status box.
    """
    css_class = {'info': '', 'success': ' success', 'error': ' error'}.get(kind, '')
    return '<div class="status-box' + css_class + '">' + text + '</div>'


def render_css():
    """Inject the custom CSS into the page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header():
    """Render the main app header."""
    st.markdown('''
    <div class="main-header">
      <h1>🎭 3D Relief Model Generator</h1>
      <p>
        Convert any image into a printable 3D relief STL model using professional
        depth-map algorithms — edge-preserving bilateral filtering, non-local means
        denoising, guided filtering, and morphological smoothing.
      </p>
    </div>
    ''', unsafe_allow_html=True)


def render_step_indicators(preview_done: bool, stl_ready: bool):
    """Render the 1-2-3 step indicator bubbles.

    Args:
        preview_done: Whether the depth preview has been generated.
        stl_ready: Whether an STL model is ready for download.
    """
    step1_class = 'done' if preview_done else 'active'
    step2_class = 'active' if preview_done else 'inactive'
    step3_class = 'done' if stl_ready else 'inactive'

    st.markdown('''
    <div class="step-row">
      <div class="step-bubble ''' + step1_class + '''">1</div>
      <div class="step-label">Upload image &amp; generate depth preview</div>
    </div>
    <div class="step-row">
      <div class="step-bubble ''' + step2_class + '''">2</div>
      <div class="step-label">Generate STL model</div>
    </div>
    <div class="step-row">
      <div class="step-bubble ''' + step3_class + '''">3</div>
      <div class="step-label">Download &amp; 3D print</div>
    </div>
    ''', unsafe_allow_html=True)


def render_divider():
    """Render a thin horizontal divider."""
    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
