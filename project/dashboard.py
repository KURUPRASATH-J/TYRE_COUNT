import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import cv2

# Page configuration
st.set_page_config(
    page_title="Height Measurement Dashboard",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📏 Stereo Vision Height Measurement Dashboard</h1>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        data_file = Path("measurement_data.json")
        if not data_file.exists():
            return None, None
        
        with open(data_file, 'r') as f:
            content = f.read().strip()
            if not content or content == '{}':
                return None, None
            data = json.loads(content)
        
        measurements = data.get('measurements', [])
        statistics = data.get('statistics', {})
        
        if measurements and len(measurements) > 0:
            df = pd.DataFrame(measurements)
            return df, statistics
        return None, statistics
    except json.JSONDecodeError as e:
        return None, None
    except Exception as e:
        return None, None

df, stats = load_data()

if df is None or df.empty:
    st.warning("⚠️ No measurement data found. Please run `height.py` first to generate measurement data.")
    st.info("The dashboard will automatically update once `measurement_data.json` is available.")
    
    # Show file status
    if Path("measurement_data.json").exists():
        try:
            with open("measurement_data.json", 'r') as f:
                content = f.read().strip()
                if not content or content == '{}':
                    st.error("⚠️ Data file exists but is empty. Please run `height.py` to generate measurements.")
                else:
                    try:
                        json.loads(content)
                        st.error("⚠️ Data file exists but contains no valid measurements. Please run `height.py` to generate new data.")
                    except json.JSONDecodeError:
                        st.error("⚠️ Data file is corrupted or incomplete. Please delete `measurement_data.json` and run `height.py` again.")
        except Exception:
            st.error("⚠️ Data file exists but cannot be read. Please run `height.py` to regenerate it.")
    
    st.markdown("---")
    st.subheader("📝 Instructions")
    st.markdown("""
    1. **Run the measurement script**: Execute `python height.py` in your terminal  
    2. **Wait for completion**: The script will process your video files and generate measurements  
    3. **Refresh this page**: The dashboard will automatically load the new data  
    """)
else:
    # Simple constants
    ITEM_HEIGHT_MM = 21.5
    VIDEO_CANDIDATES = ["height_verification.mp4", "leftdetect.mp4", "left.mp4"]

    @st.cache_data
    def _pick_video_path() -> str | None:
        for p in VIDEO_CANDIDATES:
            if Path(p).exists():
                return p
        return None

    @st.cache_data
    def get_frame_rgb(video_path: str, frame_number: int):
        """Return RGB image (np.ndarray) for a frame index, or None."""
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    # -------- Summary section (very simple) --------
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        if stats and "median_height" in stats:
            st.metric("Final Height (median)", f"{stats['median_height']:.2f} mm")

    with col2:
        if stats and "median_height" in stats:
            total_height = stats["median_height"]
            num_items = int(round(total_height / ITEM_HEIGHT_MM))
            st.metric("Item Count", f"{num_items}", help=f"{total_height:.2f} mm ÷ {ITEM_HEIGHT_MM} mm")

    with col3:
        if stats:
            st.metric("Valid Measurements", stats.get("valid_measurements", len(df)))

    st.markdown("---")

    # -------- Detection frames (just numbers, no graphs) --------
    st.subheader("Detection Frames")
    frames = sorted(df["frame"].tolist())
    st.write(f"Total detection frames: **{len(frames)}**")
    if frames:
        st.write(f"Frame range: **{frames[0]} – {frames[-1]}**")
        with st.expander("Show all frame numbers"):
            st.text(", ".join(str(f) for f in frames))

    st.markdown("---")

    # -------- Detected images only (simple) --------
    st.subheader("Detected Images")

    video_path = _pick_video_path()
    if not frames:
        st.info("No detection frames available.")
    elif not video_path:
        st.warning(
            "No video found to extract frames. Place one of these files next to the dashboard: "
            + ", ".join(VIDEO_CANDIDATES)
        )
    else:
        st.caption(f"Source video: `{video_path}`")

        selected_frame = st.selectbox("Select a detected frame", frames)
        img = get_frame_rgb(video_path, int(selected_frame))
        if img is None:
            st.warning(f"Could not read frame {selected_frame} from `{video_path}`.")
        else:
            st.image(img, caption=f"Detected frame: {selected_frame}", use_container_width=True)

        # Thumbnail grid (limit for speed)
        with st.expander("Show thumbnails (first 24)"):
            max_thumbs = min(24, len(frames))
            cols_per_row = 6
            rows = (max_thumbs + cols_per_row - 1) // cols_per_row

            for r in range(rows):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    idx = r * cols_per_row + c
                    if idx >= max_thumbs:
                        break
                    fno = int(frames[idx])
                    thumb = get_frame_rgb(video_path, fno)
                    with cols[c]:
                        if thumb is None:
                            st.write(f"{fno}")
                        else:
                            h, w = thumb.shape[:2]
                            scale = 200 / max(h, w)
                            new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                            thumb_small = cv2.resize(thumb, (new_w, new_h))
                            st.image(thumb_small, caption=f"{fno}", use_container_width=True)

    st.markdown("---")

    # -------- Simple measurements table --------
    st.subheader("Measurements")

    simple_df = df[["frame", "height", "conf1", "conf2"]].copy()
    simple_df["height"] = simple_df["height"].round(2)
    simple_df["items"] = np.round(simple_df["height"] / ITEM_HEIGHT_MM).astype(int)
    simple_df["conf1"] = simple_df["conf1"].round(3)
    simple_df["conf2"] = simple_df["conf2"].round(3)

    simple_df = simple_df[["frame", "height", "items", "conf1", "conf2"]]
    simple_df.columns = ["Frame", "Height (mm)", "Items", "Cam1 Conf", "Cam2 Conf"]

    st.dataframe(simple_df, use_container_width=True, height=400)

    # Optional CSV download
    csv = simple_df.to_csv(index=False)
    st.download_button(
        label="Download measurements as CSV",
        data=csv,
        file_name="height_measurements_simple.csv",
        mime="text/csv",
    )


