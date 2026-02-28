"""UMC Streamlit Dashboard — Modern compression analytics UI.

Launch:
    streamlit run umc/dashboard.py --theme.base=dark
    python -m umc dashboard
"""

import io
import time
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="UMC Dashboard",
    page_icon="<compressed>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom styling
st.markdown("""
<style>
    .stMetric { background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 0.5rem; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


def _sidebar():
    st.sidebar.title("UMC")
    st.sidebar.caption("Universal Manifold Codec")
    return st.sidebar.radio(
        "Navigation",
        ["Compress", "Decompress", "Inspect", "Benchmark", "About"],
        label_visibility="collapsed",
    )


def _page_compress():
    st.header("Compress")
    st.caption("Encode data to .mnf format using the storage tier (no model needed)")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Upload data (.npy or .csv)", type=["npy", "csv"])
        mode = st.selectbox("Storage mode", ["lossless", "near_lossless", "lossless_zstd", "lossless_lzma",
                                             "normalized_lossless", "normalized_lossless_zstd"])

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".npy"):
                data = np.load(io.BytesIO(uploaded.read()))
            else:
                import pandas as pd
                df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
                data = df.values.astype(np.float32)
                # Reshape to 3D if needed
                if data.ndim == 2:
                    win_size = min(32, data.shape[0])
                    n_windows = data.shape[0] // win_size
                    data = data[:n_windows * win_size].reshape(n_windows, win_size, -1)

            if data.ndim != 3:
                st.error(f"Expected 3D array (n_windows, window_size, n_features), got shape {data.shape}")
                return

            st.info(f"Loaded: {data.shape[0]} windows, shape {data.shape}, {data.nbytes:,} bytes")

            if st.button("Compress", type="primary"):
                from umc.codec.tiered import _compress_storage

                with st.spinner("Compressing..."):
                    start = time.perf_counter()
                    compressed = _compress_storage(data.astype(np.float32), mode)
                    elapsed = time.perf_counter() - start

                ratio = data.nbytes / len(compressed)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Raw Size", f"{data.nbytes:,} B")
                c2.metric("Compressed", f"{len(compressed):,} B")
                c3.metric("Ratio", f"{ratio:.1f}x")
                c4.metric("Time", f"{elapsed*1000:.0f} ms")

                # Download button
                st.download_button(
                    "Download .mnf",
                    data=compressed,
                    file_name=f"compressed_{mode}.mnf.bin",
                    mime="application/octet-stream",
                )

        except Exception as e:
            st.error(f"Error: {e}")


def _page_decompress():
    st.header("Decompress")
    st.caption("Decode compressed storage tier data back to float32")

    uploaded = st.file_uploader("Upload compressed file", type=["bin", "mnf"])

    if uploaded is not None:
        try:
            data = uploaded.read()
            from umc.codec.tiered import _decompress_storage

            with st.spinner("Decompressing..."):
                start = time.perf_counter()
                decoded = _decompress_storage(data)
                elapsed = time.perf_counter() - start

            c1, c2, c3 = st.columns(3)
            c1.metric("Compressed", f"{len(data):,} B")
            c2.metric("Decoded Shape", str(decoded.shape))
            c3.metric("Time", f"{elapsed*1000:.0f} ms")

            # Preview first window
            st.subheader("Preview (first window)")
            fig = go.Figure()
            for f in range(min(decoded.shape[2], 5)):
                fig.add_trace(go.Scatter(y=decoded[0, :, f], name=f"Feature {f}"))
            fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

            # Download
            buf = io.BytesIO()
            np.save(buf, decoded)
            st.download_button(
                "Download .npy",
                data=buf.getvalue(),
                file_name="decoded.npy",
                mime="application/octet-stream",
            )

        except Exception as e:
            st.error(f"Error: {e}")


def _page_inspect():
    st.header("Inspect")
    st.caption("Analyze .mnf file structure and compression breakdown")

    uploaded = st.file_uploader("Upload .mnf file", type=["mnf"])

    if uploaded is not None:
        try:
            from umc.storage.mnf_format import MNFReader, HEADER_SIZE

            data = uploaded.read()

            from umc.storage.mnf_format import MNFHeader
            header = MNFHeader.from_bytes(data[:HEADER_SIZE])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Header")
                st.json({
                    "magic": header.magic.decode("ascii", errors="replace"),
                    "version": header.version,
                    "domain_id": header.domain_id,
                    "n_samples": header.n_samples,
                    "latent_dim": header.latent_dim,
                    "n_charts": header.n_charts,
                    "coord_dtype": "float16" if header.coord_dtype == 0 else "float32",
                    "has_index": header.has_index,
                    "has_confidence": header.has_confidence,
                    "has_scale_factors": header.has_scale_factors,
                    "has_vq_codes": header.has_vq_codes,
                    "has_residual": header.has_residual,
                    "has_tiered": header.has_tiered,
                })

            with col2:
                st.subheader("Size Breakdown")
                file_size = len(data)
                coord_size = header.n_samples * header.latent_dim * header.bytes_per_coord
                chart_size = header.n_samples

                sizes = {
                    "Header": HEADER_SIZE,
                    "Coordinates": coord_size,
                    "Chart IDs": chart_size,
                    "Other blocks": file_size - HEADER_SIZE - coord_size - chart_size,
                }

                fig = go.Figure(data=[go.Pie(
                    labels=list(sizes.keys()),
                    values=list(sizes.values()),
                    hole=0.4,
                )])
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)

                st.metric("Total File Size", f"{file_size:,} bytes")

        except Exception as e:
            st.error(f"Error: {e}")


def _page_benchmark():
    st.header("Benchmark")
    st.caption("Compare UMC storage compression across data types and against baselines")

    n_windows = st.slider("Windows per data type", 100, 5000, 500, 100)

    if st.button("Run Benchmark", type="primary"):
        from umc.data.synthetic import generate_all_types
        from umc.codec.tiered import _compress_storage
        from umc.codec.gorilla import gorilla_compress
        import zlib

        with st.spinner("Generating data and running benchmarks..."):
            all_data = generate_all_types(n_windows=n_windows)

            rows = []
            for dtype_name, data in all_data.items():
                raw_bytes = data.nbytes
                flat = data.ravel().astype(np.float32)

                # UMC lossless
                umc_comp = _compress_storage(data, "lossless")
                umc_ratio = raw_bytes / len(umc_comp)

                # UMC near-lossless
                umc_nl_comp = _compress_storage(data, "near_lossless")
                umc_nl_ratio = raw_bytes / len(umc_nl_comp)

                # Gorilla
                gorilla_comp = gorilla_compress(flat)
                gorilla_ratio = flat.nbytes / len(gorilla_comp)

                # zlib baseline
                zlib_comp = zlib.compress(flat.tobytes(), 9)
                zlib_ratio = flat.nbytes / len(zlib_comp)

                rows.append({
                    "Data Type": dtype_name,
                    "UMC Lossless": umc_ratio,
                    "UMC Near-Lossless": umc_nl_ratio,
                    "Gorilla": gorilla_ratio,
                    "zlib": zlib_ratio,
                })

        # Bar chart
        fig = go.Figure()
        for method in ["UMC Lossless", "UMC Near-Lossless", "Gorilla", "zlib"]:
            fig.add_trace(go.Bar(
                x=[r["Data Type"] for r in rows],
                y=[r[method] for r in rows],
                name=method,
            ))
        fig.update_layout(
            title="Compression Ratio by Data Type",
            yaxis_title="Compression Ratio (x)",
            barmode="group",
            height=450,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        import pandas as pd
        df = pd.DataFrame(rows)
        for col in ["UMC Lossless", "UMC Near-Lossless", "Gorilla", "zlib"]:
            df[col] = df[col].map(lambda x: f"{x:.1f}x")
        st.dataframe(df, use_container_width=True, hide_index=True)


def _page_about():
    st.header("About UMC")
    st.markdown("""
**Universal Manifold Codec** is a neural compression system for structured data.

### Architecture
- **Tier 1 (Search):** VQ codebook indices for FAISS similarity search
- **Tier 2 (Storage):** Byte-transposed compressed data for lossless retrieval

### Storage Modes
| Mode | Ratio | Fidelity |
|------|-------|----------|
| `lossless` | 1.2-1.5x | Bit-exact |
| `lossless_zstd` | 1.2-1.5x | Bit-exact |
| `lossless_lzma` | 1.2-1.7x | Bit-exact |
| `normalized_lossless` | 1.2-1.4x | ~1e-7 RMSE |
| `normalized_lossless_zstd` | 1.2-1.4x | ~1e-7 RMSE |
| `near_lossless` | 2.2-2.6x | <0.01% RMSE |

### File Format
`.mnf` — Manifold Native File format with 64-byte header,
coordinate blocks, chart IDs, and optional VQ/tiered/index blocks.
    """)


def main():
    page = _sidebar()

    if page == "Compress":
        _page_compress()
    elif page == "Decompress":
        _page_decompress()
    elif page == "Inspect":
        _page_inspect()
    elif page == "Benchmark":
        _page_benchmark()
    elif page == "About":
        _page_about()


if __name__ == "__main__":
    main()
