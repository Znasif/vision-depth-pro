#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
streamlit run "$SCRIPT_DIR/src/depth_pro/cli/streamlit_server.py" "$@"