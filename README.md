# Indian-Railways-Inventory-and-Revenue-Management
A single Streamlit app that unifies forecasting, fare optimization, seat/coach operations, compliance constraints, revenue simulation, and audit-ready persistence for IRCTC trains.
# Steps to run the application
Create the python environment using: python -m venv .venv

Activate the scripts using .venv\scripts\activate

Install streamlit: pip install streamlit

Install other dependencies : pip install "streamlit>=1.32,<2.0" pyyaml pandas "numpy>=1.26,<3" requests plotly matplotlib pillow "altair>=5.2" "pyarrow>=15.0" "openpyxl>=3.1" "xlsxwriter>=3.1" python-dateutil

Run the app using command: streamlit run irctc_rirm_revenue_mgmt.py
