import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import io

# --- 1. MATH LOGIC (Core functionality from your desktop version) ---
def preprocess_ode_string(ode_str):
    """Adds explicit multiplication and handles exponents."""
    processed_str = ode_str.replace('^', '**')
    processed_str = re.sub(r'(\d(?:\.\d*)?)([xy])', r'\1*\2', processed_str)
    processed_str = re.sub(r'([xy])(\d(?:\.\d*)?)', r'\1*\2', processed_str)
    processed_str = re.sub(r'([x])([y])', r'\1*\2', processed_str)
    processed_str = re.sub(r'([y])([x])', r'\1*\2', processed_str)
    return processed_str

def solve_math(ode_str, x, y):
    """Safely evaluates the ODE string using numpy functions."""
    processed = preprocess_ode_string(ode_str)
    allowed_globals = {
        "x": x, "y": y, "np": np, "exp": np.exp, "log": np.log, 
        "sin": np.sin, "cos": np.cos, "tan": np.tan, "sqrt": np.sqrt, "pi": np.pi
    }
    return eval(processed, {"__builtins__": None}, allowed_globals)

# --- 2. NUMERICAL METHODS ---
def euler_method(f_str, x0, y0, xf, h):
    x_v, y_v = [x0], [y0]
    cx, cy = x0, y0
    while cx < xf - 1e-9:
        step = min(h, xf - cx)
        cy += step * solve_math(f_str, cx, cy)
        cx += step
        x_v.append(cx); y_v.append(cy)
    return x_v, y_v

def improved_euler(f_str, x0, y0, xf, h):
    """Heun's Method - Re-added as requested."""
    x_v, y_v = [x0], [y0]
    cx, cy = x0, y0
    while cx < xf - 1e-9:
        step = min(h, xf - cx)
        f_start = solve_math(f_str, cx, cy)
        y_predict = cy + step * f_start
        f_end = solve_math(f_str, cx + step, y_predict)
        cy += (step / 2) * (f_start + f_end)
        cx += step
        x_v.append(cx); y_v.append(cy)
    return x_v, y_v

def rk4_method(f_str, x0, y0, xf, h):
    x_v, y_v = [x0], [y0]
    cx, cy = x0, y0
    while cx < xf - 1e-9:
        step = min(h, xf - cx)
        k1 = solve_math(f_str, cx, cy)
        k2 = solve_math(f_str, cx + step/2, cy + (step/2)*k1)
        k3 = solve_math(f_str, cx + step/2, cy + (step/2)*k2)
        k4 = solve_math(f_str, cx + step, cy + step*k3)
        cy += (step/6) * (k1 + 2*k2 + 2*k3 + k4)
        cx += step
        x_v.append(cx); y_v.append(cy)
    return x_v, y_v

# --- 3. WEB INTERFACE (Streamlit UI) ---
st.set_page_config(page_title="Numerical ODE Solver", layout="wide")
st.title("🧮 Numerical ODE Solver")

with st.sidebar:
    st.header("Input Parameters")
    
    # 1. Blank ODE Input
    ode_input = st.text_input("dy/dx = f(x, y)", value="", placeholder="e.g., x + y")
    
    # 2. Blank Numerical Inputs (using value=None)
    col1, col2 = st.columns(2)
    x0 = col1.number_input("Initial x (x0)", value=None, placeholder="0.0")
    y0 = col2.number_input("Initial y (y0)", value=None, placeholder="1.0")
    xf = col1.number_input("Final x (xf)", value=None, placeholder="2.0")
    h = col2.number_input("Step Size (h)", value=None, placeholder="0.1")
    
    # 3. Method Selection with no default
    method = st.selectbox(
        "Choose Method", 
        ["Euler", "Improved Euler", "RK4", "Compare All"],
        index=None,
        placeholder="Select a method..."
    )
    
    solve_btn = st.button("Solve ODE")

# --- 4. EXECUTION AND VALIDATION ---
if solve_btn:
    # Validate that all inputs are provided
    if not ode_input or x0 is None or y0 is None or xf is None or h is None or method is None:
        st.error("⚠️ Please fill in all input fields and select a method.")
    else:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_results = pd.DataFrame()

            # Execute based on selection
            if method in ["Euler", "Compare All"]:
                ex, ey = euler_method(ode_input, x0, y0, xf, h)
                ax.plot(ex, ey, 'o-', label="Euler")
                df_results["x"] = ex
                df_results["Euler Y"] = ey

            if method in ["Improved Euler", "Compare All"]:
                hx, hy = improved_euler(ode_input, x0, y0, xf, h)
                ax.plot(hx, hy, '^-', label="Improved Euler (Heun)")
                if "x" not in df_results: df_results["x"] = hx
                df_results["Improved Euler Y"] = hy

            if method in ["RK4", "Compare All"]:
                rx, ry = rk4_method(ode_input, x0, y0, xf, h)
                ax.plot(rx, ry, 's-', label="RK4")
                if "x" not in df_results: df_results["x"] = rx
                df_results["RK4 Y"] = ry

            # Styling the plot
            ax.set_title(f"Numerical Solution for dy/dx = {ode_input}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            ax.grid(True)
            
            # Display Results
            st.pyplot(fig)
            st.subheader("Solution Data Table")
            st.dataframe(df_results, use_container_width=True)
            
            # Download Button (replaces record_solution_to_csv)
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="ode_solution.csv",
                mime="text/csv",
            )
            
        except Exception as e:
            st.error(f"Error evaluating math: {e}. Check your equation syntax!")