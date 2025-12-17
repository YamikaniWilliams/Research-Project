import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import time

# --- MATH LOGIC (Kept from your original script) ---
def preprocess_ode_string(ode_str):
    processed_str = ode_str.replace('^', '**')
    processed_str = re.sub(r'(\d)([xy])', r'\1*\2', processed_str)
    processed_str = re.sub(r'([xy])(\d)', r'\1*\2', processed_str)
    processed_str = re.sub(r'([x])([y])', r'\1*\2', processed_str)
    processed_str = re.sub(r'([y])([x])', r'\1*\2', processed_str)
    return processed_str

def parse_function(ode_str, x, y):
    processed = preprocess_ode_string(ode_str)
    allowed_globals = {
        "x": x, "y": y, "np": np, "exp": np.exp, "log": np.log,
        "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "pi": np.pi
    }
    return eval(processed, {"__builtins__": None}, allowed_globals)

def euler_method(f_str, x0, y0, xf, h):
    x_vals, y_vals = [x0], [y0]
    curr_x, curr_y = x0, y0
    while curr_x < xf - 1e-9:
        step = min(h, xf - curr_x)
        curr_y += step * parse_function(f_str, curr_x, curr_y)
        curr_x += step
        x_vals.append(curr_x); y_vals.append(curr_y)
    return x_vals, y_vals

def rk4_method(f_str, x0, y0, xf, h):
    x_vals, y_vals = [x0], [y0]
    curr_x, curr_y = x0, y0
    while curr_x < xf - 1e-9:
        step = min(h, xf - curr_x)
        k1 = parse_function(f_str, curr_x, curr_y)
        k2 = parse_function(f_str, curr_x + step/2, curr_y + (step/2)*k1)
        k3 = parse_function(f_str, curr_x + step/2, curr_y + (step/2)*k2)
        k4 = parse_function(f_str, curr_x + step, curr_y + step*k3)
        curr_y += (step/6) * (k1 + 2*k2 + 2*k3 + k4)
        curr_x += step
        x_vals.append(curr_x); y_vals.append(curr_y)
    return x_vals, y_vals

# --- STREAMLIT UI ---
st.set_page_config(page_title="ODE Solver", layout="wide")
st.title("🧮 Numerical ODE Solver")

# Sidebar for Inputs (Works great on Mobile)
with st.sidebar:
    st.header("Parameters")
    ode_input = st.text_input("dy/dx = f(x, y)", value="x + y")
    col1, col2 = st.columns(2)
    x0 = col1.number_input("Initial x (x0)", value=0.0)
    y0 = col2.number_input("Initial y (y0)", value=1.0)
    xf = col1.number_input("Final x (xf)", value=2.0)
    h = col2.number_input("Step Size (h)", value=0.1, format="%.4f")
    
    method = st.selectbox("Method", ["Euler", "RK4", "Compare Both"])

if st.button("Solve ODE"):
    try:
        start_time = time.time()
        
        # Calculation Logic
        if method == "Euler" or method == "Compare Both":
            ex, ey = euler_method(ode_input, x0, y0, xf, h)
        if method == "RK4" or method == "Compare Both":
            rx, ry = rk4_method(ode_input, x0, y0, xf, h)

        # Plotting
        fig, ax = plt.subplots()
        if method == "Euler":
            ax.plot(ex, ey, 'o-', label="Euler")
        elif method == "RK4":
            ax.plot(rx, ry, 's-', label="RK4", color="orange")
        else:
            ax.plot(ex, ey, 'o-', label="Euler")
            ax.plot(rx, ry, 's-', label="RK4")
            
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        
        # Display Results in Columns
        res_col, plot_col = st.columns([1, 2])
        
        with plot_col:
            st.pyplot(fig)
            
        with res_col:
            st.success(f"Solved in {time.time() - start_time:.4f}s")
            # Create Table
            df = pd.DataFrame({"x": ex, "y": ey}) if method == "Euler" else pd.DataFrame({"x": rx, "y": ry})
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")