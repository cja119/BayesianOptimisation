import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.animation as animation




@tf.function
def unknown_function(X):
    A = st.session_state.A
    B = st.session_state.B
    C = st.session_state.C
    D = st.session_state.D
    E = st.session_state.E
    F = st.session_state.F
    G = st.session_state.G
    H = st.session_state.H
    return -A * tf.sin(B * X - C) + D * tf.cos(E * X) - F + G * tf.square(X) - H * (tf.exp(-X) - tf.exp(X))

def plot_function(X,A,B,C,D,E,F,G,H):
    return -A* tf.sin(B*X - C) + D * tf.cos(E*X) - F + G*tf.square(X) - H * (tf.exp(-X) - tf.exp(X))

@tf.function(reduce_retracing=True)
def radial_basis_kernel(X1, X2, variance, lengthscale):
    pairwise_diffs = tf.expand_dims(X1, 1) - tf.expand_dims(X2, 0)
    squared_diffs  = tf.square(pairwise_diffs / lengthscale)
    kernel_matrix = tf.multiply(variance, tf.exp(-squared_diffs))
    return kernel_matrix

def calculate_kernel_matrices(X_train, Y_train, lim):
    n_train = len(X_train)
    lengthscale = tf.Variable(1.0)
    variance = tf.Variable(1.0)
    RBF_prior = radial_basis_kernel(X_train, X_train, variance, lengthscale) + tf.eye(n_train) * 1e-5

    # Specifying parameters of testing 
    n_test = 100
    X_test = tf.sort(tf.random.uniform([n_test], -lim, lim))

    # Calculate the kernel matrix for the posterior distribution
    K_s = radial_basis_kernel(X_test, X_train, variance, lengthscale)
    K_ss = radial_basis_kernel(X_test, X_test, variance, lengthscale) + tf.eye(n_test) * 1e-4

    # Cholesky decomposition of the prior kernel matrix
    L = tf.linalg.cholesky(RBF_prior)
    alpha = tf.linalg.cholesky_solve(L, tf.expand_dims(Y_train, -1))
    mu_s = tf.matmul(K_s, alpha)
    v = tf.linalg.cholesky_solve(L, tf.transpose(K_s))
    cov_s = K_ss - tf.matmul(K_s, v)

    return X_test, K_s, K_ss, L, alpha, mu_s, v, cov_s

def main():
    st.title("Bayesian Optimization with Gaussian Processes")
    col1,col2,col3 = st.columns(3)
    with col1:
        noise = st.slider("Noise Level", 0.0, 1.0, 0.1)
    with col2:
        n_train = st.slider("Number of Training Points", 1, 10, 2)
    with col3:
        lim = st.slider("Limit", 1, 10, 5)

    st.text("Your unknown function is as follows:")
    st.markdown("<h6 style='text-align: center;'>Asin(BX - C) + Dcos(EX) - F + GX^2 - H (exp(-X) -exp(X))</h6>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        A = st.slider("A", -5.0, 5.0, 2.0)
        B = st.slider("B", -5.0, 5.0, 5.0)
    with col2:
        C = st.slider("C", -5.0, 5.0, 0.5)
        D = st.slider("D", -5.0, 5.0, 2.0)
    with col3:
        E = st.slider("E", -5.0, 5.0, 3.0)
        F = st.slider("F", -5.0, 5.0, 5.0)
    with col4:
        G = st.slider("G", -5.0, 5.0, 1.0)
        H = st.slider("H", -.2, 0.2, 0.05)


    fig, ax = plt.subplots()
    ax.plot(tf.linspace(-lim, lim, 100), plot_function(tf.linspace(-lim, lim, 100),A,B,C,D,E,F,G,H), color='grey', label='True function')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    if st.button("Begin Bayesian Inference"):

        if 'A' not in st.session_state:
            st.session_state.A = A
        if 'B' not in st.session_state:
            st.session_state.B = B
        if 'C' not in st.session_state:
            st.session_state.C = C
        if 'D' not in st.session_state:
            st.session_state.D = D
        if 'E' not in st.session_state:
            st.session_state.E = E
        if 'F' not in st.session_state:
            st.session_state.F = F
        if 'G' not in st.session_state:
            st.session_state.G = G
        if 'H' not in st.session_state:
            st.session_state.H = H


        if 'X_train' not in st.session_state:
            st.session_state.X_train = tf.random.uniform([n_train], -lim, lim)
            st.session_state.Y_train = unknown_function(st.session_state.X_train) + np.random.normal(0, noise, n_train)
            st.session_state.old_opt = 10000

        X_test, K_s, K_ss, L, alpha, mu_s, v, cov_s = calculate_kernel_matrices(st.session_state.X_train, st.session_state.Y_train, lim)

        mv_normal = tfp.distributions.MultivariateNormalTriL(
            loc=tf.reshape(mu_s, [-1]),
            scale_tril=tf.linalg.cholesky(cov_s) * 7.5
        )

        samples = mv_normal.sample(200)
        mean_prediction = tf.reduce_mean(samples, axis=0)
        std_prediction = tf.math.reduce_std(samples, axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in samples:
            ax.plot(X_test, i, alpha=0.25, linestyle='-')
        ax.fill_between(X_test, 
                        (mean_prediction - 3 * std_prediction).numpy(), 
                        (mean_prediction + 3 * std_prediction).numpy(), 
                        color='gray', alpha=0.2)
        ax.scatter(st.session_state.X_train, st.session_state.Y_train, color='red', label='Training data', zorder=10000)
        ax.plot(tf.linspace(-lim, lim, 100), unknown_function(tf.linspace(-lim, lim, 100)), color='blue', label='True function')
        ax.legend()
        ax.grid()

        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'r-', label='Mean prediction')
        scatter = ax.scatter([], [], color='red', label='Training data', zorder=10000)
        fill_between = ax.fill_between([], [], [], color='gray', alpha=0.2)
        mini_lines = []
        true_function_line, = ax.plot(tf.linspace(-lim, lim, 1000), unknown_function(tf.linspace(-lim, lim, 1000)), color='blue', label='True function')
        ax.legend()
        ax.grid()

        ax.set_xlim(-lim, lim)
        ax.set_ylim((np.min((mean_prediction - 3 * std_prediction).numpy()), np.max((mean_prediction + 3 * std_prediction).numpy())))

        def init():
            line.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
            nonlocal fill_between, mini_lines
            for mini_line in mini_lines:
                mini_line.remove()
            mini_lines = []
            if fill_between:
                fill_between.remove()
                fill_between = None
            return line, scatter

        def update(frame):
            nonlocal fill_between, mini_lines, mean_prediction, std_prediction, X_test
            arg_min = np.argmin((mean_prediction - 3 * std_prediction).numpy())
            new_opt = unknown_function(X_test[arg_min].numpy())
            if np.abs(new_opt - st.session_state.old_opt) < 10e-1:
                arg_min = np.argmax(std_prediction)
                new_opt = unknown_function(X_test[arg_min].numpy())
            else:
                st.session_state.old_opt = new_opt
            st.session_state.X_train = tf.concat([st.session_state.X_train, [X_test[arg_min].numpy()]], axis=0)
            st.session_state.Y_train = tf.concat([st.session_state.Y_train, [new_opt + np.random.normal(0, noise)]], axis=0)

            X_test, K_s, K_ss, L, alpha, mu_s, v, cov_s = calculate_kernel_matrices(st.session_state.X_train, st.session_state.Y_train, lim)
            mv_normal = tfp.distributions.MultivariateNormalTriL(loc=tf.reshape(mu_s, [-1]), scale_tril=tf.linalg.cholesky(cov_s) * 7.5)
            samples = mv_normal.sample(20)
            mean_prediction = tf.reduce_mean(samples, axis=0)
            std_prediction = tf.math.reduce_std(samples, axis=0)

            line.set_data(X_test.numpy(), mean_prediction.numpy())
            scatter.set_offsets(np.c_[st.session_state.X_train.numpy(), st.session_state.Y_train.numpy()])

            if fill_between:
                fill_between.remove()
            fill_between = ax.fill_between(X_test.numpy().flatten(), 
                                        (mean_prediction - 3 * std_prediction).numpy().flatten(), 
                                        (mean_prediction + 3 * std_prediction).numpy().flatten(), 
                                        color='gray', alpha=0.2)

            for mini_line in mini_lines:
                mini_line.remove()
            mini_lines = ax.plot(X_test.numpy(), samples.numpy().T, alpha=0.25, linestyle='-')
            plt.title(f'Bayesian optimisation of a Gaussian Process with {len(st.session_state.X_train)} Evaluations')
            return line, scatter, fill_between, *mini_lines

        ani = animation.FuncAnimation(fig, update, frames=18, init_func=init, blit=True, repeat=False)
        ani_html = ani.to_jshtml()
        st.components.v1.html(f"<div style='display: flex; justify-content: left;'>{ani_html}</div>", height=3000, width=1000)

if __name__ == "__main__":
    main()