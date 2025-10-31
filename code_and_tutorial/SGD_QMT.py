import numpy as np
import jax
import jax.numpy as jnp
import optax
import itertools
import matplotlib.pyplot as plt
from tqdm import trange
from functools import partial
from itertools import product
import time  # <-- Added for dynamic seeding
from qutip import rand_unitary
import cvxpy as cp
from scipy.stats import wasserstein_distance
from jax.scipy.special import gammaln

####### Generating Pauli projectors ############

paulis = {
    "I": jnp.array([[1,0],[0,1]], dtype=jnp.complex64),
    "X": jnp.array([[0,1],[1,0]], dtype=jnp.complex64),
    # "Y": jnp.array([[0,-1j],[1j,0]], dtype=jnp.complex64),
    "Z": jnp.array([[1,0],[0,-1]], dtype=jnp.complex64),
}

def kron_n(mats):
    """Kronecker product of list of matrices"""
    out = mats[0]
    for m in mats[1:]:
        out = jnp.kron(out, m)
    return out

def random_pauli_projector_povm(key, n_qubits):
    """
    Generate a POVM consisting of 2^n projectors
    from the eigenbasis of a random n-qubit Pauli operator.
    """
    dim = 2**n_qubits
    
    # pick random Pauli string (excluding all-Identity to avoid trivial POVM)
    pauli_labels = list(itertools.product("IXZ", repeat=n_qubits))
    pauli_labels = [lbl for lbl in pauli_labels if not all(ch=="I" for ch in lbl)]
    
    key, subkey = jax.random.split(key)
    label = pauli_labels[jax.random.randint(subkey, (), 0, len(pauli_labels))]
    
    # build operator
    mats = [paulis[ch] for ch in label]
    P = kron_n(mats)
    
    # diagonalize
    eigvals, eigvecs = jnp.linalg.eigh(P)
    
    # projectors onto each eigenvector
    povms = jnp.stack([
        eigvecs[:,i][:,None] @ eigvecs[:,i][None,:].conj()
        for i in range(dim)
    ])
    
    # check: sum of POVM = identity
    povm_sum = jnp.sum(povms, axis=0)
    assert jnp.allclose(povm_sum, jnp.eye(dim), atol=1e-6), "POVM not normalized!"
    
    return povms


def generate_random_povms(key, dimension, num_povms):
    dim=dimension
    keys = jax.random.split(key, num_povms)
    def random_psd(k):
        A = jax.random.normal(k, (dim, dim)) + 1j * jax.random.normal(k, (dim, dim))
        return A.conj().T @ A
    unnormalized = jnp.stack([random_psd(k) for k in keys])
    S = jnp.sum(unnormalized, axis=0)
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(jnp.real(eigvals), 1e-6)
    D_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(eigvals))
    S_inv_sqrt = eigvecs @ D_inv_sqrt @ eigvecs.T.conj()
    normalized = jnp.array([S_inv_sqrt @ E @ S_inv_sqrt for E in unnormalized])
    return normalized

def get_default_probe_states(n_qubits):
    zero = jnp.array([1.0, 0.0])
    one = jnp.array([0.0, 1.0])
    plus = (zero + one) / jnp.sqrt(2)
    plus_i = (zero + 1j * one) / jnp.sqrt(2)
    basis = [zero, one, plus, plus_i]
    combos = list(product(basis, repeat=n_qubits))
    states = []
    for combo in combos:
        ket = combo[0]
        for b in combo[1:]:
            ket = jnp.kron(ket, b)
        rho = jnp.outer(ket, ket.conj())
        states.append(rho)
    return jnp.array(states)


def n_qubit_depolarizing_channel(states, p):
    """
    Apply N-qubit depolarizing channel to input states.
    
    Args:
        states: jnp.array of shape (num_states, 2^N, 2^N) or (2^N, 2^N)
        p: depolarizing probability (0 <= p <= 1)
    
    Returns:
        Depolarized states (same shape as input)
    """
    # detect N from dimension
    if states.ndim == 2:  
        dim = states.shape[0]
    elif states.ndim == 3:  
        dim = states.shape[1]
    else:
        raise ValueError("States must have shape (dim, dim) or (num, dim, dim).")

    identity = jnp.eye(dim, dtype=states.dtype)
    noise = identity / dim

    if states.ndim == 2:
        return (1 - p) * states + p * noise
    else:  # batch case
        return (1 - p) * states + p * noise[None, :, :]
    

def unitarily_complete_probe_states(n_qubits):
    # Single-qubit states
    zero = jnp.array([1.0, 0.0])
    one = jnp.array([0.0, 1.0])
    plus = (zero + one) / jnp.sqrt(2)

    states = []

    # Generate computational basis states
    for bits in product([0, 1], repeat=n_qubits):
        if all(b == 1 for b in bits):
            # Replace all-ones with |+++...+>
            ket = plus
            for _ in range(n_qubits - 1):
                ket = jnp.kron(ket, plus)
        else:
            # Standard computational basis
            ket = zero if bits[0] == 0 else one
            for b in bits[1:]:
                ket = jnp.kron(ket, zero if b == 0 else one)
        
        rho = jnp.outer(ket, ket.conj())
        states.append(rho)

    return jnp.array(states)


def get_coherent_probes(dimension=32, grid_size=32, grid_range=4.0):
    """
    Generate a set of coherent probe states in a truncated Hilbert space.

    Args:
        dim (int): Truncated Hilbert space dimension.
        grid_size (int): Number of points per axis in phase space grid.
        grid_range (float): Range of x,y values (grid spans [-grid_range, grid_range]).

    Returns:
        jnp.ndarray: Array of shape (grid_size**2, dim, dim),
                     each entry is a density matrix |α><α|.
    """
    # Fock basis indices
    n = jnp.arange(dimension)

    # Precompute sqrt(n!) using gammaln
    sqrt_fact = jnp.exp(0.5 * gammaln(n + 1.0))

    # Coherent state coefficients
    def coherent_state(alpha):
        coeffs = jnp.exp(-0.5 * jnp.abs(alpha)**2) * (alpha**n) / sqrt_fact
        coeffs = coeffs / jnp.linalg.norm(coeffs)  # renormalize in truncated space
        return coeffs

    xs = jnp.linspace(-grid_range, grid_range, grid_size)
    ys = jnp.linspace(-grid_range, grid_range, grid_size)

    states = []
    for x, y in itertools.product(xs, ys):
        alpha = x + 1j*y
        ket = coherent_state(alpha)
        rho = jnp.outer(ket, ket.conj())
        states.append(rho)

    return jnp.array(states)  # shape: (grid_size**2, dim, dim)



def get_true_povms_photon_detection(dimension=32, dtype=jnp.complex64):
    """
    Photon-detection POVMs in a truncated Fock basis:
        Π1 = |0><0|   (no photon)
        Π2 = I - |0><0|  (photon detected)

    Returns:
        povms: jnp.ndarray of shape (2, dim, dim)
    """
    dim = dimension
    e0 = jnp.zeros((dim,), dtype=dtype)
    e0 = e0.at[0].set(1.0 + 0j)
    Pi1 = jnp.outer(e0, jnp.conj(e0))    # |0><0|
    I = jnp.eye(dim, dtype=dtype)
    Pi2 = I - Pi1
    return jnp.stack([Pi1, Pi2], axis=0)  # shape (2, dim, dim)


def get_true_povms_photon_counting(dimension=32, dtype=jnp.complex64):
    """
    Photon-counting POVMs in a truncated Fock basis:
        Π_i = |i><i|   for i = 0, ..., dim-1

    Args:
        dimension (int): Hilbert space dimension (truncated Fock basis).
        dtype: JAX dtype (default complex64).

    Returns:
        povms: jnp.ndarray of shape (dim, dim, dim),
               where povms[i] = |i><i|
    """
    povms = []
    for i in range(dimension):
        ei = jnp.zeros((dimension,), dtype=dtype)
        ei = ei.at[i].set(1.0 + 0j)        # |i>
        Pi = jnp.outer(ei, jnp.conj(ei))   # |i><i|
        povms.append(Pi)
    return jnp.stack(povms, axis=0)


def make_psd(L):
    return L.conj().T @ L

def inv_sqrtm_psd(matrix):
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    eigvals = jnp.real(jnp.clip(eigvals, a_min=1e-6))
    D_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(eigvals))
    return eigvecs @ D_inv_sqrt @ eigvecs.T.conj()

def construct_povms(Ls):
    Es = jax.vmap(make_psd)(Ls)
    S = jnp.sum(Es, axis=0)
    S_inv_sqrt = inv_sqrtm_psd(S)
    return jax.vmap(lambda E: S_inv_sqrt @ E @ S_inv_sqrt)(Es)

def simulate_measurements(povms, states):
    def prob(E, rho):
        return jnp.real(jnp.trace(E @ rho))
    probs_for_state = jax.vmap(prob, in_axes=(0, None))
    return jax.vmap(probs_for_state, in_axes=(None, 0))(povms, states)

def plot_true_vs_reconstructed(true_probs, recon_probs, num_points=50):
    true_probs = np.array(true_probs)
    recon_probs = np.array(recon_probs)
    
    num_probe_states, num_povms = true_probs.shape
    total_points = num_probe_states * num_povms

    true_flat = true_probs.flatten()
    recon_flat = recon_probs.flatten()
    
    indices = np.random.choice(total_points, size=min(num_points, total_points), replace=False)

    sampled_true = true_flat[indices]
    sampled_recon = recon_flat[indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(sampled_true)), sampled_true, marker='o', label='True')
    plt.scatter(range(len(sampled_recon)), sampled_recon, marker='x', label='Reconstructed')
    plt.xlabel('Sample Index (random measurement outcomes)')
    plt.ylabel('Probability')
    plt.title('Random Samples of True vs. Reconstructed Probabilities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_povms_3d_random(true_povms, recon_povms, m):
    assert true_povms.shape == recon_povms.shape
    num_povms = true_povms.shape[0]
    assert m <= num_povms, "m must be less or equal to the number of available POVMs"

    # Randomly select m unique indices from the POVMs
    selected_indices = np.random.choice(num_povms, size=m, replace=False)

    fig = plt.figure(figsize=(2 * m * 2, 8))
    max_height = max(jnp.max(jnp.abs(true_povms)), jnp.max(jnp.abs(recon_povms)))

    for plot_i, i in enumerate(selected_indices):
        ax1 = fig.add_subplot(2, m, plot_i + 1, projection='3d')
        z_true = jnp.abs(true_povms[i])
        xpos, ypos = jnp.meshgrid(jnp.arange(z_true.shape[0]), jnp.arange(z_true.shape[1]), indexing="ij")
        ax1.bar3d(xpos.ravel(), ypos.ravel(), 0, 0.5, 0.5, z_true.ravel())
        ax1.set_title(f"True POVM {i}")
        ax1.set_zlim(0, max_height * 1.2)

        ax2 = fig.add_subplot(2, m, plot_i + 1 + m, projection='3d')
        z_recon = jnp.abs(recon_povms[i])
        ax2.bar3d(xpos.ravel(), ypos.ravel(), 0, 0.5, 0.5, z_recon.ravel())
        ax2.set_title(f"Recon POVM {i}")
        ax2.set_zlim(0, max_height * 1.2)

    plt.tight_layout()
    plt.show()

def computational_basis_projectors(n_qubits):
    dim = 2 ** n_qubits
    projectors = []
    for i in range(dim):
        e = jnp.zeros((dim,))
        e = e.at[i].set(1.0)
        proj = jnp.outer(e, e.conj())
        projectors.append(proj)
    return jnp.stack(projectors)

######################################

def Run_HonestQMT_StatePovmBatched(
    dimension,
    num_povms,
    rank,
    state_batch_size,
    povm_batch_size,
    learning_rate,
    n_steps,
    stop,
    probe_states,
    target_probs,
    LossPlot=True,
    seed=None,
    loss_type="mse",
):
    dim = dimension
    num_states = probe_states.shape[0]

    if seed is None:
        seed = int(time.time_ns() % (2**32))
    print(f"Seed value: {seed}")
    key_params = jax.random.PRNGKey(seed)

    # Initialize params
    keys = jax.random.split(key_params, num_povms)
    params = jnp.stack([jax.random.normal(k, (rank, dim)) * 0.1 for k in keys])
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Loss function
    def loss_fn(params, batch_states, batch_targets, povm_indices):
        full_povms = construct_povms(params)
        selected_povms = full_povms[povm_indices]
        pred_probs = simulate_measurements(selected_povms, batch_states)

        if loss_type.lower() == "mse":
            return jnp.mean((pred_probs - batch_targets[:, povm_indices]) ** 2)
        elif loss_type.lower() == "mle":
            epsilon = 1e-8
            clipped_preds = jnp.clip(pred_probs, epsilon, 1.0)
            return -jnp.mean(batch_targets[:, povm_indices] * jnp.log(clipped_preds))
        else:
            raise ValueError(f"Unsupported loss type: '{loss_type}'")

    @jax.jit
    def update(params, opt_state, batch_states, batch_targets, povm_indices):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_states, batch_targets, povm_indices)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # ---- Training loop ----
    print(f"Training system with {num_povms} POVMs using '{loss_type}' loss")

    losses, times = [], []
    start_time = time.time()
    pbar = trange(n_steps, desc="Training", leave=True)

    recon_povms_step = []

    for step in pbar:
        key = jax.random.PRNGKey(seed + step)
        state_idx = jax.random.choice(key, num_states, shape=(state_batch_size,), replace=False)
        povm_idx = jax.random.choice(key, num_povms, shape=(povm_batch_size,), replace=False)
        batch_states = probe_states[state_idx]
        batch_targets = target_probs[state_idx]

        params, opt_state, loss = update(params, opt_state, batch_states, batch_targets, povm_idx)

        current_time = time.time() - start_time
        losses.append(loss)
        times.append(current_time)
        recon_povms_step.append(construct_povms(params))

        pbar.set_description(f"Step {step}, Loss: {loss:.5e}")
        if loss < stop:
            print(f"Early stopping at step {step} with loss {loss:.6e}")
            break

    total_time = times[-1]
    print(f"Total training time: {total_time:.2f} seconds")

    recon_povms = construct_povms(params)
    final_probs = simulate_measurements(recon_povms, probe_states)

    # ---- Loss plot only ----
    if LossPlot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(losses)
        axs[0].set_title("Training Loss (Linear Scale)")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss")
        axs[0].grid(True)

        axs[1].plot(losses)
        axs[1].set_title("Training Loss (Log Scale)")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Loss")
        axs[1].set_yscale("log")
        axs[1].grid(True, which='both', linestyle='--')
        plt.suptitle(f"Loss Curves ({loss_type})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return recon_povms, final_probs, losses, times, recon_povms_step, total_time


def Stiefel_ansatz(rank, dim, num_povms, seed=None):
    # Generate rectangular random matrices
    if seed is None:
        seed = int(time.time_ns() % (2**32))
    print(f"Seed value for initial guess for Run_StiefelManiQMT_StatePovmBatched: {seed}")
    key_params = jax.random.PRNGKey(seed)

    keys = jax.random.split(key_params, num_povms)
    
    T = [jax.random.normal(k, (rank, dim)) for k in keys]

    # Make PSD matrices
    unnormalized = jnp.stack([make_psd(L) for L in T])  # (num_povms, dim, dim)

    # Normalize: enforce ∑ E_i = I
    S = jnp.sum(unnormalized, axis=0)  # (dim, dim)
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(jnp.real(eigvals), 1e-6, None)
    D_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(eigvals))
    S_inv_sqrt = eigvecs @ D_inv_sqrt @ eigvecs.T.conj()

    ansatz = jnp.array([E @ S_inv_sqrt for E in T])
    return ansatz 



def Run_StiefelManiQMT_StatePovmBatched(dimension, num_povms, rank, state_batch_size, povm_batch_size, 
                                        learning_rate, decay, n_steps, stop, probe_states, target_probs, LossPlot=True, seed=None, loss_type="mse"):
    
    @jax.jit
    def stiefel_update(params,
                   grads,
                   step_size):

        U = jnp.hstack([grads, params])
        V = jnp.hstack([params, -grads])

        prod_term = V.T.conj()@U
        invterm = jnp.eye(prod_term.shape[0]) + (step_size/2.)*prod_term
        A = step_size*(U@jnp.linalg.inv(invterm))

        B = V.T.conj()@params

        updated_params = params - A@B
        return updated_params


    @jax.jit
    def get_block(kops):
        return jnp.concatenate([*kops])


    @partial(jax.jit, static_argnums=1)
    def get_unblock(kmat, num_povms):
        return jnp.array(jnp.vsplit(kmat, num_povms))
    

    dim = dimension
    
    num_states = probe_states.shape[0]

    if seed is None:
        seed = int(time.time_ns() % (2**32))

    # initializing with valid ansatz
    # params = jnp.array([rand_unitary(dim, density=0.5).full()/np.sqrt(num_povms) for w in range(num_povms)])

    params = Stiefel_ansatz(rank, dim, num_povms, seed=None)
    params = get_block(params)

    @jax.jit
    def loss_fn(params, batch_states, batch_targets, povm_indices):
        ops = get_unblock(params, num_povms)
        full_povms = jnp.matmul(ops.conj().transpose(0, 2, 1), ops)  # Ai† Ai for each POVM
        selected_povms = full_povms[povm_indices]                # Then select batched POVMs
        pred_probs = simulate_measurements(selected_povms, batch_states)

        if loss_type.lower() == "mse":
            return jnp.mean((pred_probs - batch_targets[:, povm_indices]) ** 2)

        elif loss_type.lower() == "mle":
            epsilon = 1e-8
            clipped_preds = jnp.clip(pred_probs, epsilon, 1.0)
            return -jnp.mean(batch_targets[:, povm_indices] * jnp.log(clipped_preds))

        else:
            raise ValueError(f"Unsupported loss type: '{loss_type}'")  


    @jax.jit
    def update(params, batch_states, batch_targets, povm_indices):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_states, batch_targets, povm_indices)
        grads = jnp.conj(grads) 
        grads = grads/jnp.linalg.norm(grads)

        params = stiefel_update(params, grads, learning_rate)

        return params, loss

    print(f"Training system with {num_povms} POVMs")


    losses, times = [], []
    recon_povms_step = []

    start_time = time.time()
    pbar = trange(n_steps, desc="Training", leave=True)

    for step in pbar:
        key = jax.random.PRNGKey(seed + step)
        state_idx = jax.random.choice(key, num_states, shape=(state_batch_size,), replace=False)
        povm_idx = jax.random.choice(key, num_povms, shape=(povm_batch_size,), replace=False)
        batch_states = probe_states[state_idx]
        batch_targets = target_probs[state_idx]
    
        params, loss = update(params, batch_states, batch_targets, povm_idx)
    
        # Track loss and time
        current_time = time.time() - start_time
        losses.append(loss)
        times.append(current_time)

        # Compute current POVMs and Frobenius norm
        ops_step = get_unblock(params, num_povms)
        povms_step = jnp.matmul(ops_step.conj().transpose(0, 2, 1), ops_step)
        recon_povms_step.append(povms_step)

        pbar.set_description(f"Step {step}, Loss: {loss:.5e}")
        if loss < stop:
            print(f"Early stopping at step {step} with loss {loss:.6e}")
            break

        # decaying learning rate
        learning_rate = learning_rate*decay
    
    total_time = times[-1]
    print(f"Total training time for Run_StiefelManiQMT_StatePovmBatched: {total_time:.2f} seconds")   

    ops = get_unblock(params, num_povms)
    recon_povms = jnp.matmul(ops.conj().transpose(0, 2, 1), ops)  # Ai† Ai for each POVM  # E_i = K_i† K_i
    final_probs = simulate_measurements(recon_povms, probe_states)


    if LossPlot:

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        # Linear scale plot
        axs[0].plot(losses)
        axs[0].set_title("Training Loss (Linear Scale)")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("MSE Loss")
        axs[0].grid(True)

        # Logarithmic scale plot
        axs[1].plot(losses)
        axs[1].set_title("Training Loss (Log Scale)")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("MSE Loss")
        axs[1].set_yscale("log")
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.suptitle(" StiefelManifold GD-QDT (StatesPOVMsBatched) Loss Curves", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    return recon_povms, final_probs, losses, times, recon_povms_step, total_time


############### # --- CVXPY Quantum Measurement Tomography ---###################

def qmt_ls_cvx(probe_states, target_probs, num_povms, true_povms=None):

    num_states, dim, _ = probe_states.shape

    # pred_probs = np.zeros((num_states, num_povms))


    start_time = time.time()

    # Optimization variables
    Es = [cp.Variable((dim, dim), hermitian=True) for _ in range(num_povms)]

    # Constraints: positivity and completeness
    constraints = [E >> 0 for E in Es]
    constraints.append(sum(Es) == np.eye(dim))

    # Vectorized least-squares objective
    pred_probs_expr = cp.vstack([
        cp.hstack([cp.real(cp.trace(E @ probe_states[i])) for E in Es])
        for i in range(num_states)])  # shape: (num_states, num_povms)

    loss_expr = cp.sum(cp.square(pred_probs_expr - target_probs))

    objective = cp.Minimize(loss_expr)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    end_time = time.time()

    total_time = end_time - start_time
    # Extract solution
    recon_povms = [np.array(E.value) for E in Es]
    recon_povms_array = np.array(recon_povms, dtype=np.complex64)

    # # Evaluate metrics if true_povms is provided
    # frob_norms, cosine_sims = [], []
    # if true_povms is not None:
    #     for E, F in zip(true_povms, recon_povms_array):
    #         frob_sq = np.linalg.norm(E - F, ord="fro") ** 2
    #         dot = np.real(np.trace(E @ F))
    #         norm_E = np.linalg.norm(E, ord="fro")
    #         norm_F = np.linalg.norm(F, ord="fro")
    #         cosine_sim = 1 - ( dot / (norm_E * norm_F + 1e-12) )

    #         frob_norms.append(frob_sq)
    #         cosine_sims.append(cosine_sim)
    #     avg_frob = np.mean(frob_norms)
    #     avg_cosine = np.mean(cosine_sims)
    # else:
    #     avg_frob = avg_cosine = None

    # # Predicted probabilities (now evaluated after optimization)
    # for i in range(num_states):
    #     rho = probe_states[i]
    #     for j in range(num_povms):
    #         pred_probs[i, j] = np.real(np.trace(recon_povms_array[j] @ rho))

    # # Wasserstein distance over all probe states
    # wasserstein_vals = [
    #     wasserstein_distance(target_probs[i], pred_probs[i])
    #     for i in range(num_states)
    # ]
    # avg_wasserstein = np.mean(wasserstein_vals)

    return recon_povms_array, total_time
    



def evaluate_metrics(recon_povms, true_povms, probe_states, target_probs):
    """
    Compute evaluation metrics between reconstructed and true POVMs:
    - Avg. Frobenius norm (with std) over POVM elements
    - Avg. Wasserstein distance (with std) over probe states
    """
    num_povms = len(true_povms)

    # --- Frobenius norms per POVM element ---
    frob_norms = []
    for i in range(num_povms):
        E, F = true_povms[i], recon_povms[i]
        frob = jnp.linalg.norm(E - F, ord="fro")**2
        frob_norms.append(frob)

    frob_norms = jnp.array(frob_norms)
    frob_mean = float(jnp.mean(frob_norms))
    frob_std = float(jnp.std(frob_norms))

    # --- Wasserstein distance per probe state ---
    pred_probs = simulate_measurements(recon_povms, probe_states)
    wass_list = [
        wasserstein_distance(target_probs[i], pred_probs[i])
        for i in range(probe_states.shape[0])
    ]
    wass_array = np.array(wass_list)
    wasserstein_mean = float(np.mean(wass_array))
    wasserstein_std = float(np.std(wass_array))

    return {frob_mean, wasserstein_mean, frob_std, wasserstein_std}













#----Example Usage------#

# n_qubits = 3
# dim = 2 ** n_qubits
# num_povms = 8

# # Create random seed and PRNGKey and Generate random POVMs
# seed = int(time.time_ns() % (2**32))
# print(f"Using seed for true_povms : {seed}")
# key_true = jax.random.PRNGKey(seed)

# # true_povms = computational_basis_projectors(n_qubits=3)
# true_povms = generate_random_povms(key_true, dim, num_povms)

# probe_states = get_default_probe_states(n_qubits)
# target_probs = simulate_measurements(true_povms, probe_states)

# #------Running GD-QDT
# if __name__ == "__main__":
# recon_povms, target_probs, final_probs = Run_HonestQDT_StatePovmBatched(n_qubits, num_povms, state_batch_size=200, povm_batch_size=2, learning_rate=1e-2, n_steps=1000, stop=1e-10,
#                                                                            probe_states=probe_states, target_probs=target_probs, true_povms = true_povms,
#                                                                            LossPlot=True, seed=None, loss_type="mse")

# #--- Plotting
# compare_povms_3d_random(true_povms, recon_povms, m=3)
# plot_true_vs_reconstructed(target_probs, final_probs, num_points=50)