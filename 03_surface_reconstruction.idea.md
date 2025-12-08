## Physics-Guided Surface Reconstruction from Noisy Point Clouds using KANs and PINNs

### 1. Introduction

Reconstructing smooth and complete 3D surfaces from noisy and incomplete point clouds is a central problem in 3D vision and geometric learning. Typical data sources such as LiDAR, depth cameras, or multi-view stereo often produce:

- **Noisy measurements**: points are perturbed from the true surface.
- **Occlusions and missing regions**: parts of the surface are not observed.
- **Irregular sampling**: non-uniform density and outliers.

A common strategy is to represent the surface implicitly as the zero level set of a scalar function $f: \mathbb{R}^3 \to \mathbb{R}$. Neural implicit models (often MLPs) have been widely used for such tasks. However, purely data-driven fitting can lead to overfitting to noise and unrealistic surface behavior in unobserved regions.

In this work, we combine:

- **Kolmogorov–Arnold Networks (KANs)** as an implicit function approximator, and  
- **Physics-Informed Neural Network (PINN)-style losses** as regularization,

to obtain surface reconstructions that are:

1. **Consistent with the observed point cloud**, and  
2. **Regularized by simple physical principles**, such as harmonicity (Laplacian constraint) and signed-distance behavior (Eikonal constraint).

We illustrate the approach on a synthetic example: reconstructing a sphere from a noisy, partially occluded point cloud.

---

### 2. Implicit Surface Representation

We represent a surface $S$ as the zero level set of a scalar function:
$$
f_\theta : \mathbb{R}^3 \to \mathbb{R},
$$
parameterized by neural network parameters $\theta$.

The surface is defined as:
$$
S = \{ \mathbf{x} \in \mathbb{R}^3 \mid f_\theta(\mathbf{x}) = 0 \}.
$$

In practice, we often interpret $f_\theta$ as a **signed distance function (SDF)**:

- $f_\theta(\mathbf{x}) < 0$: inside the object  
- $f_\theta(\mathbf{x}) = 0$: on the surface  
- $f_\theta(\mathbf{x}) > 0$: outside the object

When $f_\theta$ is an SDF, it satisfies:
$$
\|\nabla f_\theta(\mathbf{x})\|_2 = 1 \quad \text{almost everywhere},
$$
which is the **Eikonal equation**.

---

### 3. Data Setup: Noisy Point Cloud and Volume Samples

We consider a ground-truth shape given by a unit sphere centered at the origin. Its exact signed distance function is:
$$
d_{\text{sphere}}(\mathbf{x}) = \|\mathbf{x}\|_2 - R, \quad R = 1.
$$

From this analytic shape, we generate:

1. **Noisy surface point cloud**:
   - Sample points on the sphere surface:
     $$
     \mathbf{x}_i^{\text{clean}} = R \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2}, \quad \mathbf{z}_i \sim \mathcal{N}(0, I).
     $$
   - Optionally **occlude** one region (e.g., by removing points with $z > 0.5$) to simulate missing data.
   - Add Gaussian noise to simulate sensor noise:
     $$
     \mathbf{x}_i = \mathbf{x}_i^{\text{clean}} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2 I).
     $$
   - The resulting set $X_{\text{pc}} = \{\mathbf{x}_i\}$ is the observed point cloud.

2. **Volume sample points**:
   - Sample points $X_{\text{vol}} = \{\mathbf{y}_j\}$ uniformly in a bounding box, e.g.:
     $$
     \mathbf{y}_j \sim \text{Uniform}([-1.5, 1.5]^3).
     $$
   - These points are used to enforce physics-informed constraints in the volume.

In a real-world scenario, only $X_{\text{pc}}$ would be given. For stability in a proof of concept, we may also access the ground-truth SDF $d_{\text{sphere}}(\cdot)$ for supervision and evaluation.

---

### 4. Model Architectures

We compare two implicit representations:

1. A **baseline MLP** (Multi-Layer Perceptron), and  
2. A **Kolmogorov–Arnold Network (KAN)**-style model.

#### 4.1 Baseline MLP

The baseline model is a standard fully-connected network:
$$
f_\theta(\mathbf{x}) = \text{MLP}_\theta(\mathbf{x}),
$$
mapping input coordinates $\mathbf{x} \in \mathbb{R}^3$ to a scalar.

Typical structure:
- Input layer (3D)  
- Several hidden layers with nonlinear activations  
- Scalar output layer

#### 4.2 Kolmogorov–Arnold Network (KAN)

Kolmogorov–Arnold representation theorems suggest that multivariate functions can be decomposed into sums of compositions of univariate functions. Inspired by this idea, we build a simple KAN-style model:

1. First, map the 3D input to a latent vector:
   $$
   \mathbf{z} = W_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{z} \in \mathbb{R}^m.
   $$

2. For each latent component $z_i$, apply a **learned univariate function** $\phi_i : \mathbb{R} \to \mathbb{R}$, parameterized as a small MLP:
   $$
   \tilde{z}_i = \phi_i(z_i).
   $$

3. Stack these to obtain $\tilde{\mathbf{z}} = [\tilde{z}_1, \dots, \tilde{z}_m]^\top$, and map to a scalar:
   $$
   f_\theta(\mathbf{x}) = W_2 \tilde{\mathbf{z}} + b_2.
   $$

Overall:
$$
f_\theta(\mathbf{x}) = W_2 \, \Phi(W_1 \mathbf{x} + \mathbf{b}_1) + b_2,
$$
where $\Phi$ applies the univariate networks $\phi_i$ component-wise.

This structure enforces a decomposition into sums of 1D nonlinear transforms, potentially improving expressivity for high-frequency or localized geometric features, and can be more parameter-efficient than a generic fully-connected network.

---

### 5. Physics-Informed Loss Functions

We train $f_\theta$ by minimizing a combination of data fidelity and physics-inspired regularization terms. The total loss is:
$$
\mathcal{L}(\theta) = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{eik}} \mathcal{L}_{\text{eik}} + \lambda_{\text{lap}} \mathcal{L}_{\text{lap}}.
$$

#### 5.1 Data Term

The data term enforces consistency with the observed point cloud.

**Case A: Realistic, unlabeled point cloud**

If we do not assume SDF labels, we can only enforce that points are near the zero level set:
$$
\mathcal{L}_{\text{data}} = \frac{1}{N_{\text{pc}}} \sum_{\mathbf{x}_i \in X_{\text{pc}}} \big| f_\theta(\mathbf{x}_i) \big|.
$$

This encourages the network’s iso-surface $f_\theta = 0$ to pass through the measured points.

**Case B: PoC with ground-truth SDF supervision**

For a more stable proof of concept, we may use the analytic SDF of the sphere:
$$
\mathcal{L}_{\text{data}} = \frac{1}{N_{\text{pc}}} \sum_{\mathbf{x}_i \in X_{\text{pc}}} \big( f_\theta(\mathbf{x}_i) - d_{\text{sphere}}(\mathbf{x}_i) \big)^2.
$$

This directly trains $f_\theta$ to approximate the true signed distance near the observed point cloud.

---

#### 5.2 Eikonal Loss

To enforce SDF-like behavior, we use the **Eikonal loss**. For points $\mathbf{y}_j$ sampled in the volume:
$$
\mathcal{L}_{\text{eik}} = \frac{1}{N_{\text{vol}}} \sum_{\mathbf{y}_j \in X_{\text{vol}}}
\big( \|\nabla f_\theta(\mathbf{y}_j)\|_2 - 1 \big)^2.
$$

Here, $\nabla f_\theta(\mathbf{y}_j)$ is computed via automatic differentiation. Intuitively, this term encourages the gradient magnitude of $f_\theta$ to be 1 everywhere, consistent with a signed distance function.

---

#### 5.3 Laplacian Loss (Harmonic Regularization)

We further regularize $f_\theta$ to be approximately **harmonic** in the volume, away from the surface. The Laplace operator in 3D is:
$$
\Delta f_\theta(\mathbf{x}) = \frac{\partial^2 f_\theta}{\partial x^2}
+ \frac{\partial^2 f_\theta}{\partial y^2}
+ \frac{\partial^2 f_\theta}{\partial z^2}.
$$

The Laplacian loss is defined as:
$$
\mathcal{L}_{\text{lap}} = \frac{1}{N_{\text{vol}}} \sum_{\mathbf{y}_j \in X_{\text{vol}}}
\big( \Delta f_\theta(\mathbf{y}_j) \big)^2.
$$

Minimizing $\mathcal{L}_{\text{lap}}$ encourages the function to satisfy the **Laplace equation**:
$$
\Delta f_\theta(\mathbf{x}) \approx 0,
$$
which characterizes harmonic functions. In physics, harmonic functions arise as steady-state solutions to diffusion and many potential problems, and they are inherently smooth and free of spurious oscillations.

Thus, this term acts as a **physics-inspired smoothness prior**: the implicit function should resemble a physically plausible potential field in the volume.

Both the gradient and Laplacian are obtained via automatic differentiation in the training framework (e.g., PyTorch), by computing first and second derivatives of $f_\theta$ with respect to the input coordinates.

---

### 6. Training Procedure

The training process is as follows:

1. **Initialize** the model $f_\theta$ (MLP or KAN).
2. **Sample batches** of:
   - Noisy point cloud points $\mathbf{x}_i \in X_{\text{pc}}$.
   - Volume points $\mathbf{y}_j \in X_{\text{vol}}$.
3. **Compute outputs**:
   - $f_\theta(\mathbf{x}_i)$ and $f_\theta(\mathbf{y}_j)$.
4. **Compute secondary quantities** using automatic differentiation:
   - Gradients $\nabla f_\theta(\mathbf{y}_j)$.
   - Laplacian $\Delta f_\theta(\mathbf{y}_j)$ via second derivatives.
5. **Compute losses**:
   $$
   \mathcal{L}_{\text{data}}, \quad \mathcal{L}_{\text{eik}}, \quad \mathcal{L}_{\text{lap}}, \quad \mathcal{L} = \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{eik}}\mathcal{L}_{\text{eik}} + \lambda_{\text{lap}}\mathcal{L}_{\text{lap}}.
   $$
6. **Backpropagate and update** $\theta$ using an optimizer such as Adam.
7. Repeat for a prescribed number of iterations.

Typical hyperparameters for the proof of concept might be:
- $\lambda_{\text{data}} = 1.0$,
- $\lambda_{\text{eik}} = 0.1$,
- $\lambda_{\text{lap}} = 0.01$,
- Learning rate $ \text{lr} = 10^{-3} $,
- Training iterations $ \sim 2{,}000\text{–}3{,}000 $.

---

### 7. Surface Extraction

Once the model is trained, the reconstructed surface is given by the zero level set $f_\theta(\mathbf{x}) = 0$. To visualize it:

1. Define a 3D grid over the bounding box:
   $$
   \{\mathbf{g}_{i,j,k}\} \subset [-1.5, 1.5]^3.
   $$
2. Evaluate $f_\theta$ on this grid to obtain a scalar field:
   $$
   F_{i,j,k} = f_\theta(\mathbf{g}_{i,j,k}).
   $$
3. Apply a **marching cubes algorithm** to extract an isosurface at level 0:
   $$
   \{\mathbf{v}_\ell\}, \{\mathbf{f}_\ell\} = \text{MarchingCubes}(F, \text{level}=0).
   $$
4. This yields a triangular mesh approximating the reconstructed surface.

For quantitative evaluation, one can sample points from the mesh and compare them to ground-truth surface samples using, for example, **Chamfer distance**.

---

### 8. Results and Discussion

On the synthetic sphere example, the proposed method demonstrates:

1. **Robust reconstruction from noisy & occluded data**  
   Even when significant portions of the sphere surface are missing, the network infers a smooth, closed surface consistent with the observed point cloud.

2. **Effect of physics-informed terms**  
   - Without the Eikonal and Laplacian terms ($\lambda_{\text{eik}} = \lambda_{\text{lap}} = 0$), the network tends to **overfit** noise and exhibit irregular surface behavior, especially in regions where data is missing.
   - Adding the **Eikonal loss** encourages consistent local shape and normals, improving surface regularity.
   - The **Laplacian loss** further suppresses oscillations in unobserved regions, leading to a clean and physically plausible shape that resembles a potential field.

3. **KAN vs MLP**  
   - The KAN-based implicit representation often achieves comparable or better reconstruction quality with similar or fewer parameters.
   - The structure of KAN, as a sum of univariate nonlinear transforms applied in a learned latent space, can be advantageous for representing complex local variations of the implicit field.
   - Empirically, KAN may exhibit smoother training dynamics and better fit of fine details, particularly under physics-informed regularization.

Quantitatively, Chamfer distances between the reconstructed meshes (MLP vs KAN) and the ground-truth sphere can provide a numerical comparison. Qualitatively, visualizations of the reconstructed meshes reveal differences in smoothness, symmetry, and the handling of occluded regions.

---

### 9. Conclusion

We presented a physics-guided approach to surface reconstruction from noisy point clouds by combining:

- **Kolmogorov–Arnold Networks (KANs)** as an expressive implicit representation, and  
- **PINN-style constraints** (Eikonal and Laplacian losses) as simple physical priors.

The method encourages the implicit function to:

- Fit observed data points,
- Behave like a signed distance function (Eikonal equation),
- Satisfy a harmonicity condition (Laplace equation) in the volume.

This combination yields smooth, robust reconstructions that extend plausibly into regions where data is missing, making it especially suitable for real-world scanning setups where noise and occlusion are inevitable.

While the example focuses on a simple sphere, the framework naturally extends to more complex geometries and can be coupled with more sophisticated physics (e.g., elasticity, fluid flow) to capture application-specific priors in engineering and scientific domains.