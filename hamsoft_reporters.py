  def canonical_eom(self):
        q = np.asarray(self.q, dtype=float)
        p = np.asarray(self.p, dtype=float)
        m = self._get_masses()
        G, k_soft, mu_soft, _ = self._get_constants()
        n, d = int(q.shape[0]), int(q.shape[1])

        # qdot = p/m
        qdot = np.zeros((n, d), dtype=float)
        i = 0
        while i < n:
            mi = m[i]
            if mi != 0.0:
                qdot[i] = p[i] / mi
            else:
                qdot[i] = 0.0 * p[i]
            i = i + 1

        # Forces and dV/dε at CURRENT ε (no override in EOM; Prompt S3)
        forces, dVgrav_deps = self._evaluate_gravity_forces_and_dVdeps(
            use_override=False, want_potential=False
        )

        # Barrier derivative at current ε
        deps_bar = self._dSbar_deps(float(self.epsilon))

        # ε⋆ and its gradient at q (same path as S-flow)
        eps_star, grad_eps_star = self._eps_star_and_grad()

        # pdot = -∇V + k_soft(ε-ε⋆)∇ε⋆
        # Here -∇V_grav = F_grav; barrier has no q-gradient
        Delta = float(self.epsilon) - float(eps_star)
        pdot = forces + k_soft * Delta * grad_eps_star

        # epsdot = π/μ_soft
        if mu_soft != 0.0:
            epsdot = float(self.pi) / mu_soft
        else:
            epsdot = 0.0

        # pidot = -∂εV(q,ε) - k_soft(ε-ε⋆)
        # ∂εV = dVgrav/dε + dSbar/dε
        pidot = -(dVgrav_deps + deps_bar) - k_soft * Delta

        return qdot, pdot, epsdot, pidot

