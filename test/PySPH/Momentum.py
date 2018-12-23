class MomentumEquation():
    def __init__(self,
                 alpha=1.0, beta=1.0, gx=0.0, gy=0.0, gz=0.0,
                 tensile_correction=False):
        self.alpha = alpha
        self.beta = beta
        self.gx = gx
        self.gy = gy
        self.gz = gz

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, WIJ, WDP, d_dt_cfl):

        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        tmpi = d_p[d_idx]*rhoi21
        tmpj = s_p[s_idx]*rhoj21

        # gradient and correction terms
        tmp = (tmpi + tmpj)

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        #d_aw[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[2]