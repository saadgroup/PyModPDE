import numpy as np

class RhsFuncs:
    def __init__(self,advection_str,diffusion_str):
        self.advection_str=advection_str
        self.diffusion_str=diffusion_str

    def __call__(self, dx, advection_vel,diffusion_coef):
        rhs_result = ''
        if self.advection_str == 'Forward Difference':
            rhs_result+= 'Advection.advection_FD({},{})(phi)'.format(dx,advection_vel)
        elif self.advection_str == 'Backward Difference':
            rhs_result += 'Advection.advection_BD({},{})(phi)'.format(dx,advection_vel)
        elif self.advection_str == 'Central Difference':
            rhs_result += 'Advection.advection_CD({},{})(phi)'.format(dx,advection_vel)

        rhs_result+='+'

        if self.diffusion_str == 'Central Difference':
            rhs_result += 'Diffusion.diffusion_CD({},{})(phi)'.format(dx,diffusion_coef)
        elif self.diffusion_str == 'Backward Difference':
            rhs_result += 'Diffusion.diffusion_BD({},{})(phi)'.format(dx,diffusion_coef)
        elif self.diffusion_str == 'Forward Difference':
            rhs_result += 'Diffusion.diffusion_FD({},{})(phi)'.format(dx,diffusion_coef)

        return lambda phi: eval(rhs_result)

    @staticmethod
    def periodicbcfun():
        def bcfun(phi):
            phi[0] = phi[-2]
            phi[-1] = phi[1]

        return bcfun


class Advection:
    @staticmethod
    def advection_BD(dx,c):
        def rhs(phi):
            rhs_result = np.zeros(phi.size)
            rhs_result[1:-1] =  -c*(phi[1:-1] - phi[:-2])/dx
            return rhs_result
        return rhs

    @staticmethod
    def advection_CD(dx, c):
        def rhs(phi):
            rhs_result = np.zeros(phi.size)
            rhs_result[1:-1] = -c * (phi[2:] - phi[:-2]) / dx /2
            return rhs_result
        return rhs

    @staticmethod
    def advection_FD(dx, c):
        def rhs(phi):
            rhs_result = np.zeros(phi.size)
            rhs_result[1:-1] = -c * (phi[2:] - phi[1:-1]) / dx
            return rhs_result
        return rhs

    @staticmethod
    def periodicbcfun():
        def bcfun(phi):
            phi[0]=phi[-2]
            phi[-1] = phi[1]
        return bcfun


class Diffusion:
    @staticmethod
    def diffusion_CD(dx,a):
        def rhs(phi):
            rhs_result = np.zeros(phi.size)
            rhs_result[1:-1] =  a*(phi[2:] -2 * phi[1:-1] + phi[:-2])/dx/dx
            return rhs_result
        return rhs

    @staticmethod
    def diffusion_FD(dx, a):
        def rhs(phi):
            rhs_result = np.zeros(phi.size+2)
            temp_phi = np.zeros_like(rhs_result)
            temp_phi[2:-2] = phi[1:-1]
            Diffusion.periodicbcfunwidestencile()(temp_phi)
            rhs_result[2:-2] = a * (temp_phi[2:-2] - 2 * temp_phi[3:-1] + temp_phi[4:]) / dx / dx
            return rhs_result[1:-1]
        return rhs

    @staticmethod
    def diffusion_BD(dx, a):
        def rhs(phi):
            rhs_result = np.zeros(phi.size + 2)
            temp_phi = np.zeros_like(rhs_result)
            temp_phi[2:-2] = phi[1:-1]
            Diffusion.periodicbcfunwidestencile()(temp_phi)
            rhs_result[2:-2] = a * (temp_phi[2:-2] - 2 * temp_phi[1:-3] + temp_phi[:-4]) / dx / dx
            return rhs_result[1:-1]
        return rhs

    @staticmethod
    def periodicbcfun():
        def bcfun(phi):
            phi[0] = phi[-2]
            phi[-1] = phi[1]

        return bcfun

    @staticmethod
    def periodicbcfunwidestencile():
        def bcfun(phi):
            phi[0] = phi[-3]
            phi[1] = phi[-4]
            phi[-1] = phi[2]
            phi[-2] = phi[3]

        return bcfun
