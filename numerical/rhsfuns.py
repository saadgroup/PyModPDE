import numpy as np

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
    def periodicbcfun():
        def bcfun(phi):
            phi[0] = phi[-2]
            phi[-1] = phi[1]

        return bcfun
