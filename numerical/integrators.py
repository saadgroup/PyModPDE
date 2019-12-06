from numpy import zeros
from scipy.optimize import fsolve

def forward_euler(rhsfun, phi0, times, bcfun):
    """
    foward_euler - implements the forward euler algorithm
    :param rhsfun: the function that computes the RHS of the PDE(s) that we are solving.  Should take one arguments: the value of the solution variables at which the RHS should be evaluated.
    :param phi0: the solution vector at the first element in "times"
    :param times: the vector of times at which we should evaluate the RHS and obtain the solution
    :param bcfun: the boundary condition function
    :return: the solution variables at each of the requested times as a matrix with each time in a row and each variable in a column
    """

    size = phi0.size +2 # +2 is for the ghost cells
    phi = zeros((times.size,size))  # allocate storage for the results
    phi[0, 1:-1] = phi0  # store the initial condition in the results

    bcfun(phi[0, :]) # apply bcs for the initial condition

    for i in range(times.size - 1):
        dt = times[i + 1] - times[i]

        phi[i+1,:] = phi[i,:] + dt * rhsfun(phi[i,:]) # time advance

        bcfun(phi[i+1,:]) # apply bcs

    return phi


def backward_euler(rhsfun, phi0, times, bcfun):
  """
  backward_euler - implements the backward euler algorithm
  :param rhsFun: the function that computes the RHS of the PDE(s) that we are solving.  Should take one arguments: the value of the solution variables at which the RHS should be evaluated.
  :param phi0: the solution vector at the first element in "times"
  :param times: the vector of times at which we should evaluate the RHS and obtain the solution
  :param bcfun: the boundary condition function
  :return: the solution variables at each of the requested times as a matrix with each time in a row and each variable in a column
  """

  size = phi0.size + 2  # +2 is for the ghost cells
  phi = zeros((times.size, size))  # allocate storage for the results
  phi[0, 1:-1] = phi0  # store the initial condition in the results

  bcfun(phi[0, :]) # apply bcs for the initial condition

  for i in range(times.size - 1):
      dt = times[i + 1] - times[i]

      def resfun(phinew):
          return phinew - phi[i, :] - dt * rhsfun(phinew)

      phi[i + 1, :] = fsolve(resfun, phi[i, :]) # time advance

      bcfun(phi[i + 1,:])


  return phi


def crank_nicholson(rhsfun, phi0, times,bcfun):
    """
    crank_nicholson - implements the Crank-Nicholson algorithm
    :param rhsFun: the function that computes the RHS of the PDE(s) that we are solving.  Should take one arguments: the value of the solution variables at which the RHS should be evaluated.
    :param phi0: the solution vector at the first element in "times"
    :param times: the vector of times at which we should evaluate the RHS and obtain the solution
    :param bcfun: the boundary condition function
    :return: the solution variables at each of the requested times as a matrix with each time in a row and each variable in a column
    """

    size = phi0.size + 2  # +2 is for the ghost cells
    phi = zeros((times.size, size))  # allocate storage for the results
    phi[0, 1:-1] = phi0  # store the initial condition in the results

    bcfun(phi[0, :])  # apply bcs for the initial condition

    for i in range(times.size - 1):
        dt = times[i + 1] - times[i]
        rhsold = rhsfun(phi[i,:])
        def resfun(phinew):
            return (phinew - phi[i, :])/dt - 0.5 * ( rhsold + rhsfun(phinew) )

        phi[i + 1, :] = fsolve(resfun, phi[i, :])

        bcfun(phi[i + 1, :])

    return phi

