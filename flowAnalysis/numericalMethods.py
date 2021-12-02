
class RungeKutta:
    def __init__(self):
        pass

    def dydx(self, x, y):
        "return derivate on one variable w.r.t. another"
        return 2*(x + 1e-6)

    def iterFourthOrder(self, x0, y0, x, h):
        "finds value of y for a given x"
        n = round((x - x0)/h)
        x = x0
        y = y0
        # iterate 
        for i in range(0, n+1):
            # get coeffs
            k1 = self.dydx(x, y)*h
            k2 = self.dydx(x + 0.5*h, y + 0.5*k1)*h
            k3 = self.dydx(x + 0.5*h, y + 0.5*k2)*h
            k4 = self.dydx(x + h, y + k3)*h
            # update x , y for new round
            y += (1./6.)* (k1 + 2*k2 + 2*k3+k4)
            x += h
        
        return y

if __name__=="__main__":
    x0 = 0
    y0 = 0
    x = 3
    h = 0.001

    algo = RungeKutta()
    print(algo.iterFourthOrder(x0, y0, x, h))
