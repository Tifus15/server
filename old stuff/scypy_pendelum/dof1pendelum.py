import sympy as sym

l = sym.symbols("l")
m = sym.symbols("m")
g = sym.symbols("g")
th = sym.symbols("th")

pth = sym.symbols("pth")
dth = sym.symbols("dth")

x = l*sym.sin(th)
y = - l*sym.cos(th)

dx = dth*l*sym.cos(th)
dy = dth*l*sym.sin(th)

vs = sym.simplify(dx**2+dy**2)
print(vs)

K = m*vs/2
U = m*g*y

L = K - U

p_th =sym.simplify(sym.diff(L,dth))

M = sym.diff(p_th,dth)
d_th = pth/M
print(d_th)

H = K + U
H = H.subs([(dth,d_th)])
print(H)

dtTheta =sym.diff(H,pth)
dtpTheta = -sym.diff(H,th)
print(dtTheta)
print(dtpTheta)

