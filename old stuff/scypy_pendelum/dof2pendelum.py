import sympy as sym

a,b,db = sym.symbols("l th dth")

l1, l2 = sym.symbols("l(1:3)")
m1, m2 = sym.symbols("m(1:3)")
th1, th2 = sym.symbols("th(1:3)")
g = sym.symbols("g")

pth1, pth2 = sym.symbols("pth(1:3)")
dth1, dth2 = sym.symbols("dth(1:3)")

#positions

x = a*sym.sin(b)
y = - a*sym.cos(b)

x1 = x.subs([(a,l1),(b,th1)])
y1 = y.subs([(a,l1),(b,th1)])

x2 = x1 + x.subs([(a,l2),(b,th2)])
y2 = y1 + y.subs([(a,l2),(b,th2)])

dx =db*a*sym.cos(b)
dy =db*a*sym.sin(b)

dx1 = dx.subs([(a,l1),(b,th1),(db,dth1)])
dy1 = dy.subs([(a,l1),(b,th1),(db,dth1)])
dx2 = dx1 + dx.subs([(a,l2),(b,th2),(db,dth2)])
dy2 = dy1 + dy.subs([(a,l2),(b,th2),(db,dth2)])

vs1 = sym.simplify(dx1**2+dy1**2)
vs2 = sym.simplify(dx2**2+dy2**2)


print(dx1)
print(dy1)
print(dx2)
print(dy2)
print(x1)
print(y1)
print(x2)
print(y2)

K = m1*vs1/2 +m2*vs2/2
U = m1*g*y1 + m2*g*y2

L = K - U

#print(L)

p_theta1 = sym.simplify(sym.diff(L,dth1))
p_theta2 = sym.simplify(sym.diff(L,dth2))

print(p_theta1)
print(p_theta2)

p_theta = sym.Matrix([[p_theta1],[p_theta2]])
Y = sym.Matrix([[dth1],[dth2]])
pth_vec=sym.Matrix([[pth1],[pth2]])
M = p_theta.jacobian(Y)
print(M)

M_inv = sym.simplify(M**-1)

d_th = M_inv @ pth_vec

print(d_th)

H = K + U

H = sym.simplify(H.subs([(dth1,d_th[0]),(dth2,d_th[1])]))

print(H)

dHdq = sym.Matrix([0,0])
dHdp = sym.Matrix([0,0])

dHdq[0] = sym.simplify(sym.diff(H,th1)) 
dHdq[1] = sym.simplify(sym.diff(H,th2)) 

dHdp[0] = sym.simplify(sym.diff(H,pth1))
dHdp[1] = sym.simplify(sym.diff(H,pth2))

print(dHdq)
print(dHdp)

length = 1
mass = 1
gravity = 10

symplectic_vec = sym.Matrix([[dHdp],[-dHdq]])
symplectic_vec = sym.simplify(symplectic_vec.subs([(l1,length),(l2,length),(m1,mass),(m2,mass),(g,gravity)]))
print(symplectic_vec)

func = sym.lambdify([th1,th2,pth1,pth2],symplectic_vec)
H = sym.simplify(H.subs([(l1,length),(l2,length),(m1,mass),(m2,mass),(g,gravity)]))
H_func= sym.lambdify([th1,th2,pth1,pth2],H)
print(func(0,0,1,1))
print(H_func(0,0,1,1))

